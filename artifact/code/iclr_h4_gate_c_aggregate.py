#!/usr/bin/env python3
"""Frozen H4 Gate C aggregation, blocked on full v2 integrity."""

from __future__ import annotations

import argparse
import hashlib
import json
import statistics
from pathlib import Path
from typing import Any

import numpy as np

from future_qc.iclr_h4.estimands import image_set_decomposition


ROOT = Path(__file__).resolve().parents[1]
MODELS = ("detr_r50_500", "dino_r50_4s_12e")
DOSES = (0.0, 0.25, 0.5, 0.75, 1.0)
INTERMEDIATE = (0.25, 0.5, 0.75)
DEADZONE = 1e-4
BOOTSTRAP_DRAWS = 10000
INTEGRITY = ROOT / "artifacts/iclr_h4/H4_GATE_C_D_FULL_INTEGRITY_AUDIT_V2.json"
CONFIG = ROOT / "configs/fq_iclr_h4_gate_c_aggregation_v1.json"
OUTPUT = ROOT / "artifacts/iclr_h4/H4_GATE_C_SCIENTIFIC_AGGREGATE.json"


def load(path: Path) -> dict[str, Any]:
    value = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise RuntimeError(f"JSON object required: {path}")
    return value


def sha256_file(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def arm(kind: str, dose: float) -> str:
    return f"{kind}_alpha_{str(dose).replace('.', 'p')}"


def effects(receipt: dict[str, Any], kind: str, dose: float) -> dict[str, float]:
    baseline = receipt["arm_measurements"]["baseline"]
    changed = receipt["arm_measurements"][arm(kind, dose)]
    decomposition = image_set_decomposition(
        sham_quality=baseline["quality_matrix"],
        intervention_quality=changed["quality_matrix"],
        focal_prediction=int(receipt["pair"]["competitor"]),
        native_intervention_quality=receipt["native_measurements"][arm(kind, dose)][
            "quality_matrix"
        ],
    )
    return {
        "local": float(changed["fixed_competitor_quality"])
        - float(baseline["fixed_competitor_quality"]),
        "target": float(changed["target_set_quality"])
        - float(baseline["target_set_quality"]),
        "image": float(decomposition["utilities"]["u3"])
        - float(decomposition["utilities"]["u0"]),
        "native": float(decomposition["total"]),
        "closure": float(decomposition["closure_error"]),
    }


def bootstrap_ci(
    values: list[float], strata: list[str], seed_text: str
) -> list[float]:
    digest = hashlib.sha256(seed_text.encode()).digest()
    rng = np.random.default_rng(int.from_bytes(digest[:8], "big"))
    array = np.asarray(values, dtype=np.float64)
    draws = np.zeros(BOOTSTRAP_DRAWS, dtype=np.float64)
    unique = sorted(set(strata))
    for start in range(0, BOOTSTRAP_DRAWS, 500):
        stop = min(start + 500, BOOTSTRAP_DRAWS)
        subtotal = np.zeros(stop - start, dtype=np.float64)
        count = 0
        for label in unique:
            indices = np.flatnonzero(np.asarray(strata) == label)
            sampled = rng.choice(indices, size=(stop - start, len(indices)), replace=True)
            subtotal += array[sampled].sum(axis=1)
            count += len(indices)
        draws[start:stop] = subtotal / count
    return [float(np.quantile(draws, 0.025)), float(np.quantile(draws, 0.975))]


def summarize(values: list[float], strata: list[str], seed: str) -> dict[str, Any]:
    return {
        "mean": statistics.fmean(values),
        "median": statistics.median(values),
        "bootstrap_95": bootstrap_ci(values, strata, seed),
    }


def aggregate() -> dict[str, Any]:
    integrity = load(INTEGRITY)
    config = load(CONFIG)
    if (
        integrity.get("status") != "PASS_H4_GATE_C_D_FULL_INTEGRITY_NO_AGGREGATE"
        or integrity.get("aggregation_allowed") is not True
        or integrity.get("scientific_effect_readout") is not False
        or config.get("status") != "FROZEN_BEFORE_GATE_C_D_OUTCOME_READOUT"
        or config.get("bootstrap_draws") != BOOTSTRAP_DRAWS
    ):
        raise RuntimeError("Gate C integrity/config gate failure")
    model_results = {}
    passing_candidates = []
    population = load(ROOT / "artifacts/iclr_h4/H4_D_FULL_GT_MANIFEST.json")
    stratum_by_id = {row["image_id"]: row["stratum"] for row in population["rows"]}
    for model in MODELS:
        receipts = [
            load(path)
            for path in sorted(
                (ROOT / f"artifacts/iclr_h4/d_gate_c_raw_v2/{model}").glob("*.json")
            )
        ]
        if len(receipts) != integrity["models"][model]["receipt_count"]:
            raise RuntimeError(f"{model} receipt count changed")
        strata = [stratum_by_id[row["image_id"]] for row in receipts]
        dose_results = {}
        for dose in DOSES:
            target = [effects(row, "target", dose) for row in receipts]
            control = [effects(row, "control", dose) for row in receipts]
            fields = {}
            for name in ("local", "target", "image", "native"):
                fields[f"target_{name}"] = summarize(
                    [row[name] for row in target], strata, f"{model}|{dose}|target|{name}"
                )
                fields[f"control_{name}"] = summarize(
                    [row[name] for row in control], strata, f"{model}|{dose}|control|{name}"
                )
            target_joint = [
                float(row["local"] > DEADZONE and row["target"] < -DEADZONE)
                for row in target
            ]
            control_joint = [
                float(row["local"] > DEADZONE and row["target"] < -DEADZONE)
                for row in control
            ]
            difference = [a - b for a, b in zip(target_joint, control_joint, strict=True)]
            joint = {
                "target_rate": statistics.fmean(target_joint),
                "control_rate": statistics.fmean(control_joint),
                "paired_rate_difference": summarize(
                    difference, strata, f"{model}|{dose}|joint_difference"
                ),
            }
            local_pass = fields["target_local"]["bootstrap_95"][0] > 0
            target_harm = fields["target_target"]["bootstrap_95"][1] < 0
            image_harm = fields["target_image"]["bootstrap_95"][1] < 0
            control_pass = joint["paired_rate_difference"]["bootstrap_95"][0] > 0
            candidate_pass = (
                dose in INTERMEDIATE
                and local_pass
                and (target_harm or image_harm)
                and control_pass
            )
            if candidate_pass:
                passing_candidates.append({"model": model, "dose": dose})
            dose_results[str(dose)] = {
                "summaries": fields,
                "joint_reversal": joint,
                "maximum_absolute_closure_error": max(
                    abs(row["closure"]) for row in target + control
                ),
                "gate_checks": {
                    "intermediate_dose": dose in INTERMEDIATE,
                    "local_mean_ci_positive": local_pass,
                    "target_or_image_harm_ci_negative": target_harm or image_harm,
                    "paired_joint_difference_ci_positive": control_pass,
                },
                "candidate_pass": candidate_pass,
            }
        model_results[model] = {"n": len(receipts), "doses": dose_results}
    gate_pass = bool(passing_candidates)
    result = {
        "schema_version": "fq.iclr.h4.gate_c_scientific_aggregate.v1",
        "status": "GO_H4D_NOT_MASK_ARTIFACT" if gate_pass else "STOP_GENERALIZATION_HARD_MASK_ONLY",
        "integrity_audit_sha256": sha256_file(INTEGRITY),
        "aggregation_config_sha256": sha256_file(CONFIG),
        "bootstrap_draws": BOOTSTRAP_DRAWS,
        "deadzone": DEADZONE,
        "models": model_results,
        "passing_candidates": passing_candidates,
        "gate_c_pass": gate_pass,
        "scientific_effect_readout": True,
        "V_F_read": False,
        "next_gate": "H4_GATE_D_ARCHITECTURE_BOUNDARY" if gate_pass else "STOP",
    }
    OUTPUT.write_text(json.dumps(result, indent=2) + "\n", encoding="utf-8")
    return result


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--execute", action="store_true")
    args = parser.parse_args()
    if not args.execute:
        result = {
            "status": "PASS_H4_GATE_C_AGGREGATOR_PREFLIGHT_NO_RECEIPT_READ",
            "scientific_effect_readout": False,
        }
    else:
        result = aggregate()
    print(json.dumps(result, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
