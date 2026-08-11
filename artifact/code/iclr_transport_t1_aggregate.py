#!/usr/bin/env python3
"""Frozen T1 CPU aggregation over existing hard and mass receipts."""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import statistics
from pathlib import Path
from typing import Any

import numpy as np


ROOT = Path(__file__).resolve().parents[1]
CONFIG = ROOT / "configs/fq_iclr_transport_t1_aggregation_v1.json"
MAPPING = ROOT / "artifacts/iclr_transport/T1_REALIZED_DOSE_MAPPING.json"
HARD_INTEGRITY = ROOT / "artifacts/iclr_h4/H4_D_FULL_INTEGRITY_AUDIT.json"
MASS_INTEGRITY = ROOT / "artifacts/iclr_h4/H4_GATE_C_D_FULL_INTEGRITY_AUDIT_V2.json"
OUTPUT = ROOT / "artifacts/iclr_transport/T1_PRE_READOUT_LOCALIZATION.json"
MODELS = ("detr_r50_500", "dino_r50_4s_12e")
FIELDS = (
    "local",
    "focal",
    "spillover",
    "fixed",
    "matching",
    "rematched",
    "selection",
    "native",
)
PRE = ("local", "focal", "spillover", "fixed")
READOUT = ("matching", "selection")


def load(path: Path) -> dict[str, Any]:
    value = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise RuntimeError(f"JSON object required: {path}")
    return value


def sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def fixed_utility(measurement: dict[str, Any], assignment: list[int]) -> float:
    matrix = measurement["quality_matrix"]
    total = sum(
        float(matrix[prediction][gt])
        for gt, prediction in enumerate(assignment)
        if prediction >= 0
    )
    return total / len(assignment)


def effect(receipt: dict[str, Any], changed_name: str) -> dict[str, float]:
    baseline = receipt["arm_measurements"]["baseline"]
    changed = receipt["arm_measurements"][changed_name]
    assignment = [int(value) for value in baseline["optimal_assignment"]]
    competitor = int(receipt["pair"]["competitor"])
    fixed = fixed_utility(changed, assignment) - float(baseline["image_set_utility"])
    focal = sum(
        float(changed["quality_matrix"][competitor][gt])
        - float(baseline["quality_matrix"][competitor][gt])
        for gt, prediction in enumerate(assignment)
        if prediction == competitor
    ) / len(assignment)
    rematched = float(changed["image_set_utility"]) - float(
        baseline["image_set_utility"]
    )
    native = float(receipt["native_measurements"][changed_name]["image_set_utility"]) - float(
        receipt["native_measurements"]["baseline"]["image_set_utility"]
    )
    return {
        "local": float(changed["fixed_competitor_quality"])
        - float(baseline["fixed_competitor_quality"]),
        "focal": focal,
        "spillover": fixed - focal,
        "fixed": fixed,
        "matching": rematched - fixed,
        "rematched": rematched,
        "selection": native - rematched,
        "native": native,
    }


def bootstrap(
    values: np.ndarray,
    strata: np.ndarray,
    *,
    seed: str,
    draws: int,
) -> tuple[np.ndarray, np.ndarray, float]:
    groups = [np.flatnonzero(strata == label) for label in sorted(set(strata))]
    rng = np.random.default_rng(
        int.from_bytes(hashlib.sha256(seed.encode()).digest()[:8], "big")
    )
    sampled_means = np.empty((draws, values.shape[1]), dtype=np.float64)
    for start in range(0, draws, 250):
        stop = min(draws, start + 250)
        subtotal = np.zeros((stop - start, values.shape[1]), dtype=np.float64)
        for group in groups:
            sampled = rng.choice(group, size=(stop - start, len(group)), replace=True)
            subtotal += values[sampled].sum(axis=1)
        sampled_means[start:stop] = subtotal / len(values)
    means = values.mean(axis=0)
    se = sampled_means.std(axis=0, ddof=1)
    safe = np.where(se > 0, se, 1.0)
    critical = float(
        np.quantile(np.max(np.abs((sampled_means - means) / safe), axis=1), 0.95)
    )
    intervals = np.column_stack((means - critical * se, means + critical * se))
    return means, intervals, critical


def aggregate() -> dict[str, Any]:
    config = load(CONFIG)
    mapping = load(MAPPING)
    hard_integrity = load(HARD_INTEGRITY)
    mass_integrity = load(MASS_INTEGRITY)
    if (
        config.get("status") != "FROZEN_BEFORE_T1_OUTCOME_AGGREGATION"
        or mapping.get("status") != "PASS_T1_MAPPING_DECOMPOSITION_PREFLIGHT_NO_OUTCOME"
        or mapping.get("config_sha256") != sha256(CONFIG)
        or mapping.get("scientific_outcome_read") is not False
        or hard_integrity.get("status") != "PASS_H4_D_FULL_INTEGRITY_NO_AGGREGATE"
        or mass_integrity.get("status")
        != "PASS_H4_GATE_C_D_FULL_INTEGRITY_NO_AGGREGATE"
    ):
        raise RuntimeError("T1 aggregation gate failure")
    margin = float(config["statistics"]["equivalence_margin"])
    draws = int(config["statistics"]["bootstrap_draws"])
    models = {}
    any_pre_difference = False
    any_readout_difference = False
    all_equivalent = True
    for model in MODELS:
        hard = {
            path.stem: load(path)
            for path in sorted(
                (ROOT / f"artifacts/iclr_h4/d_raw/{model}").glob("*.json")
            )
        }
        mass = {
            path.stem: load(path)
            for path in sorted(
                (ROOT / f"artifacts/iclr_h4/d_gate_c_raw_v2/{model}").glob("*.json")
            )
        }
        map_rows = mapping["models"][model]["rows"]
        if len(hard) != 710 or set(hard) != set(mass) or {
            row["image_id"] for row in map_rows
        } != set(hard):
            raise RuntimeError(f"{model} T1 sample closure failure")
        rows = []
        for mapped in map_rows:
            image_id = mapped["image_id"]
            if hard[image_id]["image_sha256"] != mapped["image_sha256"]:
                raise RuntimeError("T1 mapping image hash mismatch")
            hard_effect = effect(hard[image_id], "targeted_reverse")
            dose_name = str(mapped["selected_mass_dose"]).replace(".", "p")
            mass_effect = effect(mass[image_id], f"target_alpha_{dose_name}")
            contrast = {
                field: hard_effect[field] - mass_effect[field] for field in FIELDS
            }
            if (
                abs(contrast["fixed"] - contrast["focal"] - contrast["spillover"])
                > 1e-12
                or abs(
                    contrast["rematched"]
                    - contrast["fixed"]
                    - contrast["matching"]
                )
                > 1e-12
                or abs(
                    contrast["native"]
                    - contrast["rematched"]
                    - contrast["selection"]
                )
                > 1e-12
            ):
                raise RuntimeError("T1 decomposition closure failure")
            rows.append({"stratum": mapped["stratum"], **contrast})
        values = np.asarray(
            [[float(row[field]) for field in FIELDS] for row in rows],
            dtype=np.float64,
        )
        strata = np.asarray([row["stratum"] for row in rows])
        means, intervals, critical = bootstrap(
            values, strata, seed=f"T1|{model}", draws=draws
        )
        summaries = {}
        model_pre = False
        model_readout = False
        model_all_equivalent = True
        for index, field in enumerate(FIELDS):
            low, high = map(float, intervals[index])
            difference = low > margin or high < -margin
            equivalent = low >= -margin and high <= margin
            model_all_equivalent = model_all_equivalent and equivalent
            model_pre = model_pre or (field in PRE and difference)
            model_readout = model_readout or (field in READOUT and difference)
            summaries[field] = {
                "mean": float(means[index]),
                "median": statistics.median(float(row[field]) for row in rows),
                "simultaneous_95": [low, high],
                "difference_outside_margin": difference,
                "equivalent_within_margin": equivalent,
            }
        any_pre_difference = any_pre_difference or model_pre
        any_readout_difference = any_readout_difference or model_readout
        all_equivalent = all_equivalent and model_all_equivalent
        models[model] = {
            "n": len(rows),
            "simultaneous_critical_value": critical,
            "pre_readout_difference": model_pre,
            "readout_specific_difference": model_readout,
            "all_endpoints_equivalent": model_all_equivalent,
            "summaries": summaries,
        }
    if any_pre_difference and any_readout_difference:
        status = "T1_PARTIAL_MIXED"
    elif any_pre_difference:
        status = "T1_PRE_READOUT_DIFFERENCE"
    elif any_readout_difference:
        status = "T1_READOUT_DEPENDENT"
    elif all_equivalent:
        status = "T1_TRANSPORT_EQUIVALENT"
    else:
        status = "T1_INCONCLUSIVE"
    result = {
        "schema_version": "fq.iclr.transport.t1_localization.v1",
        "status": status,
        "config_sha256": sha256(CONFIG),
        "mapping_sha256": sha256(MAPPING),
        "hard_integrity_sha256": sha256(HARD_INTEGRITY),
        "mass_integrity_sha256": sha256(MASS_INTEGRITY),
        "equivalence_margin": margin,
        "bootstrap_draws": draws,
        "models": models,
        "next_route": config["next_route"][status],
        "scientific_effect_readout": True,
        "new_model_inference": False,
        "V_F_read": False,
    }
    OUTPUT.write_text(json.dumps(result, indent=2) + "\n")
    return result


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--execute", action="store_true")
    args = parser.parse_args()
    result = (
        aggregate()
        if args.execute
        else {
            "status": "PASS_T1_AGGREGATOR_PREFLIGHT_NO_OUTCOME_READ",
            "mapping_required": "PASS_T1_MAPPING_DECOMPOSITION_PREFLIGHT_NO_OUTCOME",
            "scientific_effect_readout": False,
            "V_F_read": False,
        }
    )
    print(json.dumps(result, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
