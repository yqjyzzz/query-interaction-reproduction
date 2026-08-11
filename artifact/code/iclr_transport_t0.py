#!/usr/bin/env python3
"""T0 common-estimand reaggregation from existing H4 receipts only."""

from __future__ import annotations

import argparse
import hashlib
import json
import statistics
from pathlib import Path
from typing import Any

import numpy as np


ROOT = Path(__file__).resolve().parents[1]
CONFIG = ROOT / "configs/fq_iclr_transport_t0_v1.json"
HARD_INTEGRITY = ROOT / "artifacts/iclr_h4/H4_D_FULL_INTEGRITY_AUDIT.json"
MASS_INTEGRITY = ROOT / "artifacts/iclr_h4/H4_GATE_C_D_FULL_INTEGRITY_AUDIT_V2.json"
OUTPUT = ROOT / "artifacts/iclr_transport/T0_COMMON_ESTIMAND_AUDIT.json"
MODELS = ("detr_r50_500", "dino_r50_4s_12e")
DOSES = (0.0, 0.25, 0.5, 0.75, 1.0)
ENDPOINTS = ("local", "target", "fixed", "rematched", "native")


class T0Error(RuntimeError):
    pass


def load(path: Path) -> dict[str, Any]:
    value = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise T0Error(f"JSON object required: {path}")
    return value


def sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def arm(kind: str, dose: float) -> str:
    return f"{kind}_alpha_{str(dose).replace('.', 'p')}"


def fixed_utility(measurement: dict[str, Any], assignment: list[int]) -> float:
    matrix = measurement["quality_matrix"]
    if len(assignment) != len(matrix[0]):
        raise T0Error("fixed assignment GT count mismatch")
    total = 0.0
    for gt, prediction in enumerate(assignment):
        if prediction >= 0:
            total += float(matrix[prediction][gt])
    return total / len(assignment)


def effect(
    receipt: dict[str, Any],
    changed_name: str,
) -> dict[str, float]:
    baseline = receipt["arm_measurements"]["baseline"]
    changed = receipt["arm_measurements"][changed_name]
    native_baseline = receipt["native_measurements"]["baseline"]
    native_changed = receipt["native_measurements"][changed_name]
    assignment = [int(value) for value in baseline["optimal_assignment"]]
    return {
        "local": float(changed["fixed_competitor_quality"])
        - float(baseline["fixed_competitor_quality"]),
        "target": float(changed["target_set_quality"])
        - float(baseline["target_set_quality"]),
        "fixed": fixed_utility(changed, assignment)
        - float(baseline["image_set_utility"]),
        "rematched": float(changed["image_set_utility"])
        - float(baseline["image_set_utility"]),
        "native": float(native_changed["image_set_utility"])
        - float(native_baseline["image_set_utility"]),
    }


def expected_hard_mapping(pair: dict[str, int]) -> dict[str, tuple[int, int]]:
    return {
        "identity": (int(pair["leader"]), int(pair["competitor"])),
        "targeted_reverse": (int(pair["competitor"]), int(pair["leader"])),
        "reverse_sham": (int(pair["sham"]), int(pair["leader"])),
        "random_edge": (
            int(pair["random_recipient"]),
            int(pair["random_source"]),
        ),
        "different_object": (
            int(pair["different_recipient"]),
            int(pair["different_source"]),
        ),
    }


def close_mapping(hard: dict[str, Any], mass: dict[str, Any]) -> None:
    if hard["pair"] != mass["pair"]:
        raise T0Error("hard/mass frozen pair mismatch")
    pair = hard["pair"]
    for name, (recipient, source) in expected_hard_mapping(pair).items():
        audit = hard["hook_audits"][name]
        if (
            int(audit["recipient_query_id"]) != recipient
            or int(audit["source_query_id"]) != source
        ):
            raise T0Error(f"hard hook mapping mismatch: {name}")
    expected_mass = {
        "target": (
            int(pair["competitor"]),
            int(pair["leader"]),
            int(pair["sham"]),
        ),
        "control": (
            int(pair["sham"]),
            int(pair["leader"]),
            int(pair["competitor"]),
        ),
    }
    for kind, expected in expected_mass.items():
        for dose in DOSES:
            audit = mass["swap_audits"][arm(kind, dose)]
            observed = (
                int(audit["recipient"]),
                int(audit["target_source"]),
                int(audit["control_source"]),
            )
            if observed != expected:
                raise T0Error(f"mass hook mapping mismatch: {kind} {dose}")


def paired_bootstrap(
    rows: list[dict[str, Any]],
    fields: list[str],
    seed_text: str,
    draws: int,
) -> tuple[dict[str, dict[str, Any]], float]:
    values = np.asarray([[float(row[field]) for field in fields] for row in rows])
    means = values.mean(axis=0)
    strata = np.asarray([row["stratum"] for row in rows])
    groups = [np.flatnonzero(strata == label) for label in sorted(set(strata))]
    seed = int.from_bytes(hashlib.sha256(seed_text.encode()).digest()[:8], "big")
    rng = np.random.default_rng(seed)
    boot = np.empty((draws, len(fields)), dtype=np.float64)
    for start in range(0, draws, 250):
        stop = min(start + 250, draws)
        subtotal = np.zeros((stop - start, len(fields)), dtype=np.float64)
        for indices in groups:
            sampled = rng.choice(indices, size=(stop - start, len(indices)), replace=True)
            subtotal += values[sampled].sum(axis=1)
        boot[start:stop] = subtotal / len(rows)
    standard_error = boot.std(axis=0, ddof=1)
    safe_se = np.where(standard_error > 0, standard_error, 1.0)
    max_stat = np.max(np.abs((boot - means) / safe_se), axis=1)
    critical = float(np.quantile(max_stat, 0.95))
    result = {}
    for index, field in enumerate(fields):
        marginal = np.quantile(boot[:, index], [0.025, 0.975]).tolist()
        simultaneous = [
            float(means[index] - critical * standard_error[index]),
            float(means[index] + critical * standard_error[index]),
        ]
        result[field] = {
            "mean": float(means[index]),
            "median": statistics.median(float(row[field]) for row in rows),
            "marginal_bootstrap_95": [float(value) for value in marginal],
            "simultaneous_95": simultaneous,
        }
    return result, critical


def aggregate() -> dict[str, Any]:
    config = load(CONFIG)
    hard_integrity = load(HARD_INTEGRITY)
    mass_integrity = load(MASS_INTEGRITY)
    if (
        config.get("status") != "FROZEN_BEFORE_T0_REAGGREGATION"
        or hard_integrity.get("status") != "PASS_H4_D_FULL_INTEGRITY_NO_AGGREGATE"
        or mass_integrity.get("status")
        != "PASS_H4_GATE_C_D_FULL_INTEGRITY_NO_AGGREGATE"
        or config.get("V_read") is not False
        or config.get("F_read") is not False
        or config.get("new_model_inference") is not False
    ):
        raise T0Error("T0 input/config gate failure")
    draws = int(config["bootstrap"]["draws"])
    model_results = {}
    global_noise = 0.0
    for model in MODELS:
        hard_paths = sorted((ROOT / f"artifacts/iclr_h4/d_raw/{model}").glob("*.json"))
        mass_paths = sorted(
            (ROOT / f"artifacts/iclr_h4/d_gate_c_raw_v2/{model}").glob("*.json")
        )
        hard_by_id = {load(path)["image_id"]: load(path) for path in hard_paths}
        mass_by_id = {load(path)["image_id"]: load(path) for path in mass_paths}
        if set(hard_by_id) != set(mass_by_id) or len(hard_by_id) != 710:
            raise T0Error(f"{model} sample closure failure")
        rows = []
        for image_id in sorted(hard_by_id):
            hard = hard_by_id[image_id]
            mass = mass_by_id[image_id]
            if (
                hard["image_sha256"] != mass["image_sha256"]
                or hard["checkpoint_sha256"] != mass["checkpoint_sha256"]
            ):
                raise T0Error(f"{model}/{image_id} hash binding mismatch")
            close_mapping(hard, mass)
            row: dict[str, Any] = {
                "image_id": image_id,
                "stratum": hard["stratum"],
            }
            hard_effects = {
                name: effect(hard, name)
                for name in (
                    "baseline_repeat",
                    "identity",
                    "targeted_reverse",
                    "reverse_sham",
                    "random_edge",
                    "different_object",
                )
            }
            for name, endpoints in hard_effects.items():
                for endpoint, value in endpoints.items():
                    row[f"hard.{name}.{endpoint}"] = value
            for endpoint in ENDPOINTS:
                row[f"hard.target_minus_active_control.{endpoint}"] = (
                    hard_effects["targeted_reverse"][endpoint]
                    - hard_effects["reverse_sham"][endpoint]
                )
            for kind in ("target", "control"):
                for dose in DOSES:
                    current = effect(mass, arm(kind, dose))
                    for endpoint, value in current.items():
                        row[f"mass.{kind}.{dose}.{endpoint}"] = value
            for dose in DOSES:
                for endpoint in ENDPOINTS:
                    row[f"operator.hard_minus_mass_target.{dose}.{endpoint}"] = (
                        hard_effects["targeted_reverse"][endpoint]
                        - row[f"mass.target.{dose}.{endpoint}"]
                    )
            noise_fields = [
                key
                for key in row
                if key.startswith("hard.baseline_repeat.")
                or key.startswith("hard.identity.")
                or key.startswith("mass.target.0.0.")
                or key.startswith("mass.control.0.0.")
            ]
            global_noise = max(global_noise, *(abs(float(row[key])) for key in noise_fields))
            rows.append(row)
        fields = sorted(key for key in rows[0] if key not in {"image_id", "stratum"})
        summaries, critical = paired_bootstrap(rows, fields, f"T0|{model}", draws)
        model_results[model] = {
            "n": len(rows),
            "simultaneous_critical_value": critical,
            "summaries": summaries,
        }
    margin_rule = config["equivalence_margin"]
    equivalence_margin = max(
        float(margin_rule["utility_resolution"]),
        float(margin_rule["noise_multiplier"]) * global_noise,
    )
    realized_dose_present = False
    state = (
        "T0_COMPARABLE_FROM_EXISTING_RECEIPTS"
        if realized_dose_present
        else "T0_REPAIRABLE_BUT_NEW_INPUT_AUDIT_REQUIRED"
    )
    result = {
        "schema_version": "fq.iclr.transport.t0_common_estimand_audit.v1",
        "status": state,
        "config_sha256": sha256(CONFIG),
        "hard_integrity_sha256": sha256(HARD_INTEGRITY),
        "mass_integrity_sha256": sha256(MASS_INTEGRITY),
        "sample_mapping_hash_readout_closure": True,
        "actual_mapping": {
            "hard_target": "delete competitor_recipient <- leader_source",
            "hard_active_control": "delete sham_recipient <- leader_source",
            "mass_target": "competitor_recipient swaps leader_source with sham_donor",
            "mass_placebo": "sham_recipient swaps leader_source with competitor_donor",
            "same_recipient_control": False,
        },
        "common_reference": "same-receipt baseline",
        "common_readouts": list(ENDPOINTS),
        "native_same_readout_centered": True,
        "maximum_identity_noise": global_noise,
        "equivalence_margin": equivalence_margin,
        "realized_per_head_mass_or_message_displacement_present": False,
        "models": model_results,
        "T1_allowed": False,
        "T2_allowed": False,
        "next_required_input": (
            "A separately authorized input-only audit/replay that records realized "
            "per-head target/donor weights and immediate message displacement."
        ),
        "scientific_effect_readout": True,
        "new_model_inference": False,
        "V_F_read": False,
    }
    OUTPUT.parent.mkdir(parents=True, exist_ok=True)
    OUTPUT.write_text(json.dumps(result, indent=2) + "\n", encoding="utf-8")
    return result


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--execute", action="store_true")
    args = parser.parse_args()
    if not args.execute:
        result = {
            "status": "PASS_T0_PREFLIGHT_NO_RECEIPT_READ",
            "required_inputs": [str(HARD_INTEGRITY), str(MASS_INTEGRITY)],
            "new_model_inference": False,
            "V_F_read": False,
        }
    else:
        result = aggregate()
    print(json.dumps(result, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
