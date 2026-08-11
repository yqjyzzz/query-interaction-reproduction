#!/usr/bin/env python3
"""H4 scientific aggregator; blocked until the full D integrity audit passes."""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import random
import statistics
from pathlib import Path
from typing import Any

from future_qc.iclr_h4.estimands import image_set_decomposition


ROOT = Path(__file__).resolve().parents[1]
MODELS = ("detr_r50_500", "dino_r50_4s_12e")
INTEGRITY = ROOT / "artifacts/iclr_h4/H4_D_FULL_INTEGRITY_AUDIT.json"
OUTPUT = ROOT / "artifacts/iclr_h4/H4_D_SCIENTIFIC_AGGREGATE.json"
DEADZONE = 1e-4
BOOTSTRAP_DRAWS = 10000
BOOTSTRAP_SEED = "FQ-ICLR-H4-D-BOOTSTRAP-V1"


class H4AggregateError(RuntimeError):
    pass


def load(path: Path) -> dict[str, Any]:
    value = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise H4AggregateError(f"JSON object required: {path}")
    return value


def percentile(values: list[float], probability: float) -> float:
    ordered = sorted(values)
    position = (len(ordered) - 1) * probability
    lower = math.floor(position)
    upper = math.ceil(position)
    if lower == upper:
        return ordered[lower]
    return ordered[lower] * (upper - position) + ordered[upper] * (position - lower)


def bootstrap_ci(rows: list[dict[str, Any]], field: str, model: str) -> list[float]:
    rng = random.Random(f"{BOOTSTRAP_SEED}|{model}|{field}")
    by_stratum: dict[str, list[float]] = {}
    for row in rows:
        by_stratum.setdefault(row["stratum"], []).append(float(row[field]))
    draws = []
    for _ in range(BOOTSTRAP_DRAWS):
        sampled = []
        for values in by_stratum.values():
            sampled.extend(rng.choice(values) for _ in range(len(values)))
        draws.append(statistics.fmean(sampled))
    return [percentile(draws, 0.025), percentile(draws, 0.975)]


def wilson(successes: int, total: int) -> list[float]:
    z = 1.959963984540054
    p = successes / total
    denominator = 1 + z * z / total
    center = (p + z * z / (2 * total)) / denominator
    radius = z * math.sqrt(p * (1 - p) / total + z * z / (4 * total * total)) / denominator
    return [center - radius, center + radius]


def receipt_effect(row: dict[str, Any]) -> dict[str, float]:
    sham = row["arm_measurements"]["reverse_sham"]
    intervention = row["arm_measurements"]["targeted_reverse"]
    decomposition = image_set_decomposition(
        sham_quality=sham["quality_matrix"],
        intervention_quality=intervention["quality_matrix"],
        focal_prediction=int(row["pair"]["competitor"]),
        native_intervention_quality=row["native_measurements"]["targeted_reverse"][
            "quality_matrix"
        ],
    )
    baseline = row["arm_measurements"]["baseline"]
    random_edge = row["arm_measurements"]["random_edge"]
    different = row["arm_measurements"]["different_object"]
    return {
        "delta_local": float(intervention["fixed_competitor_quality"])
        - float(sham["fixed_competitor_quality"]),
        "delta_target": float(intervention["target_set_quality"])
        - float(sham["target_set_quality"]),
        "delta_image_fixed": float(decomposition["utilities"]["u2"])
        - float(decomposition["utilities"]["u0"]),
        "delta_image_rematched": float(decomposition["utilities"]["u3"])
        - float(decomposition["utilities"]["u0"]),
        "delta_native": float(decomposition["total"]),
        "delta_focal": float(decomposition["components"]["focal"]),
        "delta_spillover": float(decomposition["components"]["spillover"]),
        "delta_matching": float(decomposition["components"]["matching"]),
        "delta_selection": float(decomposition["components"]["selection"]),
        "closure_error": float(decomposition["closure_error"]),
        "random_control_local": float(random_edge["fixed_competitor_quality"])
        - float(baseline["fixed_competitor_quality"]),
        "different_control_local": float(different["fixed_competitor_quality"])
        - float(baseline["fixed_competitor_quality"]),
    }


def aggregate() -> dict[str, Any]:
    integrity = load(INTEGRITY)
    if (
        integrity.get("status") != "PASS_H4_D_FULL_INTEGRITY_NO_AGGREGATE"
        or integrity.get("aggregation_allowed") is not True
        or integrity.get("scientific_effect_readout") is not False
    ):
        raise H4AggregateError("full D integrity gate is not PASS")
    model_results = {}
    for model in MODELS:
        rows = []
        paths = sorted((ROOT / f"artifacts/iclr_h4/d_raw/{model}").glob("*.json"))
        if len(paths) != integrity["models"][model]["receipt_count"]:
            raise H4AggregateError(f"{model} receipt count changed after integrity")
        for path in paths:
            receipt = load(path)
            effect = receipt_effect(receipt)
            rows.append(
                {
                    "image_id": receipt["image_id"],
                    "stratum": receipt["stratum"],
                    **effect,
                }
            )
        summaries = {}
        for field in (
            "delta_local",
            "delta_target",
            "delta_image_fixed",
            "delta_image_rematched",
            "delta_native",
            "delta_focal",
            "delta_spillover",
            "delta_matching",
            "delta_selection",
            "random_control_local",
            "different_control_local",
        ):
            values = [float(row[field]) for row in rows]
            summaries[field] = {
                "mean": statistics.fmean(values),
                "median": statistics.median(values),
                "bootstrap_95": bootstrap_ci(rows, field, model),
            }
        target_joint = sum(
            row["delta_local"] > DEADZONE and row["delta_target"] < -DEADZONE
            for row in rows
        )
        image_joint = sum(
            row["delta_local"] > DEADZONE
            and row["delta_image_rematched"] < -DEADZONE
            for row in rows
        )
        model_results[model] = {
            "n": len(rows),
            "summaries": summaries,
            "target_joint_reversal": {
                "count": target_joint,
                "rate": target_joint / len(rows),
                "wilson_95": wilson(target_joint, len(rows)),
            },
            "image_joint_reversal": {
                "count": image_joint,
                "rate": image_joint / len(rows),
                "wilson_95": wilson(image_joint, len(rows)),
            },
            "maximum_absolute_closure_error": max(
                abs(row["closure_error"]) for row in rows
            ),
        }
    result = {
        "schema_version": "fq.iclr.h4.d_scientific_aggregate.v1",
        "status": "PASS_H4_D_SCIENTIFIC_AGGREGATE_COMPUTED",
        "integrity_audit_sha256": hashlib.sha256(INTEGRITY.read_bytes()).hexdigest(),
        "deadzone": DEADZONE,
        "bootstrap_draws": BOOTSTRAP_DRAWS,
        "models": model_results,
        "scientific_effect_readout": True,
        "V_F_read": False,
        "next_gate": "H4_GATE_A_B_EVALUATION",
    }
    OUTPUT.write_text(json.dumps(result, indent=2) + "\n", encoding="utf-8")
    return result


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--execute", action="store_true")
    args = parser.parse_args()
    if not args.execute:
        result = {
            "status": "PASS_H4_AGGREGATOR_PREFLIGHT_NO_RECEIPT_READ",
            "integrity_gate_required": "PASS_H4_D_FULL_INTEGRITY_NO_AGGREGATE",
            "scientific_effect_readout": False,
        }
    else:
        result = aggregate()
    print(json.dumps(result, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
