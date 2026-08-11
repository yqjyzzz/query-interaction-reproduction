#!/usr/bin/env python3
"""Fail-closed T0B R-pilot and D baseline-capture integrity audits."""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
from typing import Any


ROOT = Path(__file__).resolve().parents[1]
CONFIG = ROOT / "configs/fq_iclr_transport_t0b_v1.json"
MODELS = ("detr_r50_500", "dino_r50_4s_12e")


def load(path: Path) -> dict[str, Any]:
    value = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise RuntimeError(f"JSON object required: {path}")
    return value


def finite_tree(value: Any) -> bool:
    import math

    if isinstance(value, dict):
        return all(finite_tree(item) for item in value.values())
    if isinstance(value, list):
        return all(finite_tree(item) for item in value)
    return isinstance(value, (int, float)) and math.isfinite(float(value))


def audit(
    mode: str,
    *,
    input_base: Path | None = None,
    output: Path | None = None,
) -> dict[str, Any]:
    config = load(CONFIG)
    is_pilot = mode == "r-pilot"
    expected = 8 if is_pilot else 710
    base = input_base or (ROOT / "artifacts/iclr_transport")
    root = base / (
        "t0b_r_pilot_raw" if is_pilot else "t0b_d_baseline_raw"
    )
    model_results = {}
    for model in MODELS:
        paths = sorted((root / model).glob("*.json"))
        if len(paths) != expected:
            raise RuntimeError(f"{model} receipt count {len(paths)} != {expected}")
        ids = set()
        max_weight = max_message = max_repeat = 0.0
        for path in paths:
            row = load(path)
            if (
                row.get("status") != "PASS_T0B_INPUT_ONLY_RECEIPT"
                or row.get("model_id") != model
                or row.get("scientific_outcome_captured") is not False
                or row.get("scientific_effect_readout") is not False
                or row.get("V_F_read") is not False
                or row.get("D_intervention_forward") is not False
                or not finite_tree(row["realized_metrics"])
            ):
                raise RuntimeError(f"invalid receipt: {path}")
            image_id = row["image_id"]
            if image_id in ids:
                raise RuntimeError("duplicate image receipt")
            ids.add(image_id)
            if is_pilot:
                validation = row["pilot_validation"]
                max_weight = max(max_weight, float(validation["analytic_swap_weight_max_abs"]))
                max_message = max(max_message, float(validation["analytic_message_max_abs"]))
                max_repeat = max(max_repeat, float(validation["baseline_repeat_max_abs"]))
        if is_pilot and (
            max_repeat != 0.0 or max_weight > 5e-6 or max_message > 1e-5
        ):
            raise RuntimeError(f"{model} R pilot analytic closure failed")
        model_results[model] = {
            "receipt_count": len(paths),
            "unique_images": len(ids),
            "maximum_baseline_repeat_error": max_repeat,
            "maximum_analytic_swap_weight_error": max_weight,
            "maximum_analytic_message_error": max_message,
        }
    status = "PASS_T0B_R_PILOT" if is_pilot else "PASS_T0B_D_BASELINE_CAPTURE"
    result = {
        "schema_version": "fq.iclr.transport.t0b_integrity.v1",
        "status": status,
        "config_sha256": hashlib.sha256(CONFIG.read_bytes()).hexdigest(),
        "models": model_results,
        "aggregation_allowed": not is_pilot,
        "scientific_outcome_captured": False,
        "scientific_effect_readout": False,
        "V_F_read": False,
    }
    output_path = output or (
        ROOT
        / "artifacts/iclr_transport"
        / ("T0B_R_PILOT_INTEGRITY.json" if is_pilot else "T0B_D_FULL_INTEGRITY.json")
    )
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(json.dumps(result, indent=2) + "\n")
    return result


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--mode", choices=("r-pilot", "d"), required=True)
    parser.add_argument("--input-base", type=Path)
    parser.add_argument("--output", type=Path)
    args = parser.parse_args()
    print(
        json.dumps(
            audit(args.mode, input_base=args.input_base, output=args.output),
            indent=2,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
