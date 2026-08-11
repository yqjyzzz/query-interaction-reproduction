#!/usr/bin/env python3
"""Frozen T0B realized-dose overlap gate; input-only receipts."""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
from typing import Any

import numpy as np


ROOT = Path(__file__).resolve().parents[1]
CONFIG = ROOT / "configs/fq_iclr_transport_t0b_v1.json"
MODELS = ("detr_r50_500", "dino_r50_4s_12e")
DOSES = ("0.25", "0.5", "0.75", "1.0")
METRICS = {
    "absolute_target_weight_change": (
        "hard_removed_mass",
        "absolute_target_weight_change",
    ),
    "immediate_message_norm": (
        "hard_immediate_message_norm",
        "immediate_message_norm",
    ),
}


def load(path: Path) -> dict[str, Any]:
    value = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise RuntimeError(f"JSON object required: {path}")
    return value


def nearest_coverage(
    hard: np.ndarray,
    mass: np.ndarray,
    *,
    caliper_multiplier: float = 0.25,
) -> tuple[float, float, float]:
    hard_log = np.log1p(hard)
    mass_log = np.log1p(mass.reshape(-1))
    pooled = np.concatenate([hard_log, mass_log])
    caliper = caliper_multiplier * float(pooled.std(ddof=1))
    hard_distance = np.min(np.abs(hard_log[:, None] - mass_log[None, :]), axis=1)
    mass_distance = np.min(np.abs(mass_log[:, None] - hard_log[None, :]), axis=1)
    return (
        float(np.mean(hard_distance <= caliper)),
        float(np.mean(mass_distance <= caliper)),
        caliper,
    )


def coverage_indicators(
    hard: np.ndarray,
    mass: np.ndarray,
    *,
    caliper_multiplier: float = 0.25,
) -> tuple[np.ndarray, np.ndarray]:
    hard_log = np.log1p(hard)
    mass_log = np.log1p(mass.reshape(-1))
    pooled = np.concatenate([hard_log, mass_log])
    caliper = caliper_multiplier * float(pooled.std(ddof=1))
    hard_indicator = (
        np.min(np.abs(hard_log[:, None] - mass_log[None, :]), axis=1) <= caliper
    ).astype(np.float64)
    mass_indicator = (
        np.min(np.abs(mass_log[:, None] - hard_log[None, :]), axis=1) <= caliper
    ).astype(np.float64)
    return hard_indicator, mass_indicator.reshape(mass.shape)


def bootstrap_lower(
    hard: np.ndarray,
    mass: np.ndarray,
    strata: np.ndarray,
    *,
    seed: str,
    draws: int = 10000,
) -> tuple[float, float]:
    rng = np.random.default_rng(
        int.from_bytes(hashlib.sha256(seed.encode()).digest()[:8], "big")
    )
    groups = [np.flatnonzero(strata == label) for label in sorted(set(strata))]
    estimates = np.empty((draws, 2), dtype=np.float64)
    hard_indicator, mass_indicator = coverage_indicators(hard, mass)
    for draw in range(draws):
        indices = np.concatenate(
            [rng.choice(group, size=len(group), replace=True) for group in groups]
        )
        estimates[draw, 0] = hard_indicator[indices].mean()
        estimates[draw, 1] = mass_indicator[indices].mean()
    return float(np.quantile(estimates[:, 0], 0.025)), float(
        np.quantile(estimates[:, 1], 0.025)
    )


def aggregate(input_base: Path, integrity_path: Path, output: Path) -> dict[str, Any]:
    config = load(CONFIG)
    integrity = load(integrity_path)
    if (
        integrity.get("status") != "PASS_T0B_D_BASELINE_CAPTURE"
        or integrity.get("aggregation_allowed") is not True
        or integrity.get("scientific_outcome_captured") is not False
        or config["execution"]["T1_before_terminal_state"] is not False
        or config["execution"]["T2_before_terminal_state"] is not False
    ):
        raise RuntimeError("T0B overlap integrity/config gate failure")
    model_results = {}
    overall_pass = True
    for model in MODELS:
        paths = sorted((input_base / "t0b_d_baseline_raw" / model).glob("*.json"))
        if len(paths) != 710:
            raise RuntimeError(f"{model} receipt count changed")
        receipts = [load(path) for path in paths]
        strata = np.asarray([row["stratum"] for row in receipts])
        metric_results = {}
        for metric, (hard_key, mass_key) in METRICS.items():
            hard = np.asarray(
                [np.median(row["realized_metrics"][hard_key]) for row in receipts],
                dtype=np.float64,
            )
            mass = np.asarray(
                [
                    [
                        np.median(
                            row["realized_metrics"]["swap"][dose][mass_key]
                        )
                        for dose in DOSES
                    ]
                    for row in receipts
                ],
                dtype=np.float64,
            )
            hard_to_mass, mass_to_hard, caliper = nearest_coverage(hard, mass)
            hard_lower, mass_lower = bootstrap_lower(
                hard, mass, strata, seed=f"T0B|{model}|{metric}"
            )
            passed = (
                hard_to_mass >= 0.80
                and mass_to_hard >= 0.80
                and hard_lower >= 0.75
                and mass_lower >= 0.75
            )
            overall_pass = overall_pass and passed
            metric_results[metric] = {
                "hard_to_mass_coverage": hard_to_mass,
                "mass_to_hard_coverage": mass_to_hard,
                "hard_to_mass_bootstrap_95_lower": hard_lower,
                "mass_to_hard_bootstrap_95_lower": mass_lower,
                "log1p_caliper": caliper,
                "pass": passed,
            }
        model_results[model] = {"n": 710, "metrics": metric_results}
    status = (
        "T0B_OVERLAP_SUFFICIENT_ENTER_T1"
        if overall_pass
        else "T0B_OVERLAP_INSUFFICIENT_DIRECT_T2"
    )
    result = {
        "schema_version": "fq.iclr.transport.t0b_overlap.v1",
        "status": status,
        "config_sha256": hashlib.sha256(CONFIG.read_bytes()).hexdigest(),
        "integrity_sha256": hashlib.sha256(integrity_path.read_bytes()).hexdigest(),
        "models": model_results,
        "T1_allowed": overall_pass,
        "T2_preparation_allowed": not overall_pass,
        "scientific_outcome_captured": False,
        "scientific_effect_readout": False,
        "V_F_read": False,
    }
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(result, indent=2) + "\n")
    return result


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--input-base",
        type=Path,
        default=ROOT / "artifacts/iclr_transport",
    )
    parser.add_argument(
        "--integrity",
        type=Path,
        default=ROOT / "artifacts/iclr_transport/T0B_D_FULL_INTEGRITY.json",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=ROOT / "artifacts/iclr_transport/T0B_REALIZED_DOSE_OVERLAP.json",
    )
    args = parser.parse_args()
    print(json.dumps(aggregate(args.input_base, args.integrity, args.output), indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
