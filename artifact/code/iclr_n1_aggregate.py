#!/usr/bin/env python3
"""Frozen N1 mechanism-routing aggregation, gated by full receipt integrity."""

from __future__ import annotations

import hashlib
import json
import statistics
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Any


ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from future_qc.iclr_n1.statistics import (  # noqa: E402
    receipt_estimands,
    robustness_summary,
    stratified_bootstrap_mean,
)


CONFIG = ROOT / "configs" / "fq_iclr_n1_aggregation_v1.json"
INTEGRITY = ROOT / "artifacts" / "iclr_n1" / "N1_REPLAY_INTEGRITY_AUDIT.json"
RECEIPTS = ROOT / "artifacts" / "iclr_n1" / "zz02_full" / "n1_replay"
OUTPUT = ROOT / "artifacts" / "iclr_n1" / "N1_MECHANISM_AGGREGATE.json"


class AggregateError(RuntimeError):
    pass


def require(condition: bool, message: str) -> None:
    if not condition:
        raise AggregateError(message)


def load(path: Path) -> dict[str, Any]:
    value = json.loads(path.read_text(encoding="utf-8"))
    require(isinstance(value, dict), f"invalid JSON root: {path}")
    return value


def sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def signed_summary(
    values: list[float],
    strata: list[str],
    *,
    direction: int,
    config: dict[str, Any],
    seed: str,
) -> dict[str, Any]:
    robustness = config["robustness"]
    bootstrap = config["bootstrap"]
    return {
        "robustness": robustness_summary(
            values,
            expected_direction=direction,
            trim_fraction=float(robustness["trim_fraction_each_tail"]),
            winsor_limits=tuple(robustness["winsor_limits"]),
            minimum_same_direction_fraction=float(robustness["minimum_same_direction_image_fraction"]),
        ),
        "bootstrap": stratified_bootstrap_mean(
            values,
            strata,
            draws=int(bootstrap["draws"]),
            seed=seed,
            confidence_level=float(bootstrap["confidence_level"]),
            expected_direction=direction,
        ),
    }


def route_pass(summary: dict[str, Any], minimum_probability: float) -> bool:
    return (
        summary["robustness"]["pass"] is True
        and summary["bootstrap"]["direction_probability"] >= minimum_probability
    )


def run_aggregate() -> dict[str, Any]:
    config, integrity = load(CONFIG), load(INTEGRITY)
    for path_value, expected_sha in config["locks"].values():
        path = ROOT / path_value
        require(path.is_file() and sha256(path) == expected_sha, f"aggregate lock mismatch: {path_value}")
    require(
        integrity.get("status") == config["execution_boundary"]["requires_full_integrity_status"],
        "full integrity gate not PASS",
    )
    require(integrity.get("effect_readout_computed") is False, "integrity phase read effects")
    require(integrity.get("d_v_f_read") is False, "integrity phase read D/V/F")
    minimum_probability = float(config["gate"]["minimum_bootstrap_direction_probability"])
    model_results = {}
    for model_id in config["model_ids"]:
        paths = sorted((RECEIPTS / model_id).glob("*.json"))
        require(len(paths) == 256, f"aggregate receipt count mismatch: {model_id}")
        rows = [load(path) for path in paths]
        require(len({row["image_id"] for row in rows}) == 256, f"aggregate image duplication: {model_id}")
        effects = [receipt_estimands(row) for row in rows]
        strata = [row["stratum"] for row in rows]
        keys = sorted(effects[0])
        raw = {key: [float(row[key]) for row in effects] for key in keys}
        summaries = {}
        for key, expected in config["estimand_directions"].items():
            source_key = "reverse_fixed" if key in {"reverse_fixed_h1", "reverse_fixed_h2"} else key
            summaries[key] = signed_summary(
                raw[source_key],
                strata,
                direction=int(expected),
                config=config,
                seed=f"{config['bootstrap']['seed_prefix']}|{model_id}|{key}",
            )
        object_groups = {"n1": [], "n5plus": []}
        for value, stratum in zip(raw["reverse_fixed"], strata):
            prefix = stratum.split("|", 1)[0]
            if prefix in object_groups:
                object_groups[prefix].append(value)
        require(all(object_groups.values()), f"crowding endpoints missing: {model_id}")
        n5plus_gt_n1 = statistics.fmean(object_groups["n5plus"]) > statistics.fmean(object_groups["n1"])
        h1 = all(
            route_pass(summaries[key], minimum_probability)
            for key in config["gate"]["h1_required"]
        )
        h2_core = all(
            route_pass(summaries[key], minimum_probability)
            for key in config["gate"]["h2_required"]
        )
        h2_support = (
            route_pass(summaries["reverse_duplicate_pairs"], minimum_probability)
            or n5plus_gt_n1
        )
        h2 = h2_core and h2_support
        reverse_mean = summaries["reverse_fixed_h2"]["robustness"]["raw_mean"]
        control_means = {
            key: statistics.fmean(raw[key])
            for key in (
                "random_control_fixed_vs_baseline",
                "different_object_control_fixed_vs_baseline",
            )
        }
        negative_control_comparable = max(abs(value) for value in control_means.values()) >= abs(reverse_mean)
        if negative_control_comparable:
            h1 = h2 = False
        h3 = (
            route_pass(summaries["reverse_native_postprocess"], minimum_probability)
            and not route_pass(summaries["reverse_fixed_h2"], minimum_probability)
            and not route_pass(summaries["reverse_set"], minimum_probability)
        )
        model_results[model_id] = {
            "n": len(rows),
            "summaries": summaries,
            "crowding_diagnostic": {
                "reverse_fixed_n1_mean": statistics.fmean(object_groups["n1"]),
                "reverse_fixed_n5plus_mean": statistics.fmean(object_groups["n5plus"]),
                "n5plus_gt_n1": n5plus_gt_n1,
            },
            "negative_control_means": control_means,
            "negative_control_comparable": negative_control_comparable,
            "routes": {"h1": h1, "h2": h2, "h3": h3},
        }
    detr, dino = model_results["detr_r50_500"]["routes"], model_results["dino_r50_4s_12e"]["routes"]
    route = None
    if detr["h1"] and dino["h1"]:
        route = "shared_h1"
    elif detr["h2"] and dino["h2"]:
        route = "shared_h2"
    elif detr["h1"] and dino["h2"]:
        route = "detr_h1_dino_h2"
    elif detr["h2"] and dino["h1"]:
        route = "detr_h2_dino_h1"
    any_h3 = detr["h3"] or dino["h3"]
    return {
        "schema_version": "fq.iclr.n1.mechanism_aggregate.v1",
        "generated_at": datetime.now(timezone.utc).isoformat(),
        "status": (
            "PASS_N1_MECHANISM_ROUTE_TO_N2"
            if route
            else "STOP_N1_H3_ONLY"
            if any_h3
            else "NO_GO_N1_NO_STABLE_MECHANISM_ROUTE"
        ),
        "post_m1_new_hypothesis": True,
        "original_m1_claim_pass": False,
        "m1_remains_no_go": True,
        "original_m2_allowed": False,
        "config_sha256": sha256(CONFIG),
        "integrity_audit_sha256": sha256(INTEGRITY),
        "models": model_results,
        "selected_route": route,
        "n2_new_continuation_allowed": route is not None,
        "f_split_read": False,
    }


def main() -> int:
    if not INTEGRITY.is_file():
        print(
            json.dumps(
                {
                    "status": "WAIT_N1_AGGREGATION_FULL_INTEGRITY_MISSING",
                    "effect_readout_computed": False,
                    "f_split_read": False,
                },
                indent=2,
            )
        )
        return 0
    try:
        result = run_aggregate()
        OUTPUT.write_text(json.dumps(result, indent=2) + "\n", encoding="utf-8")
    except Exception as exc:
        print(json.dumps({"status": "FAIL_N1_AGGREGATION", "reason": f"{type(exc).__name__}: {exc}"}, indent=2))
        return 2
    print(json.dumps({"status": result["status"], "selected_route": result["selected_route"], "n2_new_continuation_allowed": result["n2_new_continuation_allowed"]}, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
