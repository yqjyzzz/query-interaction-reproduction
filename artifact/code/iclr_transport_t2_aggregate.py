#!/usr/bin/env python3
"""Frozen T2 initial-128 aggregation after full-integrity PASS."""

from __future__ import annotations

import argparse
import hashlib
import json
import math
from pathlib import Path
from typing import Any

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
CONFIG = ROOT / "configs/fq_iclr_transport_t2_confirmation_v1.json"
EXECUTION = ROOT / "configs/fq_iclr_transport_t2_confirmation_execution_v2.json"
INTEGRITY = ROOT / "artifacts/iclr_transport/T2_CONFIRMATION_V2_FULL_INTEGRITY.json"
MAPPING = ROOT / "artifacts/iclr_transport/T2_OUTCOME_BLIND_MAPPING.json"
INPUT = ROOT / "artifacts/iclr_transport/t2_input_capture_raw"
RAW = ROOT / "artifacts/iclr_transport/t2_confirmation_raw_initial_v2"
OUTPUT = ROOT / "artifacts/iclr_transport/T2_CONFIRMATION_INITIAL_128_AGGREGATE.json"
MODELS = ("detr_r50_500", "dino_r50_4s_12e")
ENDPOINTS = ("local", "spillover", "fixed")
DOSES = ("0.25", "0.5", "0.75")
PAIRWISE = (("M_MINUS_P", "M", "P"), ("M_MINUS_D", "M", "D"), ("P_MINUS_D", "P", "D"))
SUPPORT = ("M", "P", "D", "H")


def load(path: Path) -> dict[str, Any]:
    value = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise RuntimeError(f"JSON object required: {path}")
    return value


def sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def fixed_utility(measurement: dict[str, Any], assignment: list[int]) -> float:
    if not assignment:
        return 0.0
    matrix = measurement["quality_matrix"]
    return sum(
        float(matrix[prediction][gt])
        for gt, prediction in enumerate(assignment)
        if prediction >= 0
    ) / len(assignment)


def primary_effect(receipt: dict[str, Any], arm: str) -> dict[str, float]:
    baseline = receipt["arm_measurements"]["baseline"]
    changed = receipt["arm_measurements"][arm]
    assignment = [int(item) for item in baseline["optimal_assignment"]]
    competitor = int(receipt["pair"]["competitor_query_id"])
    fixed = fixed_utility(changed, assignment) - float(baseline["image_set_utility"])
    focal = (
        sum(
            float(changed["quality_matrix"][competitor][gt])
            - float(baseline["quality_matrix"][competitor][gt])
            for gt, prediction in enumerate(assignment)
            if prediction == competitor
        )
        / len(assignment)
        if assignment
        else 0.0
    )
    result = {
        "local": float(changed["fixed_competitor_quality"])
        - float(baseline["fixed_competitor_quality"]),
        "spillover": fixed - focal,
        "fixed": fixed,
    }
    if not all(math.isfinite(value) for value in result.values()):
        raise RuntimeError("non-finite primary effect")
    return result


def classify_interval(low: float, high: float, band: float) -> str:
    if low >= -band and high <= band:
        return "EQUIVALENT"
    if low > band or high < -band:
        return "DIFFERENT"
    return "INCONCLUSIVE"


def _seed(text: str) -> int:
    return int.from_bytes(hashlib.sha256(text.encode()).digest()[:8], "big")


def bootstrap_family(
    linear: np.ndarray,
    p_effects: np.ndarray,
    p_norms: np.ndarray,
    h_effects: np.ndarray,
    h_norms: np.ndarray,
    strata: np.ndarray,
    *,
    model: str,
    draws: int,
) -> tuple[np.ndarray, np.ndarray, float]:
    """Return estimates, simultaneous intervals and max-t critical value.

    linear is [N, 21]: 9 pairwise then 12 support contrasts. The final three
    family members are H minus the P-continuity prediction.
    """
    groups = [np.flatnonzero(strata == label) for label in sorted(set(strata))]
    rng = np.random.default_rng(_seed(f"T2|{model}|initial128"))
    n = len(linear)

    def discontinuity(indices: np.ndarray) -> np.ndarray:
        # One pooled OLS per endpoint with image as the resampling unit.
        xs = np.concatenate(
            [np.zeros(len(indices)), *(p_norms[indices, dose] for dose in range(3))]
        )
        result = np.empty(3)
        for endpoint in range(3):
            ys = np.concatenate(
                [
                    np.zeros(len(indices)),
                    *(p_effects[indices, dose, endpoint] for dose in range(3)),
                ]
            )
            design = np.column_stack((np.ones(len(xs)), xs))
            coefficient = np.linalg.lstsq(design, ys, rcond=None)[0]
            predicted = coefficient[0] + coefficient[1] * h_norms[indices]
            result[endpoint] = np.mean(h_effects[indices, endpoint] - predicted)
        return result

    observed = np.concatenate((linear.mean(axis=0), discontinuity(np.arange(n))))
    samples = np.empty((draws, len(observed)), dtype=np.float64)
    for draw in range(draws):
        indices = np.concatenate(
            [rng.choice(group, size=len(group), replace=True) for group in groups]
        )
        samples[draw] = np.concatenate(
            (linear[indices].mean(axis=0), discontinuity(indices))
        )
    se = samples.std(axis=0, ddof=1)
    safe = np.where(se > 0, se, 1.0)
    max_t = np.max(np.abs((samples - observed) / safe), axis=1)
    critical = float(np.quantile(max_t, 0.95))
    intervals = np.column_stack((observed - critical * se, observed + critical * se))
    return observed, intervals, critical


def aggregate() -> dict[str, Any]:
    config, execution, integrity, mapping = map(load, (CONFIG, EXECUTION, INTEGRITY, MAPPING))
    if (
        integrity.get("status") != "PASS_T2_CONFIRMATION_V2_FULL_INTEGRITY_NO_AGGREGATE"
        or integrity.get("execution_sha256") != sha256(EXECUTION)
        or integrity.get("scientific_effect_readout") is not False
        or integrity.get("aggregation_started") is not False
        or execution.get("precision_extension_authorized") is not False
        or execution.get("V_read") is not False
        or execution.get("F_read") is not False
        or mapping.get("status") != "PASS_T2_OUTCOME_BLIND_MAPPING_FROZEN"
        or execution.get("outcome_blind_mapping_sha256") != sha256(MAPPING)
    ):
        raise RuntimeError("T2 aggregation gate failure")
    draws = int(config["inference"]["bootstrap_draws"])
    bands = config["inference"]["equivalence_bands"]
    model_results: dict[str, Any] = {}
    extension_reasons: list[str] = []
    assay_failure = False
    global_classes: set[str] = set()
    all_pairwise_equivalent = True
    for model in MODELS:
        mapped = {row["image_id"]: row for row in mapping["models"][model]["rows"]}
        paths = sorted((RAW / model).glob("*.json"))
        if len(paths) != 128 or {path.stem for path in paths} != set(mapped):
            raise RuntimeError(f"{model} raw/mapping closure failure")
        rows, strata = [], []
        p_curves, p_norm_rows, h_rows, h_norm_rows = [], [], [], []
        for path in paths:
            receipt = load(path)
            input_receipt = load(INPUT / model / path.name)
            map_row = mapped[path.stem]
            if (
                receipt["image_sha256"] != map_row["image_sha256"]
                or input_receipt["image_sha256"] != map_row["image_sha256"]
            ):
                raise RuntimeError("T2 aggregate image binding mismatch")
            effects: dict[str, list[dict[str, float]]] = {"M": [], "P": [], "D": []}
            for dose in DOSES:
                token = dose.replace(".", "p")
                effects["M"].append(primary_effect(receipt, f"C_v2_{token}"))
                a_dose = str(map_row["selected_doses"][f"A_for_M_{dose}"])
                b_dose = str(map_row["selected_doses"][f"B_for_M_{dose}"])
                effects["P"].append(primary_effect(receipt, f"A_{a_dose.replace('.', 'p')}"))
                effects["D"].append(primary_effect(receipt, f"B_{b_dose.replace('.', 'p')}"))
            averaged = {
                operator: {
                    endpoint: float(np.mean([item[endpoint] for item in values]))
                    for endpoint in ENDPOINTS
                }
                for operator, values in effects.items()
            }
            hard = primary_effect(receipt, "hard")
            vector = []
            for _, left, right in PAIRWISE:
                vector.extend(averaged[left][endpoint] - averaged[right][endpoint] for endpoint in ENDPOINTS)
            for operator in SUPPORT:
                source = hard if operator == "H" else averaged[operator]
                vector.extend(source[endpoint] for endpoint in ENDPOINTS)
            rows.append(vector)
            strata.append(map_row["stratum"])
            p_curves.append(
                [[primary_effect(receipt, f"A_{dose.replace('.', 'p')}")[endpoint] for endpoint in ENDPOINTS] for dose in DOSES]
            )
            metrics = input_receipt["realized_metrics"]
            p_norm_rows.append(
                [float(np.mean(metrics["A"][dose]["immediate_message_norm"])) for dose in DOSES]
            )
            h_rows.append([hard[endpoint] for endpoint in ENDPOINTS])
            h_norm_rows.append(
                float(np.mean(metrics["C_v2"]["0.75"]["immediate_message_norm"])) / 0.75
            )
        estimates, intervals, critical = bootstrap_family(
            np.asarray(rows), np.asarray(p_curves), np.asarray(p_norm_rows),
            np.asarray(h_rows), np.asarray(h_norm_rows), np.asarray(strata),
            model=model, draws=draws,
        )
        names = (
            [f"{contrast}_{endpoint}" for contrast, _, _ in PAIRWISE for endpoint in ENDPOINTS]
            + [f"{operator}_MINUS_BASELINE_{endpoint}" for operator in SUPPORT for endpoint in ENDPOINTS]
            + [f"H_MINUS_P_CONTINUITY_{endpoint}" for endpoint in ENDPOINTS]
        )
        summaries: dict[str, Any] = {}
        endpoint_classes = {endpoint: set() for endpoint in ENDPOINTS}
        model_assay = False
        for index, name in enumerate(names):
            endpoint = next(item for item in ENDPOINTS if name.endswith(f"_{item}"))
            low, high = map(float, intervals[index])
            band = float(bands[endpoint])
            state = classify_interval(low, high, band)
            half_width = (high - low) / 2
            if state == "INCONCLUSIVE" and half_width > band:
                extension_reasons.append(f"{model}:{name}")
            summaries[name] = {
                "estimate": float(estimates[index]),
                "simultaneous_95": [low, high],
                "band": band,
                "state": state,
                "half_width": half_width,
            }
            if name.startswith("H_MINUS_BASELINE_") and state == "DIFFERENT":
                model_assay = True
        assay_failure = assay_failure or not model_assay
        for endpoint in ENDPOINTS:
            mp = summaries[f"M_MINUS_P_{endpoint}"]["state"]
            md = summaries[f"M_MINUS_D_{endpoint}"]["state"]
            pd = summaries[f"P_MINUS_D_{endpoint}"]["state"]
            disc = summaries[f"H_MINUS_P_CONTINUITY_{endpoint}"]["state"]
            ms = summaries[f"M_MINUS_BASELINE_{endpoint}"]
            ps = summaries[f"P_MINUS_BASELINE_{endpoint}"]
            ds = summaries[f"D_MINUS_BASELINE_{endpoint}"]
            all_pairwise_equivalent &= mp == md == pd == "EQUIVALENT"
            if pd == "DIFFERENT":
                endpoint_classes[endpoint].add("DONOR")
            if mp == "DIFFERENT" and md == "DIFFERENT" and pd == "EQUIVALENT":
                endpoint_classes[endpoint].add("RENORMALIZATION")
            opposite_p = ps["state"] == "DIFFERENT" and ps["estimate"] * ms["estimate"] < 0
            opposite_d = ds["state"] == "DIFFERENT" and ds["estimate"] * ms["estimate"] < 0
            if ms["state"] == "DIFFERENT" and (ps["state"] == "EQUIVALENT" or opposite_p) and (ds["state"] == "EQUIVALENT" or opposite_d):
                endpoint_classes[endpoint].add("MESSAGE")
            if disc == "DIFFERENT":
                endpoint_classes[endpoint].add("DISCONTINUITY")
            global_classes.update(endpoint_classes[endpoint])
        model_results[model] = {
            "n": 128,
            "simultaneous_critical_value": critical,
            "positive_control_pass": model_assay,
            "endpoint_classes": {key: sorted(value) for key, value in endpoint_classes.items()},
            "contrasts": summaries,
        }
    precision_extension_required = bool(extension_reasons)
    if assay_failure:
        terminal = "T2_INCONCLUSIVE_PRECISION"
    elif precision_extension_required:
        terminal = "T2_INITIAL_128_PRECISION_EXTENSION_REQUIRED"
    elif all_pairwise_equivalent and not global_classes:
        terminal = "T2_TRANSPORT_EQUIVALENT_AFTER_CALIBRATION"
    elif len(global_classes) > 1:
        terminal = "T2_MIXED_OPERATOR_DEPENDENCE"
    elif global_classes == {"DONOR"}:
        terminal = "T2_DONOR_DEPENDENT_REDISTRIBUTION"
    elif global_classes == {"RENORMALIZATION"}:
        terminal = "T2_RENORMALIZATION_DEPENDENT"
    elif global_classes == {"MESSAGE"}:
        terminal = "T2_MESSAGE_SPECIFIC"
    elif global_classes == {"DISCONTINUITY"}:
        terminal = "T2_ENDPOINT_DISCONTINUITY"
    else:
        terminal = "T2_INCONCLUSIVE_PRECISION"
    return {
        "schema_version": "fq.iclr.transport.t2_confirmation_initial_aggregate.v1",
        "status": terminal,
        "config_sha256": sha256(CONFIG),
        "execution_sha256": sha256(EXECUTION),
        "integrity_sha256": sha256(INTEGRITY),
        "mapping_sha256": sha256(MAPPING),
        "bootstrap_draws": draws,
        "models": model_results,
        "positive_control_pass": not assay_failure,
        "supported_mechanism_classes": sorted(global_classes),
        "all_pairwise_equivalent": all_pairwise_equivalent,
        "precision_extension_required": precision_extension_required,
        "precision_extension_authorized": False,
        "precision_extension_reasons": extension_reasons,
        "scientific_effect_readout": True,
        "V_F_read": False,
    }


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--execute", action="store_true")
    args = parser.parse_args()
    if not args.execute:
        print(json.dumps({"status": "PASS_T2_AGGREGATOR_PREFLIGHT_NO_OUTCOME", "integrity_required": True, "V_F_read": False}, indent=2))
        return 0
    result = aggregate()
    OUTPUT.write_text(json.dumps(result, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(result, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
