#!/usr/bin/env python3
"""Reproduce frozen H4-D, Gate C, T1, and T2 results from compact row tables."""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import random
import statistics
from pathlib import Path
from typing import Any

import numpy as np


MODELS = ("detr_r50_500", "dino_r50_4s_12e")
H4_FIELDS = (
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
)
T1_FIELDS = (
    "local",
    "focal",
    "spillover",
    "fixed",
    "matching",
    "rematched",
    "selection",
    "native",
)
T1_PRE = ("local", "focal", "spillover", "fixed")
T1_READOUT = ("matching", "selection")
T2_ENDPOINTS = ("local", "spillover", "fixed")
T2_PAIRWISE = (
    ("M_MINUS_P", "M", "P"),
    ("M_MINUS_D", "M", "D"),
    ("P_MINUS_D", "P", "D"),
)
T2_SUPPORT = ("M", "P", "D", "H")
GATE_C_DOSES = (0.0, 0.25, 0.5, 0.75, 1.0)
GATE_C_INTERMEDIATE = (0.25, 0.5, 0.75)
DEADZONE = 1e-4
BOOTSTRAP_DRAWS = 10000


def read_json(path: Path) -> dict[str, Any]:
    value = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise TypeError(f"JSON object required: {path}")
    return value


def read_jsonl(path: Path) -> list[dict[str, Any]]:
    rows = []
    with path.open("r", encoding="utf-8") as handle:
        for line in handle:
            value = json.loads(line)
            if not isinstance(value, dict):
                raise TypeError(f"JSON object required in {path}")
            rows.append(value)
    return rows


def write_json(path: Path, value: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, indent=2) + "\n", encoding="utf-8")


def percentile(values: list[float], probability: float) -> float:
    ordered = sorted(values)
    position = (len(ordered) - 1) * probability
    lower = math.floor(position)
    upper = math.ceil(position)
    if lower == upper:
        return ordered[lower]
    return ordered[lower] * (upper - position) + ordered[upper] * (position - lower)


def h4_bootstrap(rows: list[dict[str, Any]], field: str, model: str) -> list[float]:
    rng = random.Random(f"FQ-ICLR-H4-D-BOOTSTRAP-V1|{model}|{field}")
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
    radius = (
        z
        * math.sqrt(p * (1 - p) / total + z * z / (4 * total * total))
        / denominator
    )
    return [center - radius, center + radius]


def reproduce_h4(data: Path) -> dict[str, Any]:
    models = {}
    for model in MODELS:
        rows = read_jsonl(data / f"h4_d_{model}.jsonl")
        summaries = {}
        for field in H4_FIELDS:
            values = [float(row[field]) for row in rows]
            summaries[field] = {
                "mean": statistics.fmean(values),
                "median": statistics.median(values),
                "bootstrap_95": h4_bootstrap(rows, field, model),
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
        models[model] = {
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
                abs(float(row["closure_error"])) for row in rows
            ),
        }
    return {
        "status": "PASS_H4_D_SCIENTIFIC_AGGREGATE_COMPUTED",
        "deadzone": DEADZONE,
        "bootstrap_draws": BOOTSTRAP_DRAWS,
        "models": models,
    }


def gate_c_bootstrap(
    values: list[float], strata: list[str], seed_text: str
) -> list[float]:
    digest = hashlib.sha256(seed_text.encode()).digest()
    rng = np.random.default_rng(int.from_bytes(digest[:8], "big"))
    array = np.asarray(values, dtype=np.float64)
    strata_array = np.asarray(strata)
    draws = np.zeros(BOOTSTRAP_DRAWS, dtype=np.float64)
    unique = sorted(set(strata))
    for start in range(0, BOOTSTRAP_DRAWS, 500):
        stop = min(start + 500, BOOTSTRAP_DRAWS)
        subtotal = np.zeros(stop - start, dtype=np.float64)
        count = 0
        for label in unique:
            indices = np.flatnonzero(strata_array == label)
            sampled = rng.choice(
                indices, size=(stop - start, len(indices)), replace=True
            )
            subtotal += array[sampled].sum(axis=1)
            count += len(indices)
        draws[start:stop] = subtotal / count
    return [float(np.quantile(draws, 0.025)), float(np.quantile(draws, 0.975))]


def gate_c_summary(
    values: list[float], strata: list[str], seed: str
) -> dict[str, Any]:
    return {
        "mean": statistics.fmean(values),
        "median": statistics.median(values),
        "bootstrap_95": gate_c_bootstrap(values, strata, seed),
    }


def reproduce_gate_c(data: Path) -> dict[str, Any]:
    models = {}
    passing_candidates = []
    for model in MODELS:
        rows = read_jsonl(data / f"gate_c_{model}.jsonl")
        strata = [row["stratum"] for row in rows]
        dose_results = {}
        for dose in GATE_C_DOSES:
            target = [row["effects"][str(dose)]["target"] for row in rows]
            control = [row["effects"][str(dose)]["control"] for row in rows]
            fields = {}
            for name in ("local", "target", "image", "native"):
                fields[f"target_{name}"] = gate_c_summary(
                    [float(row[name]) for row in target],
                    strata,
                    f"{model}|{dose}|target|{name}",
                )
                fields[f"control_{name}"] = gate_c_summary(
                    [float(row[name]) for row in control],
                    strata,
                    f"{model}|{dose}|control|{name}",
                )
            target_joint = [
                float(row["local"] > DEADZONE and row["target"] < -DEADZONE)
                for row in target
            ]
            control_joint = [
                float(row["local"] > DEADZONE and row["target"] < -DEADZONE)
                for row in control
            ]
            difference = [
                left - right
                for left, right in zip(target_joint, control_joint, strict=True)
            ]
            joint = {
                "target_rate": statistics.fmean(target_joint),
                "control_rate": statistics.fmean(control_joint),
                "paired_rate_difference": gate_c_summary(
                    difference, strata, f"{model}|{dose}|joint_difference"
                ),
            }
            local_pass = fields["target_local"]["bootstrap_95"][0] > 0
            target_harm = fields["target_target"]["bootstrap_95"][1] < 0
            image_harm = fields["target_image"]["bootstrap_95"][1] < 0
            control_pass = joint["paired_rate_difference"]["bootstrap_95"][0] > 0
            candidate_pass = (
                dose in GATE_C_INTERMEDIATE
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
                    abs(float(row["closure"])) for row in target + control
                ),
                "gate_checks": {
                    "intermediate_dose": dose in GATE_C_INTERMEDIATE,
                    "local_mean_ci_positive": local_pass,
                    "target_or_image_harm_ci_negative": target_harm or image_harm,
                    "paired_joint_difference_ci_positive": control_pass,
                },
                "candidate_pass": candidate_pass,
            }
        models[model] = {"n": len(rows), "doses": dose_results}
    gate_pass = bool(passing_candidates)
    return {
        "status": (
            "GO_H4D_NOT_MASK_ARTIFACT"
            if gate_pass
            else "STOP_GENERALIZATION_HARD_MASK_ONLY"
        ),
        "bootstrap_draws": BOOTSTRAP_DRAWS,
        "deadzone": DEADZONE,
        "models": models,
        "passing_candidates": passing_candidates,
        "gate_c_pass": gate_pass,
        "next_gate": "H4_GATE_D_ARCHITECTURE_BOUNDARY" if gate_pass else "STOP",
    }


def t1_bootstrap(
    values: np.ndarray, strata: np.ndarray, *, seed: str
) -> tuple[np.ndarray, np.ndarray, float]:
    groups = [np.flatnonzero(strata == label) for label in sorted(set(strata))]
    rng = np.random.default_rng(
        int.from_bytes(hashlib.sha256(seed.encode()).digest()[:8], "big")
    )
    sampled_means = np.empty((BOOTSTRAP_DRAWS, values.shape[1]), dtype=np.float64)
    for start in range(0, BOOTSTRAP_DRAWS, 250):
        stop = min(BOOTSTRAP_DRAWS, start + 250)
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


def reproduce_t1(data: Path) -> dict[str, Any]:
    margin = 0.0001
    models = {}
    any_pre_difference = False
    any_readout_difference = False
    all_equivalent = True
    for model in MODELS:
        rows = read_jsonl(data / f"t1_{model}.jsonl")
        values = np.asarray(
            [[float(row[field]) for field in T1_FIELDS] for row in rows],
            dtype=np.float64,
        )
        strata = np.asarray([row["stratum"] for row in rows])
        means, intervals, critical = t1_bootstrap(values, strata, seed=f"T1|{model}")
        summaries = {}
        model_pre = False
        model_readout = False
        model_all_equivalent = True
        for index, field in enumerate(T1_FIELDS):
            low, high = map(float, intervals[index])
            difference = low > margin or high < -margin
            equivalent = low >= -margin and high <= margin
            model_all_equivalent = model_all_equivalent and equivalent
            model_pre = model_pre or (field in T1_PRE and difference)
            model_readout = model_readout or (field in T1_READOUT and difference)
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
    return {
        "status": status,
        "equivalence_margin": margin,
        "bootstrap_draws": BOOTSTRAP_DRAWS,
        "models": models,
    }


def t2_classify(low: float, high: float, band: float) -> str:
    if low >= -band and high <= band:
        return "EQUIVALENT"
    if low > band or high < -band:
        return "DIFFERENT"
    return "INCONCLUSIVE"


def t2_seed(text: str) -> int:
    return int.from_bytes(hashlib.sha256(text.encode()).digest()[:8], "big")


def t2_bootstrap_family(
    linear: np.ndarray,
    p_effects: np.ndarray,
    p_norms: np.ndarray,
    h_effects: np.ndarray,
    h_norms: np.ndarray,
    strata: np.ndarray,
    *,
    model: str,
) -> tuple[np.ndarray, np.ndarray, float]:
    groups = [np.flatnonzero(strata == label) for label in sorted(set(strata))]
    rng = np.random.default_rng(t2_seed(f"T2|{model}|initial128"))
    n = len(linear)

    def discontinuity(indices: np.ndarray) -> np.ndarray:
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
    samples = np.empty((BOOTSTRAP_DRAWS, len(observed)), dtype=np.float64)
    for draw in range(BOOTSTRAP_DRAWS):
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


def reproduce_t2(data: Path) -> dict[str, Any]:
    bands = {"local": 0.005, "spillover": 0.001, "fixed": 0.001}
    model_results = {}
    extension_reasons: list[str] = []
    assay_failure = False
    global_classes: set[str] = set()
    all_pairwise_equivalent = True
    for model in MODELS:
        source_rows = read_jsonl(data / f"t2_{model}.jsonl")
        rows = []
        strata = []
        p_curves = []
        p_norm_rows = []
        h_rows = []
        h_norm_rows = []
        for row in source_rows:
            averaged = row["averaged_effects"]
            hard = row["hard_effect"]
            vector = []
            for _, left, right in T2_PAIRWISE:
                vector.extend(
                    float(averaged[left][endpoint])
                    - float(averaged[right][endpoint])
                    for endpoint in T2_ENDPOINTS
                )
            for operator in T2_SUPPORT:
                source = hard if operator == "H" else averaged[operator]
                vector.extend(float(source[endpoint]) for endpoint in T2_ENDPOINTS)
            rows.append(vector)
            strata.append(row["stratum"])
            p_curves.append(
                [
                    [float(effect[endpoint]) for endpoint in T2_ENDPOINTS]
                    for effect in row["p_curve_effects"]
                ]
            )
            p_norm_rows.append(
                [float(value) for value in row["p_immediate_message_norms"]]
            )
            h_rows.append([float(hard[endpoint]) for endpoint in T2_ENDPOINTS])
            h_norm_rows.append(float(row["h_immediate_message_norm"]))
        estimates, intervals, critical = t2_bootstrap_family(
            np.asarray(rows),
            np.asarray(p_curves),
            np.asarray(p_norm_rows),
            np.asarray(h_rows),
            np.asarray(h_norm_rows),
            np.asarray(strata),
            model=model,
        )
        names = (
            [
                f"{contrast}_{endpoint}"
                for contrast, _, _ in T2_PAIRWISE
                for endpoint in T2_ENDPOINTS
            ]
            + [
                f"{operator}_MINUS_BASELINE_{endpoint}"
                for operator in T2_SUPPORT
                for endpoint in T2_ENDPOINTS
            ]
            + [f"H_MINUS_P_CONTINUITY_{endpoint}" for endpoint in T2_ENDPOINTS]
        )
        summaries = {}
        endpoint_classes = {endpoint: set() for endpoint in T2_ENDPOINTS}
        model_assay = False
        for index, name in enumerate(names):
            endpoint = next(
                item for item in T2_ENDPOINTS if name.endswith(f"_{item}")
            )
            low, high = map(float, intervals[index])
            band = float(bands[endpoint])
            state = t2_classify(low, high, band)
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
        for endpoint in T2_ENDPOINTS:
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
            opposite_p = (
                ps["state"] == "DIFFERENT" and ps["estimate"] * ms["estimate"] < 0
            )
            opposite_d = (
                ds["state"] == "DIFFERENT" and ds["estimate"] * ms["estimate"] < 0
            )
            if (
                ms["state"] == "DIFFERENT"
                and (ps["state"] == "EQUIVALENT" or opposite_p)
                and (ds["state"] == "EQUIVALENT" or opposite_d)
            ):
                endpoint_classes[endpoint].add("MESSAGE")
            if disc == "DIFFERENT":
                endpoint_classes[endpoint].add("DISCONTINUITY")
            global_classes.update(endpoint_classes[endpoint])
        model_results[model] = {
            "n": len(source_rows),
            "simultaneous_critical_value": critical,
            "positive_control_pass": model_assay,
            "endpoint_classes": {
                key: sorted(value) for key, value in endpoint_classes.items()
            },
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
        "status": terminal,
        "bootstrap_draws": BOOTSTRAP_DRAWS,
        "models": model_results,
        "positive_control_pass": not assay_failure,
        "supported_mechanism_classes": sorted(global_classes),
        "all_pairwise_equivalent": all_pairwise_equivalent,
        "precision_extension_required": precision_extension_required,
        "precision_extension_authorized": False,
        "precision_extension_reasons": extension_reasons,
    }


def compare(
    actual: Any, expected: Any, path: str, mismatches: list[str], max_diff: list[float]
) -> None:
    if isinstance(actual, dict) and isinstance(expected, dict):
        for key, value in actual.items():
            if key not in expected:
                mismatches.append(f"{path}/{key}: missing in expected")
            else:
                compare(value, expected[key], f"{path}/{key}", mismatches, max_diff)
        return
    if isinstance(actual, list) and isinstance(expected, list):
        if len(actual) != len(expected):
            mismatches.append(f"{path}: length {len(actual)} != {len(expected)}")
            return
        for index, (left, right) in enumerate(zip(actual, expected, strict=True)):
            compare(left, right, f"{path}/{index}", mismatches, max_diff)
        return
    if (
        isinstance(actual, (int, float))
        and not isinstance(actual, bool)
        and isinstance(expected, (int, float))
        and not isinstance(expected, bool)
    ):
        difference = abs(float(actual) - float(expected))
        max_diff[0] = max(max_diff[0], difference)
        if difference > 1e-12:
            mismatches.append(f"{path}: {actual} != {expected} (diff={difference})")
        return
    if actual != expected:
        mismatches.append(f"{path}: {actual!r} != {expected!r}")


def validate(
    reproduced: dict[str, dict[str, Any]],
    *,
    frozen_root: Path | None = None,
    expected_root: Path | None = None,
) -> dict[str, Any]:
    if (frozen_root is None) == (expected_root is None):
        raise ValueError("provide exactly one validation root")
    if frozen_root is not None:
        targets = {
            "h4_d": frozen_root
            / "artifacts/iclr_h4/H4_D_SCIENTIFIC_AGGREGATE.json",
            "gate_c": frozen_root
            / "artifacts/iclr_h4/H4_GATE_C_SCIENTIFIC_AGGREGATE.json",
            "t1": frozen_root
            / "artifacts/iclr_transport/T1_PRE_READOUT_LOCALIZATION.json",
            "t2": frozen_root
            / "artifacts/iclr_transport/T2_CONFIRMATION_INITIAL_128_AGGREGATE.json",
        }
    else:
        assert expected_root is not None
        targets = {
            "h4_d": expected_root / "H4_D_SCIENTIFIC_AGGREGATE.json",
            "gate_c": expected_root / "H4_GATE_C_SCIENTIFIC_AGGREGATE.json",
            "t1": expected_root / "T1_PRE_READOUT_LOCALIZATION.json",
            "t2": expected_root / "T2_CONFIRMATION_INITIAL_128_AGGREGATE.json",
        }
    reports = {}
    for name, actual in reproduced.items():
        expected = read_json(targets[name])
        mismatches: list[str] = []
        max_diff = [0.0]
        compare(actual, expected, "", mismatches, max_diff)
        reports[name] = {
            "status": "PASS" if not mismatches else "FAIL",
            "maximum_absolute_numeric_difference": max_diff[0],
            "mismatch_count": len(mismatches),
            "mismatches": mismatches[:50],
        }
    return {
        "status": (
            "PASS_ANALYSIS_READY_EXACT_REPRODUCTION"
            if all(item["status"] == "PASS" for item in reports.values())
            else "FAIL_ANALYSIS_READY_REPRODUCTION"
        ),
        "tolerance": 1e-12,
        "results": reports,
        "V_F_read": False,
        "new_model_inference": False,
    }


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--data-root", type=Path, required=True)
    parser.add_argument("--output-root", type=Path, required=True)
    parser.add_argument("--frozen-project-root", type=Path)
    parser.add_argument("--expected-root", type=Path)
    args = parser.parse_args()

    data = args.data_root.resolve()
    output = args.output_root.resolve()
    reproduced = {
        "h4_d": reproduce_h4(data),
        "gate_c": reproduce_gate_c(data),
        "t1": reproduce_t1(data),
        "t2": reproduce_t2(data),
    }
    for name, value in reproduced.items():
        write_json(output / f"{name.upper()}_REPRODUCED.json", value)
    if args.frozen_project_root is not None and args.expected_root is not None:
        parser.error("use only one of --frozen-project-root and --expected-root")
    if args.frozen_project_root is not None or args.expected_root is not None:
        report = validate(
            reproduced,
            frozen_root=(
                args.frozen_project_root.resolve()
                if args.frozen_project_root is not None
                else None
            ),
            expected_root=(
                args.expected_root.resolve() if args.expected_root is not None else None
            ),
        )
        write_json(output / "REPRODUCTION_VALIDATION.json", report)
        print(json.dumps(report, indent=2))
        return 0 if report["status"].startswith("PASS_") else 1
    print(json.dumps({"status": "PASS_REPRODUCTION_COMPLETE_NO_VALIDATION"}, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
