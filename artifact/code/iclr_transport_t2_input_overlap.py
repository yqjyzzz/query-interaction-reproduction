#!/usr/bin/env python3
"""Fail-closed T2 input integrity, overlap, and outcome-blind mapping."""

from __future__ import annotations

import hashlib
import json
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
CONFIG = ROOT / "configs/fq_iclr_transport_t2_input_overlap_v1.json"
RAW = ROOT / "artifacts/iclr_transport/t2_input_capture_raw"
INTEGRITY = ROOT / "artifacts/iclr_transport/T2_INPUT_CAPTURE_INTEGRITY.json"
OVERLAP = ROOT / "artifacts/iclr_transport/T2_REALIZED_DOSE_OVERLAP.json"
MAPPING = ROOT / "artifacts/iclr_transport/T2_OUTCOME_BLIND_MAPPING.json"
MODELS = ("detr_r50_500", "dino_r50_4s_12e")
ARMS = ("C_v2", "A", "B")
DOSES = ("0.25", "0.5", "0.75")
PAIRS = (("C_v2", "A"), ("C_v2", "B"), ("A", "B"))


def load(path: Path):
    return json.loads(path.read_text(encoding="utf-8"))


def sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def main() -> int:
    config = load(CONFIG)
    failures, receipts, by_model = [], [], {}
    for model in MODELS:
        paths = sorted((RAW / model).glob("*.json"))
        rows = [load(path) for path in paths]
        if len(rows) != 128 or len({row["image_id"] for row in rows}) != 128:
            failures.append(f"{model}:population")
        for path, row in zip(paths, rows):
            receipts.append({"path": path.relative_to(ROOT).as_posix(), "sha256": sha256(path)})
            if (
                row.get("status") != "PASS_T2_INPUT_ONLY_RECEIPT"
                or row.get("config_sha256") != sha256(CONFIG)
                or row.get("baseline_forward_count") != 1
                or row.get("intervention_forward") is not False
                or row.get("quality_measurement_computed") is not False
                or row.get("scientific_effect_readout") is not False
                or row.get("V_F_read") is not False
            ):
                failures.append(f"{model}:{path.name}:firewall")
        by_model[model] = rows
    integrity = {
        "schema_version": "fq.iclr.transport.t2_input_integrity.v1",
        "status": "PASS_T2_INPUT_CAPTURE_INTEGRITY" if not failures else "FAIL_T2_INPUT_CAPTURE_INTEGRITY",
        "config_sha256": sha256(CONFIG),
        "receipt_count": len(receipts),
        "receipts": receipts,
        "failures": failures,
        "intervention_forward": False,
        "scientific_effect_readout": False,
        "V_F_read": False,
    }
    INTEGRITY.write_text(json.dumps(integrity, indent=2) + "\n", encoding="utf-8")
    if failures:
        print(json.dumps(integrity, indent=2))
        return 1
    caliper = float(config["mapping"]["caliper"])
    draws = int(config["overlap_gate"]["bootstrap_draws"])
    model_results, mappings, all_pass = {}, {}, True
    for model in MODELS:
        rows = by_model[model]
        flat = []
        for row in rows:
            for arm in ARMS:
                for dose in DOSES:
                    flat.extend(np.log1p(row["realized_metrics"][arm][dose]["immediate_message_norm"]))
        scale = max(float(np.std(flat)), 1e-12)
        cover_columns, labels = [], []
        for left, right in PAIRS:
            for source, target in ((left, right), (right, left)):
                image_scores = []
                for row in rows:
                    source_values = np.asarray([
                        np.log1p(row["realized_metrics"][source][dose]["immediate_message_norm"])
                        for dose in DOSES
                    ])
                    target_values = np.asarray([
                        np.log1p(row["realized_metrics"][target][dose]["immediate_message_norm"])
                        for dose in DOSES
                    ])
                    nearest = np.min(
                        np.abs(source_values[:, :, None] - target_values.T[None, :, :]),
                        axis=2,
                    ) / scale
                    image_scores.append(float(np.mean(nearest <= caliper)))
                cover_columns.append(np.asarray(image_scores))
                labels.append(f"{source}->{target}")
        matrix = np.column_stack(cover_columns)
        means = matrix.mean(axis=0)
        strata = np.asarray([row["stratum"] for row in rows])
        groups = [np.flatnonzero(strata == label) for label in sorted(set(strata))]
        rng = np.random.default_rng(int.from_bytes(hashlib.sha256(f"T2-OVERLAP|{model}".encode()).digest()[:8], "big"))
        boot = np.empty((draws, matrix.shape[1]))
        for draw in range(draws):
            indices = np.concatenate([rng.choice(group, len(group), replace=True) for group in groups])
            boot[draw] = matrix[indices].mean(axis=0)
        se = boot.std(axis=0, ddof=1)
        safe = np.where(se > 0, se, 1.0)
        critical = float(np.quantile(np.max(np.abs((boot - means) / safe), axis=1), 0.95))
        lower = means - critical * se
        summaries = {
            label: {"coverage": float(means[i]), "simultaneous_lower": float(lower[i])}
            for i, label in enumerate(labels)
        }
        passed = all(
            value["coverage"] >= config["overlap_gate"]["bidirectional_nearest_neighbor_coverage_minimum"]
            and value["simultaneous_lower"] >= config["overlap_gate"]["simultaneous_lower_bound_minimum"]
            for value in summaries.values()
        )
        all_pass &= passed
        model_results[model] = {
            "n": len(rows),
            "pooled_log1p_sd": scale,
            "simultaneous_critical": critical,
            "directions": summaries,
            "pass": passed,
        }
        map_rows = []
        for row in rows:
            selected = {}
            for anchor in DOSES:
                anchor_values = np.log1p(row["realized_metrics"]["C_v2"][anchor]["immediate_message_norm"])
                for arm in ("A", "B"):
                    candidates = []
                    for dose in DOSES:
                        values = np.log1p(row["realized_metrics"][arm][dose]["immediate_message_norm"])
                        candidates.append((float(np.mean(np.abs(anchor_values - values) / scale)), float(dose)))
                    selected[f"{arm}_for_M_{anchor}"] = min(candidates)[1]
            map_rows.append({
                "image_id": row["image_id"],
                "image_sha256": row["image_sha256"],
                "stratum": row["stratum"],
                "selected_doses": selected,
            })
        mappings[model] = {"rows": map_rows}
    overlap = {
        "schema_version": "fq.iclr.transport.t2_overlap.v1",
        "status": "PASS_T2_REALIZED_DOSE_OVERLAP" if all_pass else "STOP_T2_INCOMPARABLE_REALIZED_DOSE",
        "config_sha256": sha256(CONFIG),
        "integrity_sha256": sha256(INTEGRITY),
        "models": model_results,
        "scientific_effect_readout": False,
        "V_F_read": False,
    }
    OVERLAP.write_text(json.dumps(overlap, indent=2) + "\n", encoding="utf-8")
    mapping = {
        "schema_version": "fq.iclr.transport.t2_mapping.v1",
        "status": "PASS_T2_OUTCOME_BLIND_MAPPING_FROZEN" if all_pass else "BLOCKED_T2_MAPPING_OVERLAP_FAIL",
        "config_sha256": sha256(CONFIG),
        "integrity_sha256": sha256(INTEGRITY),
        "overlap_sha256": sha256(OVERLAP),
        "models": mappings if all_pass else {},
        "scientific_outcome_used": False,
        "V_F_read": False,
    }
    MAPPING.write_text(json.dumps(mapping, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(overlap, indent=2))
    return 0 if all_pass else 3


if __name__ == "__main__":
    raise SystemExit(main())
