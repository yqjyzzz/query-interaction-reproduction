#!/usr/bin/env python3
"""Fail-closed integrity audit for T2 confirmation v2 raw receipts.

This tool intentionally does not compute or compare scientific outcomes.
"""

from __future__ import annotations

import argparse
import json
import math
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parents[1]
EXECUTION = ROOT / "configs/fq_iclr_transport_t2_confirmation_execution_v2.json"
CONFIRMATION = ROOT / "configs/fq_iclr_transport_t2_confirmation_v1.json"
PAIRS = ROOT / "artifacts/iclr_transport/T2_CONFIRMATION_PAIR_MANIFEST.json"
PAIR_INTEGRITY = ROOT / "artifacts/iclr_transport/T2_PAIR_FREEZE_INTEGRITY.json"
MESSAGE_V2 = ROOT / "artifacts/iclr_transport/T2_MESSAGE_V2_R_INTEGRITY.json"
INPUT_INTEGRITY = ROOT / "artifacts/iclr_transport/T2_INPUT_CAPTURE_INTEGRITY.json"
OVERLAP = ROOT / "artifacts/iclr_transport/T2_REALIZED_DOSE_OVERLAP.json"
MAPPING = ROOT / "artifacts/iclr_transport/T2_OUTCOME_BLIND_MAPPING.json"
RAW = ROOT / "artifacts/iclr_transport/t2_confirmation_raw_initial_v2"
LOGS = ROOT / "artifacts/iclr_transport/t2_confirmation_v2_logs"
OUTPUT = ROOT / "artifacts/iclr_transport/T2_CONFIRMATION_V2_FULL_INTEGRITY.json"
MODELS = ("detr_r50_500", "dino_r50_4s_12e")
ARMS = {
    "baseline", "A_0p25", "A_0p5", "A_0p75", "B_0p25", "B_0p5",
    "B_0p75", "C_v2_0p25", "C_v2_0p5", "C_v2_0p75", "hard",
}
AUDITS = ARMS - {"baseline"}


def load(path: Path) -> Any:
    return json.loads(path.read_text(encoding="utf-8"))


def sha256_file(path: Path) -> str:
    import hashlib

    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def all_finite(value: Any) -> bool:
    if isinstance(value, bool) or value is None or isinstance(value, str):
        return True
    if isinstance(value, (int, float)):
        return math.isfinite(float(value))
    if isinstance(value, list):
        return all(all_finite(item) for item in value)
    if isinstance(value, dict):
        return all(all_finite(item) for item in value.values())
    return False


def audit(root: Path = ROOT) -> dict[str, Any]:
    execution = load(EXECUTION)
    pairs = load(PAIRS)
    execution_hash = sha256_file(EXECUTION)
    bound_hashes = {
        "execution_sha256": execution_hash,
        "confirmation_sha256": sha256_file(CONFIRMATION),
        "pair_manifest_sha256": sha256_file(PAIRS),
        "message_v2_integrity_sha256": sha256_file(MESSAGE_V2),
        "input_integrity_sha256": sha256_file(INPUT_INTEGRITY),
        "overlap_sha256": sha256_file(OVERLAP),
        "mapping_sha256": sha256_file(MAPPING),
    }
    prerequisite_checks = {
        "confirmation_hash": execution["confirmation_config_sha256"] == bound_hashes["confirmation_sha256"],
        "pair_manifest_hash": execution["pair_manifest_sha256"] == bound_hashes["pair_manifest_sha256"],
        "pair_integrity_hash": execution["pair_integrity_sha256"] == sha256_file(PAIR_INTEGRITY),
        "message_v2_hash": execution["message_v2_integrity_sha256"] == bound_hashes["message_v2_integrity_sha256"],
        "input_integrity_hash": execution["input_integrity_sha256"] == bound_hashes["input_integrity_sha256"],
        "overlap_hash": execution["realized_overlap_sha256"] == bound_hashes["overlap_sha256"],
        "mapping_hash": execution["outcome_blind_mapping_sha256"] == bound_hashes["mapping_sha256"],
        "overlap_pass": load(OVERLAP).get("status") == "PASS_T2_REALIZED_DOSE_OVERLAP",
        "mapping_pass": load(MAPPING).get("status") == "PASS_T2_OUTCOME_BLIND_MAPPING_FROZEN",
        "precision_sealed": execution.get("precision_extension_authorized") is False,
        "aggregation_firewall": execution.get("aggregation_before_full_integrity") is False,
        "V_F_sealed": execution.get("V_read") is False and execution.get("F_read") is False,
    }
    model_reports: dict[str, Any] = {}
    global_failures: list[str] = []
    for model in MODELS:
        expected_rows = pairs["models"][model]["initial"]
        expected = {row["image_id"]: row for row in expected_rows}
        directory = RAW / model
        files = sorted(directory.glob("*.json")) if directory.is_dir() else []
        actual_ids = {path.stem for path in files}
        failures: list[str] = []
        if actual_ids != set(expected):
            failures.append("population_file_set_mismatch")
        if len(files) != 128:
            failures.append(f"receipt_count_{len(files)}")
        checkpoint = expected_rows[0]["checkpoint_sha256"]
        for path in files:
            try:
                receipt = load(path)
            except Exception as error:
                failures.append(f"{path.name}:json:{type(error).__name__}")
                continue
            image_id = path.stem
            row = expected.get(image_id)
            checks = [
                receipt.get("schema_version") == "fq.iclr.transport.t2_confirmation_raw.v1",
                receipt.get("status") == "PASS_T2_CONFIRMATION_RAW_NO_AGGREGATE",
                receipt.get("model_id") == model,
                receipt.get("image_id") == image_id,
                receipt.get("checkpoint_sha256") == checkpoint,
                all(receipt.get(key) == value for key, value in bound_hashes.items()),
                row is not None and receipt.get("image_sha256") == row.get("image_sha256"),
                row is not None and receipt.get("stratum") == row.get("stratum"),
                row is not None and receipt.get("pair") == row.get("pair"),
                set(receipt.get("operator_audits", {})) == AUDITS,
                all(item.get("pass") is True for item in receipt.get("operator_audits", {}).values()),
                set(receipt.get("arm_measurements", {})) == ARMS,
                set(receipt.get("native_measurements", {})) == ARMS,
                receipt.get("forward_count") == 11,
                receipt.get("cross_arm_effect_computed") is False,
                receipt.get("scientific_effect_readout") is False,
                receipt.get("aggregation_started") is False,
                receipt.get("precision_extension_read") is False,
                receipt.get("V_F_read") is False,
                all_finite(receipt),
            ]
            if not all(checks):
                failures.append(f"{path.name}:binding_or_structure")
        rc_path = LOGS / f"{model}.rc"
        rc_ok = rc_path.is_file() and rc_path.read_text(encoding="utf-8").strip() == "0"
        if not rc_ok:
            failures.append("model_rc_not_zero")
        model_reports[model] = {
            "expected_receipts": 128,
            "observed_receipts": len(files),
            "unique_image_ids": len(actual_ids),
            "checkpoint_sha256": checkpoint,
            "rc_zero": rc_ok,
            "failure_count": len(failures),
            "failures": failures[:50],
        }
        global_failures.extend(f"{model}:{item}" for item in failures)
    serial_rc = LOGS / "serial.rc"
    serial_ok = serial_rc.is_file() and serial_rc.read_text(encoding="utf-8").strip() == "0"
    if not serial_ok:
        global_failures.append("serial_rc_not_zero")
    if not all(prerequisite_checks.values()):
        global_failures.append("prerequisite_binding_failure")
    passed = not global_failures
    return {
        "schema_version": "fq.iclr.transport.t2_confirmation_full_integrity.v1",
        "status": "PASS_T2_CONFIRMATION_V2_FULL_INTEGRITY_NO_AGGREGATE" if passed else "FAIL_T2_CONFIRMATION_V2_FULL_INTEGRITY",
        "generated_at": datetime.now(timezone.utc).isoformat(),
        "execution_sha256": execution_hash,
        "prerequisite_checks": prerequisite_checks,
        "models": model_reports,
        "serial_rc_zero": serial_ok,
        "total_expected_receipts": 256,
        "total_observed_receipts": sum(report["observed_receipts"] for report in model_reports.values()),
        "scientific_effect_readout": False,
        "aggregation_started": False,
        "precision_extension_read": False,
        "V_F_read": False,
        "failure_count": len(global_failures),
        "failures": global_failures,
    }


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--execute", action="store_true")
    args = parser.parse_args()
    report = audit()
    if args.execute:
        OUTPUT.write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(report, indent=2))
    return 0 if report["status"].startswith("PASS_") else 1


if __name__ == "__main__":
    raise SystemExit(main())
