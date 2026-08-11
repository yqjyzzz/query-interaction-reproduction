from __future__ import annotations

import json
import subprocess
import sys
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]


def test_manifest_verifies() -> None:
    completed = subprocess.run(
        [sys.executable, "scripts/verify_manifest.py"],
        cwd=ROOT,
        capture_output=True,
        text=True,
        check=False,
    )
    assert completed.returncode == 0, completed.stdout + completed.stderr
    assert "PASS_ARTIFACT_MANIFEST" in completed.stdout


def test_archived_receipt_preserves_firewall() -> None:
    receipt = json.loads(
        (ROOT / "artifact" / "reproduced" / "REPRODUCTION_VALIDATION.json").read_text(
            encoding="utf-8"
        )
    )
    assert receipt["status"] == "PASS_ANALYSIS_READY_EXACT_REPRODUCTION"
    assert receipt["V_F_read"] is False
    assert receipt["new_model_inference"] is False
