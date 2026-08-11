from __future__ import annotations

import json
import subprocess
import sys
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]


def test_fresh_reproduction_matches_frozen_outputs() -> None:
    completed = subprocess.run(
        [sys.executable, "scripts/run_reproduction.py"],
        cwd=ROOT,
        capture_output=True,
        text=True,
        check=False,
    )
    assert completed.returncode == 0, completed.stdout + completed.stderr
    receipt = json.loads(
        (ROOT / "results" / "reproduced" / "REPRODUCTION_VALIDATION.json").read_text(
            encoding="utf-8"
        )
    )
    assert receipt["status"] == "PASS_ANALYSIS_READY_EXACT_REPRODUCTION"
    assert all(
        result["status"] == "PASS" for result in receipt["results"].values()
    )
