#!/usr/bin/env python3
"""Run the preserved aggregate reproducer with repository-relative defaults."""

from __future__ import annotations

import subprocess
import sys
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
ENGINE = ROOT / "artifact" / "code" / "p1_reproduce_from_analysis_ready.py"
DATA = ROOT / "artifact" / "analysis_ready"
EXPECTED = ROOT / "artifact" / "expected"
OUTPUT = ROOT / "results" / "reproduced"


def main() -> int:
    command = [
        sys.executable,
        "-B",
        str(ENGINE),
        "--data-root",
        str(DATA),
        "--output-root",
        str(OUTPUT),
        "--expected-root",
        str(EXPECTED),
    ]
    completed = subprocess.run(command, cwd=ROOT, check=False)
    return completed.returncode


if __name__ == "__main__":
    raise SystemExit(main())
