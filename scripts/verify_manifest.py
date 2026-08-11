#!/usr/bin/env python3
"""Verify the SHA-256 manifest of the frozen artifact directory."""

from __future__ import annotations

import subprocess
import sys
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
ENGINE = ROOT / "artifact" / "code" / "p1_artifact_manifest.py"
ARTIFACT = ROOT / "artifact"


def main() -> int:
    command = [sys.executable, "-B", str(ENGINE), "--root", str(ARTIFACT), "--verify"]
    completed = subprocess.run(command, cwd=ROOT, check=False)
    return completed.returncode


if __name__ == "__main__":
    raise SystemExit(main())
