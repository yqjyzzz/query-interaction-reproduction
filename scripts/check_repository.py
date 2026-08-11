#!/usr/bin/env python3
"""依次执行冻结工件校验和仓库测试。"""

from __future__ import annotations

import json
import subprocess
import sys
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]


def run_step(name: str, command: list[str]) -> dict[str, object]:
    """执行单个检查步骤并保留退出码。"""

    completed = subprocess.run(command, cwd=ROOT, check=False)
    return {"name": name, "returncode": completed.returncode}


def main() -> int:
    steps = [
        run_step(
            "artifact_manifest",
            [sys.executable, "scripts/verify_manifest.py"],
        ),
        run_step(
            "exact_reproduction",
            [sys.executable, "scripts/run_reproduction.py"],
        ),
        run_step(
            "repository_tests",
            [
                sys.executable,
                "-m",
                "pytest",
                "-k",
                "not test_fresh_reproduction_matches_frozen_outputs",
            ],
        ),
    ]
    passed = all(step["returncode"] == 0 for step in steps)
    print(
        json.dumps(
            {
                "status": "PASS_REPOSITORY_CHECK" if passed else "FAIL_REPOSITORY_CHECK",
                "steps": steps,
            },
            ensure_ascii=False,
        )
    )
    return 0 if passed else 1


if __name__ == "__main__":
    raise SystemExit(main())
