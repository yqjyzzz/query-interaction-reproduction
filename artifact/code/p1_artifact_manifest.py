#!/usr/bin/env python3
"""Write or verify the SHA-256 manifest for the TMLR artifact directory."""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
from typing import Any


MANIFEST_NAME = "ARTIFACT_MANIFEST.json"


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def inventory(root: Path) -> list[dict[str, Any]]:
    rows = []
    for path in sorted(item for item in root.rglob("*") if item.is_file()):
        relative = path.relative_to(root).as_posix()
        if relative == MANIFEST_NAME or relative.startswith("reproduced_selftest/"):
            continue
        rows.append(
            {
                "path": relative,
                "bytes": path.stat().st_size,
                "sha256": sha256(path),
            }
        )
    return rows


def write_manifest(root: Path) -> dict[str, Any]:
    rows = inventory(root)
    value = {
        "schema_version": "tmlr.query_intervention_audit.artifact_manifest.v1",
        "status": "FROZEN_TMLR_ARTIFACT_MANIFEST",
        "file_count": len(rows),
        "total_bytes_excluding_manifest": sum(row["bytes"] for row in rows),
        "files": rows,
        "V_F_included": False,
        "model_weights_included": False,
        "image_pixels_included": False,
    }
    (root / MANIFEST_NAME).write_text(
        json.dumps(value, indent=2) + "\n", encoding="utf-8"
    )
    return value


def verify_manifest(root: Path) -> dict[str, Any]:
    manifest = json.loads((root / MANIFEST_NAME).read_text(encoding="utf-8"))
    failures = []
    expected_paths = {row["path"] for row in manifest["files"]}
    observed = {row["path"]: row for row in inventory(root)}
    for row in manifest["files"]:
        current = observed.get(row["path"])
        if current is None:
            failures.append(f"missing:{row['path']}")
        elif current["bytes"] != row["bytes"]:
            failures.append(f"size:{row['path']}")
        elif current["sha256"] != row["sha256"]:
            failures.append(f"sha256:{row['path']}")
    for path in sorted(set(observed) - expected_paths):
        failures.append(f"unregistered:{path}")
    return {
        "status": "PASS_ARTIFACT_MANIFEST" if not failures else "FAIL_ARTIFACT_MANIFEST",
        "checked_files": len(manifest["files"]),
        "failures": failures,
    }


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--root", type=Path, required=True)
    action = parser.add_mutually_exclusive_group(required=True)
    action.add_argument("--write", action="store_true")
    action.add_argument("--verify", action="store_true")
    args = parser.parse_args()
    root = args.root.resolve()
    result = write_manifest(root) if args.write else verify_manifest(root)
    print(json.dumps(result, indent=2))
    return 0 if not result["status"].startswith("FAIL_") else 1


if __name__ == "__main__":
    raise SystemExit(main())
