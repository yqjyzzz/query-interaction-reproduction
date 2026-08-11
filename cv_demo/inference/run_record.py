"""生成不泄露本机绝对路径的运行记录。"""

from __future__ import annotations

import hashlib
from datetime import datetime, timezone
from pathlib import Path
from typing import Any


def sha256_file(path: str | Path) -> str:
    """计算文件的 SHA-256。"""

    digest = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def describe_path(path: str | Path) -> dict[str, str | int]:
    """记录文件或目录的大小与摘要，不保存本机绝对路径。"""

    value = Path(path)
    if value.is_file():
        return {
            "kind": "file",
            "name": value.name,
            "size_bytes": value.stat().st_size,
            "sha256": sha256_file(value),
        }
    if value.is_dir():
        digest = hashlib.sha256()
        total_size = 0
        files = sorted(path for path in value.rglob("*") if path.is_file())
        for file_path in files:
            relative = file_path.relative_to(value).as_posix()
            size = file_path.stat().st_size
            total_size += size
            digest.update(relative.encode("utf-8"))
            digest.update(b"\0")
            digest.update(sha256_file(file_path).encode("ascii"))
            digest.update(b"\0")
        return {
            "kind": "directory",
            "name": value.name,
            "file_count": len(files),
            "size_bytes": total_size,
            "sha256": digest.hexdigest(),
        }
    raise FileNotFoundError(f"path not found: {value}")


def build_run_record(
    *,
    config_path: str | Path,
    config: dict[str, Any],
    builder_spec: str,
    resolved_device: str,
    image_count: int,
) -> dict[str, Any]:
    """记录一次推理所需的关键条件和输入摘要。"""

    return {
        "schema_version": 1,
        "created_at_utc": datetime.now(timezone.utc).isoformat(),
        "track": "independent_cv_demo",
        "paper_artifact_written": False,
        "backend": str(config.get("backend", "detr")).lower(),
        "builder": builder_spec,
        "device": resolved_device,
        "seed": int(config.get("seed", 7)),
        "batch_size": int(config.get("batch_size", 1)),
        "score_threshold": float(config.get("score_threshold", 0.5)),
        "iou_threshold": float(config.get("iou_threshold", 0.5)),
        "image_count": image_count,
        "config": describe_path(config_path),
        "checkpoint": describe_path(config["checkpoint"]),
        "annotations": describe_path(config["annotations"]),
    }
