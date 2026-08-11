"""从本地 Hugging Face 目录加载 DETR。"""

from __future__ import annotations

from pathlib import Path
from typing import Any


def build_local_model(checkpoint: Path) -> tuple[Any, Any]:
    """仅从本地目录加载处理器和模型，不触发网络下载。"""

    try:
        from transformers import AutoImageProcessor, DetrForObjectDetection
    except ImportError as error:  # pragma: no cover - optional dependency
        raise RuntimeError(
            '需要 transformers；请安装 python -m pip install -e ".[huggingface]"'
        ) from error
    if not checkpoint.is_dir():
        raise ValueError("Hugging Face DETR checkpoint 必须是本地模型目录")
    processor = AutoImageProcessor.from_pretrained(
        str(checkpoint),
        local_files_only=True,
    )
    model = DetrForObjectDetection.from_pretrained(
        str(checkpoint),
        local_files_only=True,
    )
    return model, processor
