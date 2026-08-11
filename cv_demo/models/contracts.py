"""模型、可视化与评估模块共享的数据契约。"""

from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True)
class Detection:
    """像素坐标系中的预测框，格式为 ``x1, y1, x2, y2``。"""

    box: tuple[float, float, float, float]
    score: float
    label: int


@dataclass(frozen=True)
class GroundTruth:
    """像素坐标系中的真实标注及其类别。"""

    box: tuple[float, float, float, float]
    label: int
