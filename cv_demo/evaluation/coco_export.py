"""将统一预测对象转换为 COCO detection result 格式。"""

from __future__ import annotations

from collections.abc import Sequence

from cv_demo.models.contracts import Detection


def to_coco_results(image_id: int, detections: Sequence[Detection]) -> list[dict[str, object]]:
    """输出可交给 COCO API 的 ``xywh`` 预测记录。"""

    rows: list[dict[str, object]] = []
    for detection in detections:
        x1, y1, x2, y2 = detection.box
        rows.append(
            {
                "image_id": image_id,
                "category_id": detection.label,
                "bbox": [x1, y1, max(0.0, x2 - x1), max(0.0, y2 - y1)],
                "score": detection.score,
            }
        )
    return rows
