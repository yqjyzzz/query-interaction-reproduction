"""统一预测进入评估前的轻量后处理。"""

from __future__ import annotations

from collections.abc import Mapping, Sequence

from cv_demo.models.contracts import Detection


def remap_category_ids(
    detections: Sequence[Detection],
    category_id_map: Mapping[int, int],
) -> list[Detection]:
    """把模型类别编号映射到目标数据集的类别编号。"""

    return [
        Detection(
            box=detection.box,
            score=detection.score,
            label=category_id_map.get(detection.label, detection.label),
        )
        for detection in detections
    ]
