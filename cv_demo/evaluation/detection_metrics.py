"""用于独立检测流程的透明评估实现。

这里的固定阈值匹配便于调试和失败样本分析，不替代 COCO 官方评估，
也不参与论文冻结结果的计算。
"""

from __future__ import annotations

from collections.abc import Sequence

from cv_demo.models.contracts import Detection, GroundTruth


def box_iou(left: tuple[float, float, float, float], right: tuple[float, float, float, float]) -> float:
    x1 = max(left[0], right[0])
    y1 = max(left[1], right[1])
    x2 = min(left[2], right[2])
    y2 = min(left[3], right[3])
    intersection = max(0.0, x2 - x1) * max(0.0, y2 - y1)
    left_area = max(0.0, left[2] - left[0]) * max(0.0, left[3] - left[1])
    right_area = max(0.0, right[2] - right[0]) * max(0.0, right[3] - right[1])
    union = left_area + right_area - intersection
    return intersection / union if union else 0.0


def precision_recall_at_iou(
    predictions: Sequence[Detection],
    ground_truth: Sequence[GroundTruth],
    *,
    iou_threshold: float = 0.5,
) -> dict[str, float | int]:
    """按置信度执行类别感知的贪心匹配，返回 TP、FP、FN。"""

    if not 0.0 <= iou_threshold <= 1.0:
        raise ValueError("iou_threshold must be in [0, 1]")
    matched: set[int] = set()
    true_positive = 0
    for prediction in sorted(predictions, key=lambda item: item.score, reverse=True):
        candidates = [
            (index, box_iou(prediction.box, target.box))
            for index, target in enumerate(ground_truth)
            if index not in matched and prediction.label == target.label
        ]
        if candidates:
            best_index, best_iou = max(candidates, key=lambda item: item[1])
            if best_iou >= iou_threshold:
                matched.add(best_index)
                true_positive += 1
    false_positive = len(predictions) - true_positive
    false_negative = len(ground_truth) - true_positive
    precision = true_positive / (true_positive + false_positive) if predictions else 0.0
    recall = true_positive / (true_positive + false_negative) if ground_truth else 0.0
    return {
        "true_positive": true_positive,
        "false_positive": false_positive,
        "false_negative": false_negative,
        "precision": precision,
        "recall": recall,
        "iou_threshold": iou_threshold,
    }


def combine_statistics(rows: Sequence[dict[str, float | int]]) -> dict[str, float | int]:
    """汇总逐图像计数，并由总计数重新计算精确率与召回率。"""

    true_positive = sum(int(row["true_positive"]) for row in rows)
    false_positive = sum(int(row["false_positive"]) for row in rows)
    false_negative = sum(int(row["false_negative"]) for row in rows)
    precision_denominator = true_positive + false_positive
    recall_denominator = true_positive + false_negative
    return {
        "true_positive": true_positive,
        "false_positive": false_positive,
        "false_negative": false_negative,
        "precision": true_positive / precision_denominator if precision_denominator else 0.0,
        "recall": true_positive / recall_denominator if recall_denominator else 0.0,
    }
