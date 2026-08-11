from cv_demo.evaluation.detection_metrics import (
    box_iou,
    combine_statistics,
    precision_recall_at_iou,
)
from cv_demo.models.contracts import Detection, GroundTruth


def test_box_iou_and_matching() -> None:
    assert box_iou((0, 0, 10, 10), (0, 0, 10, 10)) == 1.0
    result = precision_recall_at_iou(
        [Detection((0, 0, 10, 10), 0.9, 1), Detection((20, 20, 30, 30), 0.8, 1)],
        [GroundTruth((0, 0, 10, 10), 1)],
        iou_threshold=0.5,
    )
    assert result["true_positive"] == 1
    assert result["false_positive"] == 1
    assert result["recall"] == 1.0


def test_matching_is_class_aware() -> None:
    result = precision_recall_at_iou(
        [Detection((0, 0, 10, 10), 0.9, 2)],
        [GroundTruth((0, 0, 10, 10), 1)],
    )
    assert result["true_positive"] == 0
    assert result["false_positive"] == 1
    assert result["false_negative"] == 1


def test_statistics_are_combined_from_counts() -> None:
    result = combine_statistics(
        [
            {"true_positive": 2, "false_positive": 1, "false_negative": 0},
            {"true_positive": 1, "false_positive": 0, "false_negative": 2},
        ]
    )
    assert result["precision"] == 0.75
    assert result["recall"] == 0.6
