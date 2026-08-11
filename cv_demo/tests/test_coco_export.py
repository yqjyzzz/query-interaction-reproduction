from cv_demo.evaluation import to_coco_results
from cv_demo.models.contracts import Detection


def test_coco_export_uses_xywh_boxes() -> None:
    rows = to_coco_results(9, [Detection((1.0, 2.0, 6.0, 8.0), 0.75, 3)])
    assert rows == [
        {
            "image_id": 9,
            "category_id": 3,
            "bbox": [1.0, 2.0, 5.0, 6.0],
            "score": 0.75,
        }
    ]
