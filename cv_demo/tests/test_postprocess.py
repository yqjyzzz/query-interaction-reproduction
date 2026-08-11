from cv_demo.inference.postprocess import remap_category_ids
from cv_demo.models.contracts import Detection


def test_category_remapping_preserves_geometry_and_score() -> None:
    source = Detection((1.0, 2.0, 3.0, 4.0), 0.8, 0)
    result = remap_category_ids([source], {0: 1})
    assert result == [Detection((1.0, 2.0, 3.0, 4.0), 0.8, 1)]
