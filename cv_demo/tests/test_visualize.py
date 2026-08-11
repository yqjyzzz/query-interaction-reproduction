import pytest

from cv_demo.models.contracts import Detection


def test_visualization_preserves_image_size() -> None:
    pil = pytest.importorskip("PIL")
    from PIL import Image

    from cv_demo.inference.visualize import draw_detections

    image = Image.new("RGB", (32, 24), "white")
    result = draw_detections(image, [Detection((2, 3, 10, 12), 0.9, 1)])
    assert result.size == image.size
    assert result is not image
