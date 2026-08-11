"""Draw detections on images for qualitative inspection."""

from __future__ import annotations

from collections.abc import Sequence
from pathlib import Path
from typing import Any

from cv_demo.models.contracts import Detection


def draw_detections(
    image: Any,
    detections: Sequence[Detection],
    *,
    labels: dict[int, str] | None = None,
) -> Any:
    """Return a copy of a PIL image with boxes and confidence labels."""

    from PIL import ImageDraw

    canvas = image.copy()
    draw = ImageDraw.Draw(canvas)
    for detection in detections:
        x1, y1, x2, y2 = detection.box
        name = labels.get(detection.label, str(detection.label)) if labels else str(detection.label)
        text = f"{name}:{detection.score:.2f}"
        draw.rectangle((x1, y1, x2, y2), outline=(0, 210, 120), width=2)
        draw.text((x1 + 2, max(0.0, y1 - 14)), text, fill=(0, 210, 120))
    return canvas


def save_detections(
    image: Any,
    detections: Sequence[Detection],
    output_path: str | Path,
    *,
    labels: dict[int, str] | None = None,
) -> None:
    output = Path(output_path)
    output.parent.mkdir(parents=True, exist_ok=True)
    draw_detections(image, detections, labels=labels).save(output)
