"""Small COCO detection reader with no dependency on pycocotools.

The reader is for the optional demo only. It is not used by the frozen paper
reproduction track.
"""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Iterator


Box = tuple[float, float, float, float]


@dataclass(frozen=True)
class CocoSample:
    """One image and its COCO-style ground-truth boxes."""

    image_id: int
    file_name: str
    boxes: tuple[Box, ...]
    labels: tuple[int, ...]


class CocoDetectionDataset:
    """Read COCO image metadata and load images on demand."""

    def __init__(self, annotation_file: str | Path, image_root: str | Path) -> None:
        self.annotation_file = Path(annotation_file)
        self.image_root = Path(image_root)
        payload = json.loads(self.annotation_file.read_text(encoding="utf-8"))
        self._images = {int(row["id"]): row for row in payload["images"]}
        grouped: dict[int, list[dict[str, Any]]] = {
            image_id: [] for image_id in self._images
        }
        for annotation in payload.get("annotations", []):
            image_id = int(annotation["image_id"])
            if image_id in grouped and not annotation.get("iscrowd", 0):
                grouped[image_id].append(annotation)
        self._samples = tuple(
            self._make_sample(image_id, grouped[image_id])
            for image_id in sorted(self._images)
        )

    def _make_sample(self, image_id: int, annotations: list[dict[str, Any]]) -> CocoSample:
        boxes: list[Box] = []
        labels: list[int] = []
        for annotation in annotations:
            x, y, width, height = map(float, annotation["bbox"])
            boxes.append((x, y, x + width, y + height))
            labels.append(int(annotation["category_id"]))
        return CocoSample(
            image_id=image_id,
            file_name=str(self._images[image_id]["file_name"]),
            boxes=tuple(boxes),
            labels=tuple(labels),
        )

    def __len__(self) -> int:
        return len(self._samples)

    def sample(self, index: int) -> CocoSample:
        return self._samples[index]

    def load_image(self, index: int) -> Any:
        """Load a PIL image only when the optional demo is actually run."""

        from PIL import Image

        sample = self.sample(index)
        path = self.image_root / sample.file_name
        if not path.exists():
            raise FileNotFoundError(f"image not found: {path}")
        return Image.open(path).convert("RGB")

    def iter_batches(self, batch_size: int) -> Iterator[list[int]]:
        if batch_size <= 0:
            raise ValueError("batch_size must be positive")
        for start in range(0, len(self), batch_size):
            yield list(range(start, min(start + batch_size, len(self))))
