"""Decode DETR-like query outputs into a stable project-level contract."""

from __future__ import annotations

from typing import Any, Literal, Sequence

from .contracts import Detection


def decode_query_predictions(
    logits: Any,
    boxes: Any,
    image_sizes: Sequence[tuple[int, int]],
    *,
    score_threshold: float,
    no_object_index: int | None = None,
    activation: Literal["softmax", "sigmoid"] = "softmax",
) -> list[list[Detection]]:
    """Decode normalized ``cx, cy, width, height`` query outputs.

    The function expects tensors shaped ``[batch, queries, classes]`` and
    ``[batch, queries, 4]``. The last class is treated as no-object unless an
    explicit index is supplied. Model-specific loading stays outside this
    function so DETR and DINO adapters can share the same post-processing.
    """

    if logits.ndim != 3 or boxes.ndim != 3 or logits.shape[:2] != boxes.shape[:2]:
        raise ValueError("expected logits [B,Q,C] and boxes [B,Q,4]")
    if boxes.shape[-1] != 4 or len(image_sizes) != logits.shape[0]:
        raise ValueError("box shape or image_sizes does not match batch")
    if not 0.0 <= score_threshold <= 1.0:
        raise ValueError("score_threshold must be in [0, 1]")

    if activation == "softmax":
        no_object = logits.shape[-1] - 1 if no_object_index is None else no_object_index
        if not 0 <= no_object < logits.shape[-1]:
            raise ValueError("no_object_index is outside the class dimension")
        class_probabilities = logits.softmax(dim=-1).clone()
        class_probabilities[..., no_object] = -1.0
    elif activation == "sigmoid":
        class_probabilities = logits.sigmoid()
    else:
        raise ValueError(f"unsupported activation: {activation}")
    scores, labels = class_probabilities.max(dim=-1)
    decoded: list[list[Detection]] = []
    for batch_index, (height, width) in enumerate(image_sizes):
        query_boxes = boxes[batch_index]
        query_scores = scores[batch_index]
        query_labels = labels[batch_index]
        keep = query_scores >= score_threshold
        selected_boxes = query_boxes[keep].detach().cpu()
        selected_scores = query_scores[keep].detach().cpu()
        selected_labels = query_labels[keep].detach().cpu()
        image_detections: list[Detection] = []
        for box, score, label in zip(
            selected_boxes, selected_scores, selected_labels, strict=True
        ):
            center_x, center_y, box_width, box_height = map(float, box)
            x1 = max(0.0, (center_x - box_width / 2.0) * width)
            y1 = max(0.0, (center_y - box_height / 2.0) * height)
            x2 = min(float(width), (center_x + box_width / 2.0) * width)
            y2 = min(float(height), (center_y + box_height / 2.0) * height)
            image_detections.append(
                Detection(
                    box=(x1, y1, x2, y2),
                    score=float(score),
                    label=int(label),
                )
            )
        decoded.append(image_detections)
    return decoded
