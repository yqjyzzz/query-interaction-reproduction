"""DETR adapter for the optional end-to-end CV demonstration."""

from __future__ import annotations

from typing import Any

from .base_wrapper import SetDetectorWrapper


class DetrWrapper(SetDetectorWrapper):
    """Adapter for a DETR-like model returning ``logits`` and ``pred_boxes``."""

    def __init__(
        self,
        model: Any,
        processor: Any,
        *,
        device: str = "cpu",
        score_threshold: float = 0.5,
        no_object_index: int | None = None,
    ) -> None:
        super().__init__(
            model,
            processor,
            device=device,
            score_threshold=score_threshold,
            logits_key="logits",
            boxes_key="pred_boxes",
            no_object_index=no_object_index,
            activation="softmax",
        )
