"""DINO adapter for the optional end-to-end CV demonstration."""

from __future__ import annotations

from typing import Any

from .base_wrapper import SetDetectorWrapper


class DinoWrapper(SetDetectorWrapper):
    """Adapter for a DINO-like model returning ``pred_logits`` and ``pred_boxes``."""

    def __init__(
        self,
        model: Any,
        processor: Any,
        *,
        device: str = "cpu",
        score_threshold: float = 0.5,
    ) -> None:
        super().__init__(
            model,
            processor,
            device=device,
            score_threshold=score_threshold,
            logits_key="pred_logits",
            boxes_key="pred_boxes",
            activation="sigmoid",
        )
