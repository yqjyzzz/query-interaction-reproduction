"""Shared runtime wrapper for DETR-like model outputs."""

from __future__ import annotations

from typing import Any, Literal, Sequence

from .common import decode_query_predictions
from .contracts import Detection


class SetDetectorWrapper:
    """Run a processor/model pair and decode query predictions.

    A user-supplied builder must return ``(model, processor)``. The model is
    expected to return logits and normalized query boxes under the configured
    output names. This keeps checkpoint loading explicit and avoids pretending
    that a public demo checkpoint is the paper's frozen checkpoint.
    """

    def __init__(
        self,
        model: Any,
        processor: Any,
        *,
        device: str,
        score_threshold: float,
        logits_key: str,
        boxes_key: str,
        no_object_index: int | None = None,
        activation: Literal["softmax", "sigmoid"] = "softmax",
    ) -> None:
        try:
            import torch
        except ImportError as error:  # pragma: no cover - optional dependency
            raise RuntimeError("cv_demo requires torch; install cv_demo/requirements-demo.txt") from error
        self.model = model.to(device).eval()
        self.processor = processor
        self.device = device
        self.score_threshold = score_threshold
        self.logits_key = logits_key
        self.boxes_key = boxes_key
        self.no_object_index = no_object_index
        self.activation = activation
        self._torch = torch

    def _move_inputs(self, inputs: Any) -> Any:
        if hasattr(inputs, "to"):
            return inputs.to(self.device)
        if isinstance(inputs, dict):
            return {
                key: value.to(self.device) if hasattr(value, "to") else value
                for key, value in inputs.items()
            }
        raise TypeError("processor output must be a mapping or expose .to(device)")

    @staticmethod
    def _get(output: Any, key: str) -> Any:
        if isinstance(output, dict):
            return output[key]
        if hasattr(output, key):
            return getattr(output, key)
        raise KeyError(f"model output does not contain {key!r}")

    def predict_batch(self, images: Sequence[Any]) -> list[list[Detection]]:
        if not images:
            return []
        inputs = self._move_inputs(
            self.processor(images=list(images), return_tensors="pt")
        )
        with self._torch.inference_mode():
            output = self.model(**inputs)
        logits = self._get(output, self.logits_key)
        boxes = self._get(output, self.boxes_key)
        sizes = [(int(image.height), int(image.width)) for image in images]
        return decode_query_predictions(
            logits,
            boxes,
            sizes,
            score_threshold=self.score_threshold,
            no_object_index=self.no_object_index,
            activation=self.activation,
        )
