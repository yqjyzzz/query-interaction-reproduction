"""Explicit builder loading for user-provided model checkpoints."""

from __future__ import annotations

import importlib
from pathlib import Path
from typing import Any, Callable


def load_callable(spec: str) -> Callable[..., Any]:
    """Load ``package.module:callable`` without hidden imports."""

    if ":" not in spec:
        raise ValueError("builder must use module.path:callable syntax")
    module_name, callable_name = spec.split(":", maxsplit=1)
    module = importlib.import_module(module_name)
    value = getattr(module, callable_name, None)
    if not callable(value):
        raise TypeError(f"builder is not callable: {spec}")
    return value


def build_model(builder_spec: str, checkpoint: str | Path) -> tuple[Any, Any]:
    """Call a user-supplied builder and require ``(model, processor)``."""

    checkpoint_path = Path(checkpoint)
    if not checkpoint_path.exists():
        raise FileNotFoundError(f"checkpoint not found: {checkpoint_path}")
    result = load_callable(builder_spec)(checkpoint_path)
    if not isinstance(result, tuple) or len(result) != 2:
        raise TypeError("builder must return (model, processor)")
    return result
