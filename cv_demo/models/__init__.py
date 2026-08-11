"""Model adapters for the optional CV demonstration."""

from .contracts import Detection
from .detr_wrapper import DetrWrapper
from .dino_wrapper import DinoWrapper

__all__ = ["Detection", "DetrWrapper", "DinoWrapper"]
