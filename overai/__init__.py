"""OverAI: hierarchical supervised game-control imitation learning."""

from .config import ModelConfig
from .model import HierarchicalImitationController

__all__ = ["HierarchicalImitationController", "ModelConfig"]
