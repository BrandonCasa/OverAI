"""Model configuration and validation."""

from __future__ import annotations

import json
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any


@dataclass(frozen=True, slots=True)
class ModelConfig:
    """Configuration for the hierarchical imitation controller.

    Defaults implement the 720p, 30/5/60 Hz architecture for the current
    two-channel visual input and eight-button control profile. Tests and
    experiments can use smaller, structurally identical configurations.
    """

    image_height: int = 720
    image_width: int = 1280
    input_channels: int = 2
    channel_order: str = "RB"
    patch_size: int = 40
    grid_height: int = 18
    grid_width: int = 32

    video_hz: int = 30
    slow_hz: int = 5
    fast_hz: int = 60
    slow_horizon: int = 10
    fast_horizon: int = 120

    vision_dim: int = 256
    model_dim: int = 320
    controller_dim: int = 320
    vision_layers: int = 4
    fusion_layers: int = 2
    decoder_layers: int = 1
    num_heads: int = 8

    window_height: int = 6
    window_width: int = 8
    dropout: float = 0.1
    gradient_checkpointing: bool = True

    frame_summary_tokens: int = 4
    recent_entries: int = 60
    intermediate_entries: int = 40
    long_entries: int = 20
    recent_tokens_per_entry: int = 4
    intermediate_tokens_per_entry: int = 2
    long_tokens_per_entry: int = 2
    frames_per_intermediate: int = 6
    intermediate_per_long: int = 5

    control_query_tokens: int = 12
    action_dim: int = 96
    trajectory_summary_tokens: int = 4
    num_buttons: int = 8
    compressor_layers: int = 1

    def __post_init__(self) -> None:
        positive_fields = (
            "image_height",
            "image_width",
            "input_channels",
            "patch_size",
            "grid_height",
            "grid_width",
            "video_hz",
            "slow_hz",
            "fast_hz",
            "slow_horizon",
            "fast_horizon",
            "vision_dim",
            "model_dim",
            "controller_dim",
            "vision_layers",
            "fusion_layers",
            "decoder_layers",
            "num_heads",
            "window_height",
            "window_width",
            "frame_summary_tokens",
            "recent_entries",
            "intermediate_entries",
            "long_entries",
            "recent_tokens_per_entry",
            "intermediate_tokens_per_entry",
            "long_tokens_per_entry",
            "frames_per_intermediate",
            "intermediate_per_long",
            "control_query_tokens",
            "action_dim",
            "trajectory_summary_tokens",
            "num_buttons",
            "compressor_layers",
        )
        for name in positive_fields:
            if getattr(self, name) <= 0:
                raise ValueError(f"{name} must be positive")

        expected_grid = (
            self.image_height // self.patch_size,
            self.image_width // self.patch_size,
        )
        if self.image_height % self.patch_size or self.image_width % self.patch_size:
            raise ValueError("image dimensions must be divisible by patch_size")
        if expected_grid != (self.grid_height, self.grid_width):
            raise ValueError(
                "grid dimensions must match image dimensions divided by patch_size: "
                f"expected {expected_grid}, got {(self.grid_height, self.grid_width)}"
            )
        if self.input_channels != 2 or self.channel_order != "RB":
            raise ValueError("OverAI requires two input channels in RB order")
        if self.fast_hz % self.video_hz:
            raise ValueError("fast_hz must be an integer multiple of video_hz")
        if self.fast_hz % self.slow_hz:
            raise ValueError("fast_hz must be an integer multiple of slow_hz")
        if self.model_dim % self.num_heads or self.vision_dim % self.num_heads:
            raise ValueError("model_dim and vision_dim must be divisible by num_heads")
        if self.vision_dim % 4:
            raise ValueError("vision_dim must be divisible by four")
        if self.window_height > self.grid_height or self.window_width > self.grid_width:
            raise ValueError("attention windows cannot exceed the patch grid")
        if self.frame_summary_tokens != self.recent_tokens_per_entry:
            raise ValueError(
                "frame_summary_tokens and recent_tokens_per_entry describe the same "
                "entry width and must match"
            )
        if not 0.0 <= self.dropout < 1.0:
            raise ValueError("dropout must be in [0, 1)")

    @property
    def grid_tokens(self) -> int:
        return self.grid_height * self.grid_width

    @property
    def fast_ticks_per_video(self) -> int:
        return self.fast_hz // self.video_hz

    @property
    def fast_ticks_per_slow(self) -> int:
        return self.fast_hz // self.slow_hz

    @property
    def memory_tokens(self) -> int:
        return (
            self.recent_entries * self.recent_tokens_per_entry
            + self.intermediate_entries * self.intermediate_tokens_per_entry
            + self.long_entries * self.long_tokens_per_entry
        )

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)

    def to_json(self, path: str | Path) -> None:
        Path(path).write_text(
            json.dumps(self.to_dict(), indent=2) + "\n", encoding="utf-8"
        )

    @classmethod
    def from_json(cls, path: str | Path) -> ModelConfig:
        data = json.loads(Path(path).read_text(encoding="utf-8"))
        if not isinstance(data, dict):
            raise TypeError("model config must contain a JSON object")
        return cls(**data)

    @classmethod
    def tiny(cls) -> ModelConfig:
        """Small configuration for CPU smoke tests; behavior is unchanged."""

        return cls(
            image_height=32,
            image_width=48,
            patch_size=8,
            grid_height=4,
            grid_width=6,
            video_hz=2,
            slow_hz=1,
            fast_hz=4,
            slow_horizon=2,
            fast_horizon=4,
            vision_dim=32,
            model_dim=64,
            controller_dim=64,
            vision_layers=2,
            fusion_layers=1,
            decoder_layers=1,
            num_heads=4,
            window_height=2,
            window_width=3,
            dropout=0.0,
            gradient_checkpointing=False,
            frame_summary_tokens=2,
            recent_entries=4,
            intermediate_entries=2,
            long_entries=2,
            recent_tokens_per_entry=2,
            intermediate_tokens_per_entry=1,
            long_tokens_per_entry=1,
            frames_per_intermediate=2,
            intermediate_per_long=2,
            control_query_tokens=4,
            action_dim=16,
            trajectory_summary_tokens=2,
            num_buttons=3,
            compressor_layers=1,
        )
