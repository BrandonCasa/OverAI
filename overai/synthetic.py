"""Create a deterministic toy demonstration dataset for end-to-end testing."""

from __future__ import annotations

import argparse
import json
import math
from pathlib import Path

import torch
from torchvision.io import write_jpeg

from .config import ModelConfig


def create_synthetic_dataset(
    output_dir: Path,
    cfg: ModelConfig,
    seconds: float,
) -> Path:
    fast_count = round(seconds * cfg.fast_hz)
    if fast_count < cfg.fast_hz + cfg.fast_horizon:
        raise ValueError(
            "synthetic episode must be at least one second plus the horizon"
        )
    frame_count = math.ceil(fast_count / cfg.fast_ticks_per_video)
    slow_count = math.ceil(fast_count / cfg.fast_ticks_per_slow)
    episode_dir = output_dir / "episodes" / "synthetic-001"
    frame_dir = episode_dir / "frames"
    frame_dir.mkdir(parents=True, exist_ok=True)

    y = torch.linspace(0, 1, cfg.image_height).view(1, cfg.image_height, 1)
    x = torch.linspace(0, 1, cfg.image_width).view(1, 1, cfg.image_width)
    for index in range(frame_count):
        phase = index / max(frame_count - 1, 1)
        frame = torch.cat(
            (
                (x.expand(1, cfg.image_height, -1) + phase).fmod(1.0),
                (y.expand(1, -1, cfg.image_width) + phase * 0.5).fmod(1.0),
                torch.full((1, cfg.image_height, cfg.image_width), phase),
            ),
            dim=0,
        )
        write_jpeg((frame * 255).to(torch.uint8), str(frame_dir / f"{index:06d}.jpg"))

    fast_timestamps = torch.arange(fast_count, dtype=torch.float32) / cfg.fast_hz
    frame_timestamps = torch.arange(frame_count, dtype=torch.float32) / cfg.video_hz
    slow_timestamps = torch.arange(slow_count, dtype=torch.float32) / cfg.slow_hz
    axes = torch.stack(
        (
            torch.sin(fast_timestamps * 1.7),
            torch.cos(fast_timestamps * 0.9),
        ),
        dim=-1,
    )
    movement_x = torch.where(
        torch.sin(slow_timestamps) > 0.25,
        2,
        torch.where(torch.sin(slow_timestamps) < -0.25, 0, 1),
    ).long()
    movement_y = torch.where(torch.cos(slow_timestamps * 0.7) > 0, 2, 1).long()
    movement = torch.stack((movement_x, movement_y), dim=-1)
    buttons = torch.stack(
        [
            ((torch.arange(slow_count) + button) % (button + 2) == 0)
            for button in range(cfg.num_buttons)
        ],
        dim=-1,
    ).to(torch.uint8)
    controls = {
        "fast_timestamps": fast_timestamps,
        "frame_timestamps": frame_timestamps,
        "slow_timestamps": slow_timestamps,
        "health": (1.0 - slow_timestamps[:, None] / max(seconds, 1.0)).clamp(-1, 1),
        "damage_events": ((torch.arange(slow_count) % (cfg.slow_hz * 2)) == 0)
        .float()
        .unsqueeze(-1),
        "kill_events": ((torch.arange(slow_count) % (cfg.slow_hz * 3)) == 0)
        .float()
        .unsqueeze(-1),
        "charge": torch.sin(slow_timestamps[:, None] * 0.3),
        "axes": axes,
        "movement": movement,
        "buttons": buttons,
    }
    controls_path = episode_dir / "controls.pt"
    torch.save(controls, controls_path)
    manifest = {
        "version": 2,
        "episodes": [
            {
                "id": "synthetic-001",
                "frames": "episodes/synthetic-001/frames",
                "controls": "episodes/synthetic-001/controls.pt",
            }
        ],
    }
    manifest_path = output_dir / "manifest.json"
    manifest_path.write_text(json.dumps(manifest, indent=2) + "\n", encoding="utf-8")
    cfg.to_json(output_dir / "model_config.json")
    return manifest_path


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, default=Path("data/synthetic"))
    parser.add_argument("--seconds", type=float, default=4.0)
    parser.add_argument("--model-config", type=Path)
    args = parser.parse_args()
    cfg = (
        ModelConfig.from_json(args.model_config)
        if args.model_config
        else ModelConfig.tiny()
    )
    print(create_synthetic_dataset(args.output, cfg, args.seconds))


if __name__ == "__main__":
    main()
