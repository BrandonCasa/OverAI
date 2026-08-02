"""Benchmark the three scheduled inference paths against their deadlines."""

from __future__ import annotations

import argparse
import math
from collections.abc import Callable
from pathlib import Path

import torch

from .config import ModelConfig
from .model import HierarchicalImitationController
from .runtime import load_controller_checkpoint
from .types import ExecutedActions, TimingContext


def _elapsed_ms(call: Callable[[], None], iterations: int) -> float:
    start = torch.cuda.Event(enable_timing=True)
    end = torch.cuda.Event(enable_timing=True)
    start.record()
    for _ in range(iterations):
        call()
    end.record()
    end.synchronize()
    return start.elapsed_time(end) / iterations


def _scheduled_paths(cfg: ModelConfig) -> set[str]:
    period = math.lcm(cfg.fast_ticks_per_video, cfg.fast_ticks_per_slow)
    paths: set[str] = set()
    for tick in range(period):
        video_due = tick % cfg.fast_ticks_per_video == 0
        slow_due = tick % cfg.fast_ticks_per_slow == 0
        paths.add(
            ("video" if video_due else "fast")
            + ("+slow" if slow_due else "")
        )
    return paths


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--checkpoint", type=Path)
    parser.add_argument("--model-config", type=Path)
    parser.add_argument("--warmup", type=int, default=3)
    parser.add_argument("--iterations", type=int, default=20)
    args = parser.parse_args()
    if not torch.cuda.is_available():
        raise SystemExit("CUDA is required for the inference benchmark")
    if args.checkpoint and args.model_config:
        raise SystemExit("--checkpoint and --model-config are mutually exclusive")

    device = torch.device("cuda")
    if args.checkpoint:
        model = load_controller_checkpoint(args.checkpoint, device)
    else:
        cfg = (
            ModelConfig.from_json(args.model_config)
            if args.model_config
            else ModelConfig()
        )
        model = HierarchicalImitationController(cfg).to(device).eval()
    cfg = model.cfg
    state = model.initial_state(1, device)
    frame = torch.zeros(
        1,
        cfg.input_channels,
        cfg.image_height,
        cfg.image_width,
        dtype=torch.uint8,
        device=device,
    )
    zero = lambda width: torch.zeros(1, width, device=device)
    actions = ExecutedActions(
        movement=torch.ones(1, 2, dtype=torch.long, device=device),
        buttons=zero(cfg.num_buttons),
        axes=zero(2),
    )
    timing = TimingContext(zero(1), zero(1), zero(1), zero(1))

    def video(slow: bool = False) -> None:
        nonlocal state
        state = model.on_video_frame(
            frame, actions, timing, state, run_slow_decoder=slow
        ).state

    def fast() -> None:
        nonlocal state
        _, state = model.fast_tick_between_frames(actions, timing, state)

    def slow() -> None:
        nonlocal state
        _, state = model.slow_tick(state)

    def fast_slow() -> None:
        fast()
        slow()

    with torch.inference_mode(), torch.autocast("cuda", dtype=torch.bfloat16):
        for _ in range(args.warmup):
            video(True)
            fast()
        torch.cuda.synchronize()
        video_ms = _elapsed_ms(lambda: video(False), args.iterations)
        fast_ms = _elapsed_ms(fast, args.iterations)
        slow_ms = _elapsed_ms(slow, args.iterations)
        combined_ms = _elapsed_ms(lambda: video(True), args.iterations)
        fast_slow_ms = _elapsed_ms(fast_slow, args.iterations)

    fast_budget = 1000.0 / cfg.fast_hz
    path_latencies = {
        "video": video_ms,
        "fast": fast_ms,
        "video+slow": combined_ms,
        "fast+slow": fast_slow_ms,
    }
    scheduled_paths = _scheduled_paths(cfg)
    print(f"GPU: {torch.cuda.get_device_name(0)}")
    print(f"video+memory+fast: {video_ms:.2f} ms / {fast_budget:.2f} ms")
    print(f"between-frame fast: {fast_ms:.2f} ms / {fast_budget:.2f} ms")
    print(f"slow decoder alone: {slow_ms:.2f} ms")
    print(f"video+fast+slow boundary: {combined_ms:.2f} ms / {fast_budget:.2f} ms")
    print(f"between-frame fast+slow boundary: {fast_slow_ms:.2f} ms / {fast_budget:.2f} ms")
    realtime = all(
        path_latencies[path] <= fast_budget for path in scheduled_paths
    )
    print(f"scheduled paths: {', '.join(sorted(scheduled_paths))}")
    print(f"core realtime deadlines: {'PASS' if realtime else 'FAIL'}")


if __name__ == "__main__":
    main()
