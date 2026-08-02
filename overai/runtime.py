"""Real-time adapter interface and fixed-rate inference loop."""

from __future__ import annotations

import time
from abc import ABC, abstractmethod
from pathlib import Path

import torch

from .config import ModelConfig
from .model import HierarchicalImitationController, RuntimeController
from .types import DecodedSlowAction, ExecutedActions, ObservationContext, TimingContext


class GameAdapter(ABC):
    """Boundary between the game-specific capture/input code and the model.

    Returned tensors are single samples without a batch dimension.  A concrete
    adapter can use OBS, a game API, Windows input APIs, or an emulator bridge.
    """

    @abstractmethod
    def capture_frame(self) -> torch.Tensor:
        """Return an RB uint8 CHW frame at the configured resolution."""

    @abstractmethod
    def observation(self) -> ObservationContext:
        """Return 5 Hz health, damage, kill, and charge tensors of shape [1]."""

    @abstractmethod
    def executed_actions(self) -> ExecutedActions:
        """Return the controls actually applied during the previous tick."""

    @abstractmethod
    def apply_axes(self, axes: torch.Tensor) -> None:
        """Apply a CPU tensor of shape [2] with values in [-1, 1]."""

    @abstractmethod
    def apply_discrete(self, action: DecodedSlowAction) -> None:
        """Apply CPU [x, y] categories and a [num_buttons] button tensor."""

    def close(self) -> None:
        """Release adapter resources."""


def _batch_context(
    context: ObservationContext, device: torch.device
) -> ObservationContext:
    return ObservationContext(
        health=context.health.reshape(1, 1).to(device),
        damage_event=context.damage_event.reshape(1, 1).to(device),
        kill_event=context.kill_event.reshape(1, 1).to(device),
        charge=context.charge.reshape(1, 1).to(device),
    )


def _batch_actions(actions: ExecutedActions, device: torch.device) -> ExecutedActions:
    return ExecutedActions(
        movement=actions.movement.reshape(1, 2).to(device),
        buttons=actions.buttons.reshape(1, -1).to(device),
        axes=actions.axes.reshape(1, 2).to(device),
    )


def load_controller_checkpoint(
    checkpoint_path: str | Path,
    device: torch.device | str = "cuda",
) -> HierarchicalImitationController:
    payload = torch.load(checkpoint_path, map_location="cpu", weights_only=True)
    if payload.get("format_version") != 3:
        raise ValueError("unsupported checkpoint format")
    cfg = ModelConfig(**payload["model_config"])
    model = HierarchicalImitationController(cfg)
    model.load_state_dict(payload["model"])
    return model.to(device).eval()


def run_realtime(
    model: HierarchicalImitationController,
    adapter: GameAdapter,
    duration_seconds: float | None = None,
    device: torch.device | str = "cuda",
    use_bf16: bool = True,
) -> None:
    """Run scheduled 60 Hz control until duration expires or interrupted."""

    device = torch.device(device)
    if device.type == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA runtime is unavailable")
    scheduler = RuntimeController(model)
    state = model.initial_state(1, device)
    cfg = model.cfg
    start = time.perf_counter()
    tick = 0
    last_frame_time = start
    last_slow_time = start
    previous_tick_time = start
    context: ObservationContext | None = None

    try:
        while (
            duration_seconds is None or time.perf_counter() - start < duration_seconds
        ):
            deadline = start + tick / cfg.fast_hz
            remaining = deadline - time.perf_counter()
            if remaining > 0.001:
                time.sleep(remaining - 0.0005)
            while time.perf_counter() < deadline:
                pass

            now = time.perf_counter()
            video_due = tick % cfg.fast_ticks_per_video == 0
            slow_due = tick % cfg.fast_ticks_per_slow == 0
            frame = None
            if video_due:
                frame = (
                    adapter.capture_frame().unsqueeze(0).to(device, non_blocking=True)
                )
                last_frame_time = now
            if slow_due:
                last_slow_time = now
                context = _batch_context(adapter.observation(), device)

            timing = TimingContext(
                absolute_time=torch.tensor([[now - start]], device=device),
                since_video_frame=torch.tensor(
                    [[now - last_frame_time]], device=device
                ),
                since_slow_update=torch.tensor([[now - last_slow_time]], device=device),
                fast_delta_time=torch.tensor(
                    [[now - previous_tick_time]], device=device
                ),
            )
            if context is None:
                raise RuntimeError("observation context was not initialized")
            actions = _batch_actions(adapter.executed_actions(), device)
            autocast_enabled = use_bf16 and device.type == "cuda"
            with (
                torch.inference_mode(),
                torch.autocast(
                    device_type=device.type,
                    dtype=torch.bfloat16,
                    enabled=autocast_enabled,
                ),
            ):
                output = scheduler.step(frame, context, actions, timing, state)
            state = output.state
            adapter.apply_axes(output.axes[0].float().cpu())
            if output.discrete is not None:
                adapter.apply_discrete(
                    DecodedSlowAction(
                        movement=output.discrete.movement[0].cpu(),
                        buttons=output.discrete.buttons[0].cpu(),
                    )
                )
            previous_tick_time = now
            tick += 1
    finally:
        adapter.close()
