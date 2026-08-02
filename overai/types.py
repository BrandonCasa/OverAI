"""Typed inputs, predictions, and recurrent state containers."""

from __future__ import annotations

from dataclasses import dataclass, fields, is_dataclass, replace
from typing import TypeVar

import torch


@dataclass(slots=True)
class ExecutedActions:
    movement: torch.Tensor
    buttons: torch.Tensor
    axes: torch.Tensor


@dataclass(slots=True)
class ObservationContext:
    health: torch.Tensor
    damage_event: torch.Tensor
    kill_event: torch.Tensor
    charge: torch.Tensor


@dataclass(slots=True)
class TimingContext:
    absolute_time: torch.Tensor
    since_video_frame: torch.Tensor
    since_slow_update: torch.Tensor
    fast_delta_time: torch.Tensor


@dataclass(slots=True)
class HierarchicalMemoryState:
    recent: torch.Tensor
    intermediate: torch.Tensor
    long: torch.Tensor
    recent_valid: torch.Tensor
    intermediate_valid: torch.Tensor
    long_valid: torch.Tensor
    frame_counter: int
    intermediate_counter: int


@dataclass(slots=True)
class FastControllerState:
    hidden: torch.Tensor
    previous_trajectory: torch.Tensor


@dataclass(slots=True)
class ControllerState:
    memory: HierarchicalMemoryState
    fast: FastControllerState
    current_grid: torch.Tensor | None
    shared_tokens: torch.Tensor | None
    previous_axis_trajectory: torch.Tensor
    previous_slow_trajectory: torch.Tensor


@dataclass(slots=True)
class SlowPrediction:
    immediate_movement_logits: torch.Tensor
    immediate_button_logits: torch.Tensor
    trajectory_movement_logits: torch.Tensor
    trajectory_button_logits: torch.Tensor


@dataclass(slots=True)
class FastPrediction:
    immediate_axes: torch.Tensor
    axis_trajectory: torch.Tensor
    next_state: FastControllerState


@dataclass(slots=True)
class ReplanOutput:
    slow: SlowPrediction | None
    fast: FastPrediction
    state: ControllerState


@dataclass(slots=True)
class DecodedSlowAction:
    movement: torch.Tensor
    buttons: torch.Tensor


@dataclass(slots=True)
class RuntimeStepOutput:
    axes: torch.Tensor
    discrete: DecodedSlowAction | None
    slow_prediction: SlowPrediction | None
    fast_prediction: FastPrediction
    state: ControllerState


@dataclass(slots=True)
class SlowTargets:
    movement: torch.Tensor
    buttons: torch.Tensor


@dataclass(slots=True)
class FastTargets:
    axes: torch.Tensor


StateT = TypeVar("StateT")


def detach_state(value: StateT) -> StateT:
    """Recursively detach tensors in a dataclass state tree."""

    if isinstance(value, torch.Tensor):
        return value.detach()  # type: ignore[return-value]
    if is_dataclass(value) and not isinstance(value, type):
        updates = {
            field.name: detach_state(getattr(value, field.name))
            for field in fields(value)
        }
        return replace(value, **updates)  # type: ignore[arg-type, return-value]
    return value
