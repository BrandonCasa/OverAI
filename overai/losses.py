"""Supervised imitation losses for discrete and continuous control."""

from __future__ import annotations

from dataclasses import dataclass
from functools import lru_cache

import torch
from torch.nn import functional as F

from .config import ModelConfig
from .types import FastPrediction, FastTargets, SlowPrediction, SlowTargets


@dataclass(frozen=True, slots=True)
class LossWeights:
    slow_movement: float = 1.0
    slow_buttons: float = 1.0
    fast_axis1: float = 1.0
    fast_axis2: float = 1.0
    velocity: float = 0.10
    acceleration: float = 0.05
    immediate_supervision: float = 1.0
    immediate_consistency: float = 0.25


@lru_cache(maxsize=32)
def horizon_weights(
    horizon: int,
    device: torch.device,
    dtype: torch.dtype = torch.float32,
    minimum_weight: float = 0.15,
    decay: float = 3.0,
) -> torch.Tensor:
    """Return shared, read-only horizon weights for a training configuration."""

    normalized_time = torch.linspace(0.0, 1.0, horizon, device=device, dtype=dtype)
    weights = minimum_weight + (1.0 - minimum_weight) * torch.exp(
        -decay * normalized_time
    )
    return weights / weights.mean()


def weighted_mean(loss: torch.Tensor, weights: torch.Tensor) -> torch.Tensor:
    shape = [1, weights.shape[0], *([1] * (loss.ndim - 2))]
    return (loss * weights.view(*shape)).mean()


def slow_control_loss(
    prediction: SlowPrediction,
    targets: SlowTargets,
    cfg: ModelConfig,
) -> dict[str, torch.Tensor]:
    if targets.movement.shape[1:] != (cfg.slow_horizon, 2):
        raise ValueError("slow target horizon does not match model config")
    weights = horizon_weights(
        cfg.slow_horizon,
        prediction.trajectory_movement_logits.device,
        prediction.trajectory_movement_logits.dtype,
    )
    movement_loss = weighted_mean(
        F.cross_entropy(
            prediction.trajectory_movement_logits.permute(0, 3, 1, 2),
            targets.movement.long(),
            reduction="none",
        ).sum(dim=-1),
        weights,
    )
    button_loss = weighted_mean(
        F.binary_cross_entropy_with_logits(
            prediction.trajectory_button_logits,
            targets.buttons.float(),
            reduction="none",
        ),
        weights,
    )

    immediate_supervision = (
        2.0
        * F.cross_entropy(
            prediction.immediate_movement_logits.transpose(1, 2),
            targets.movement[:, 0].long(),
        )
        + F.binary_cross_entropy_with_logits(
            prediction.immediate_button_logits, targets.buttons[:, 0].float()
        )
    )
    immediate_consistency = (
        F.kl_div(
            F.log_softmax(prediction.immediate_movement_logits, dim=-1),
            F.softmax(prediction.trajectory_movement_logits[:, 0].detach(), dim=-1),
            reduction="batchmean",
        )
        + F.mse_loss(
            prediction.immediate_button_logits,
            prediction.trajectory_button_logits[:, 0].detach(),
        )
    )
    return {
        "slow_movement": movement_loss,
        "slow_buttons": button_loss,
        "slow_immediate_supervision": immediate_supervision,
        "slow_immediate_consistency": immediate_consistency,
    }


def temporal_derivative(values: torch.Tensor) -> torch.Tensor:
    return values[:, 1:] - values[:, :-1]


def fast_axis_loss(
    prediction: FastPrediction,
    targets: FastTargets,
    cfg: ModelConfig,
) -> dict[str, torch.Tensor]:
    if targets.axes.shape[1] != cfg.fast_horizon:
        raise ValueError("fast target horizon does not match model config")
    weights = horizon_weights(
        cfg.fast_horizon,
        prediction.axis_trajectory.device,
        prediction.axis_trajectory.dtype,
    )
    axis1_loss = weighted_mean(
        F.huber_loss(
            prediction.axis_trajectory[..., 0],
            targets.axes[..., 0],
            reduction="none",
        ),
        weights,
    )
    axis2_loss = weighted_mean(
        F.huber_loss(
            prediction.axis_trajectory[..., 1],
            targets.axes[..., 1],
            reduction="none",
        ),
        weights,
    )
    predicted_velocity = temporal_derivative(prediction.axis_trajectory)
    target_velocity = temporal_derivative(targets.axes)
    velocity_loss = F.huber_loss(predicted_velocity, target_velocity)
    predicted_acceleration = temporal_derivative(predicted_velocity)
    target_acceleration = temporal_derivative(target_velocity)
    acceleration_loss = F.huber_loss(predicted_acceleration, target_acceleration)
    immediate_supervision = F.huber_loss(prediction.immediate_axes, targets.axes[:, 0])
    immediate_consistency = F.mse_loss(
        prediction.immediate_axes, prediction.axis_trajectory[:, 0].detach()
    )
    return {
        "fast_axis1": axis1_loss,
        "fast_axis2": axis2_loss,
        "fast_velocity": velocity_loss,
        "fast_acceleration": acceleration_loss,
        "fast_immediate_supervision": immediate_supervision,
        "fast_immediate_consistency": immediate_consistency,
    }


def total_imitation_loss(
    slow_prediction: SlowPrediction,
    fast_prediction: FastPrediction,
    slow_targets: SlowTargets,
    fast_targets: FastTargets,
    cfg: ModelConfig,
    weights: LossWeights | None = None,
) -> tuple[torch.Tensor, dict[str, torch.Tensor]]:
    weights = weights or LossWeights()
    slow_losses = slow_control_loss(slow_prediction, slow_targets, cfg)
    fast_losses = fast_axis_loss(fast_prediction, fast_targets, cfg)
    total = (
        weights.slow_movement * slow_losses["slow_movement"]
        + weights.slow_buttons * slow_losses["slow_buttons"]
        + weights.fast_axis1 * fast_losses["fast_axis1"]
        + weights.fast_axis2 * fast_losses["fast_axis2"]
        + weights.velocity * fast_losses["fast_velocity"]
        + weights.acceleration * fast_losses["fast_acceleration"]
        + weights.immediate_supervision
        * (
            slow_losses["slow_immediate_supervision"]
            + fast_losses["fast_immediate_supervision"]
        )
        + weights.immediate_consistency
        * (
            slow_losses["slow_immediate_consistency"]
            + fast_losses["fast_immediate_consistency"]
        )
    )
    return total, {**slow_losses, **fast_losses}


def weighted_fast_total(
    losses: dict[str, torch.Tensor], weights: LossWeights
) -> torch.Tensor:
    return (
        weights.fast_axis1 * losses["fast_axis1"]
        + weights.fast_axis2 * losses["fast_axis2"]
        + weights.velocity * losses["fast_velocity"]
        + weights.acceleration * losses["fast_acceleration"]
        + weights.immediate_supervision * losses["fast_immediate_supervision"]
        + weights.immediate_consistency * losses["fast_immediate_consistency"]
    )


def weighted_slow_total(
    losses: dict[str, torch.Tensor], weights: LossWeights
) -> torch.Tensor:
    return (
        weights.slow_movement * losses["slow_movement"]
        + weights.slow_buttons * losses["slow_buttons"]
        + weights.immediate_supervision * losses["slow_immediate_supervision"]
        + weights.immediate_consistency * losses["slow_immediate_consistency"]
    )
