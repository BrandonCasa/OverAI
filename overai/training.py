"""CUDA streaming supervised-imitation training entry point."""

from __future__ import annotations

import argparse
import json
import math
import random
import time
from collections.abc import Iterable
from contextlib import nullcontext
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

import torch
from torch.nn.utils import clip_grad_norm_
from torch.optim import AdamW
from torch.utils.data import DataLoader

from .config import ModelConfig
from .data import DemonstrationWindowDataset, SequenceBatch, dataset_summary
from .losses import (
    LossWeights,
    fast_axis_loss,
    slow_control_loss,
    weighted_fast_total,
    weighted_slow_total,
)
from .model import HierarchicalImitationController
from .types import FastTargets, SlowTargets, detach_state


@dataclass(frozen=True, slots=True)
class TrainingConfig:
    epochs: int = 10
    batch_size: int = 1
    learning_rate: float = 3.0e-4
    weight_decay: float = 0.05
    history_seconds: float = 30.0
    optimization_seconds: float = 2.0
    stride_seconds: float = 2.0
    tbptt_seconds: float = 0.2
    num_workers: int = 2
    seed: int = 1337
    bf16: bool = True
    compile_vision: bool = False
    max_grad_norm: float = 1.0
    checkpoint_every_steps: int = 250

    def __post_init__(self) -> None:
        positive = (
            "epochs",
            "batch_size",
            "learning_rate",
            "optimization_seconds",
            "stride_seconds",
            "tbptt_seconds",
            "max_grad_norm",
            "checkpoint_every_steps",
        )
        for name in positive:
            if getattr(self, name) <= 0:
                raise ValueError(f"{name} must be positive")
        if self.history_seconds < 0 or self.num_workers < 0 or self.weight_decay < 0:
            raise ValueError(
                "history_seconds, num_workers, and weight_decay cannot be negative"
            )


@dataclass(slots=True)
class BatchResult:
    mean_loss: float
    optimizer_steps: int
    metrics: dict[str, float]


def _read_manifest_payload(path: Path, expected_split: str) -> dict[str, Any]:
    payload: Any = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(payload, dict):
        raise TypeError(f"{expected_split} manifest must be an object")
    actual_split = payload.get("split")
    if actual_split != expected_split:
        raise ValueError(
            f"{expected_split} manifest split must be {expected_split}, "
            f"found {actual_split!r}"
        )
    return payload


def _build_training_datasets(
    training_manifest: Path,
    validation_manifest: Path,
    model_cfg: ModelConfig,
    training_cfg: TrainingConfig,
) -> tuple[
    DemonstrationWindowDataset,
    DemonstrationWindowDataset,
    dict[str, Any],
    dict[str, Any],
]:
    if training_manifest.resolve() == validation_manifest.resolve():
        raise ValueError("training and validation manifests must be different files")

    training_payload = _read_manifest_payload(training_manifest, "train")
    validation_payload = _read_manifest_payload(validation_manifest, "validation")
    training_dataset = DemonstrationWindowDataset(
        training_manifest,
        model_cfg,
        history_seconds=training_cfg.history_seconds,
        optimization_seconds=training_cfg.optimization_seconds,
        stride_seconds=training_cfg.stride_seconds,
    )
    validation_dataset = DemonstrationWindowDataset(
        validation_manifest,
        model_cfg,
        history_seconds=training_cfg.history_seconds,
        optimization_seconds=training_cfg.optimization_seconds,
        stride_seconds=training_cfg.stride_seconds,
    )

    for key, label in (
        ("axis_normalization", "axis normalization"),
        ("control_profile_sha256", "control profile"),
        ("telemetry", "telemetry configuration"),
        ("num_buttons", "button count"),
    ):
        if training_payload.get(key) != validation_payload.get(key):
            raise ValueError(
                f"training and validation manifests have different {label}"
            )

    training_ids = {record.episode_id for record in training_dataset.records}
    validation_ids = {record.episode_id for record in validation_dataset.records}
    overlapping_ids = sorted(training_ids & validation_ids)
    if overlapping_ids:
        raise ValueError(
            "training and validation manifests share episode ids: "
            + ", ".join(overlapping_ids)
        )

    training_files = {
        path.resolve()
        for record in training_dataset.records
        for pair in record.frame_pairs
        for path in (*pair, record.controls_path)
    }
    validation_files = {
        path.resolve()
        for record in validation_dataset.records
        for pair in record.frame_pairs
        for path in (*pair, record.controls_path)
    }
    if training_files & validation_files:
        raise ValueError("training and validation manifests share episode files")

    return (
        training_dataset,
        validation_dataset,
        training_payload,
        validation_payload,
    )


def configure_runtime(device: torch.device, seed: int) -> None:
    random.seed(seed)
    torch.manual_seed(seed)
    if device.type == "cuda":
        torch.cuda.manual_seed_all(seed)
        torch.set_float32_matmul_precision("high")
        torch.backends.cuda.matmul.allow_tf32 = True
        torch.backends.cudnn.allow_tf32 = True
        torch.backends.cudnn.benchmark = True
        torch.backends.cuda.enable_flash_sdp(True)


def parameter_count(model: torch.nn.Module) -> int:
    return sum(parameter.numel() for parameter in model.parameters())


def _compile_vision_for_training(
    model: HierarchicalImitationController,
) -> None:
    compile_vision = getattr(model.vision, "compile", None)
    if not callable(compile_vision):
        raise TypeError("this PyTorch build does not provide nn.Module.compile")

    # A TBPTT chunk invokes vision several times before its shared backward pass.
    # Those compiled outputs must remain live for autograd, so they cannot safely
    # be treated as separate CUDA Graph iterations. Keep Inductor compilation,
    # but disable CUDA Graph capture for this training-specific call pattern.
    compile_vision(
        fullgraph=False,
        options={"triton.cudagraphs": False},
    )


def _autocast_context(device: torch.device, enabled: bool):
    if enabled:
        return torch.autocast(device_type=device.type, dtype=torch.bfloat16)
    return nullcontext()


def _run_tick(
    model: HierarchicalImitationController,
    batch: SequenceBatch,
    tick: int,
    state,
    device: torch.device,
):
    context = batch.observation_context(tick, device)
    actions = batch.executed_actions(tick, device)
    timing = batch.timing_context(tick, device)
    slow_due = tick % model.cfg.fast_ticks_per_slow == 0
    if tick % model.cfg.fast_ticks_per_video == 0:
        frame_index = tick // model.cfg.fast_ticks_per_video
        output = model.on_video_frame(
            batch.load_frame(frame_index, device),
            context,
            actions,
            timing,
            state,
            run_slow_decoder=slow_due,
        )
        return output.fast, output.slow, output.state
    fast_prediction, state = model.fast_tick_between_frames(
        context, actions, timing, state
    )
    slow_prediction = None
    if slow_due:
        slow_prediction, state = model.slow_tick(state)
    return fast_prediction, slow_prediction, state


def train_sequence_batch(
    model: HierarchicalImitationController,
    batch: SequenceBatch,
    optimizer: torch.optim.Optimizer,
    device: torch.device,
    tbptt_ticks: int,
    use_bf16: bool,
    max_grad_norm: float = 1.0,
    loss_weights: LossWeights | None = None,
) -> BatchResult:
    """Warm causal history, then optimize using bounded streaming BPTT chunks."""

    cfg = model.cfg
    weights = loss_weights or LossWeights()
    state = model.initial_state(batch.batch_size, device)
    optimizer.zero_grad(set_to_none=True)

    pending_loss: torch.Tensor | None = None
    pending_predictions = 0
    optimizer_steps = 0
    reported_total = 0.0
    reported_count = 0
    metric_totals: dict[str, float] = {}
    metric_counts: dict[str, int] = {}

    # History constructs all three memory tiers but is outside the gradient tape.
    model.train()
    for tick in range(batch.history_ticks):
        with torch.no_grad(), _autocast_context(device, use_bf16):
            _, _, state = _run_tick(model, batch, tick, state, device)
    state = detach_state(state)

    for tick in range(batch.history_ticks, batch.process_ticks):
        with _autocast_context(device, use_bf16):
            fast_prediction, slow_prediction, state = _run_tick(
                model, batch, tick, state, device
            )
            fast_targets = FastTargets(
                axes=batch.axes[:, tick : tick + cfg.fast_horizon].to(
                    device, non_blocking=True
                )
            )
            fast_losses = fast_axis_loss(fast_prediction, fast_targets, cfg)
            tick_loss = weighted_fast_total(fast_losses, weights)
            prediction_terms = 1
            all_losses = dict(fast_losses)

            if slow_prediction is not None:
                slow_index = tick // cfg.fast_ticks_per_slow
                slow_targets = SlowTargets(
                    movement=batch.movement[
                        :, slow_index : slow_index + cfg.slow_horizon
                    ].to(device, non_blocking=True),
                    buttons=batch.buttons[
                        :, slow_index : slow_index + cfg.slow_horizon
                    ].to(device, non_blocking=True),
                )
                slow_losses = slow_control_loss(slow_prediction, slow_targets, cfg)
                tick_loss = tick_loss + weighted_slow_total(slow_losses, weights)
                prediction_terms += 1
                all_losses.update(slow_losses)

        pending_loss = tick_loss if pending_loss is None else pending_loss + tick_loss
        pending_predictions += prediction_terms
        reported_total += float(tick_loss.detach())
        reported_count += prediction_terms
        for name, value in all_losses.items():
            metric_totals[name] = metric_totals.get(name, 0.0) + float(value.detach())
            metric_counts[name] = metric_counts.get(name, 0) + 1

        ticks_in_optimization = tick - batch.history_ticks + 1
        boundary = ticks_in_optimization % tbptt_ticks == 0
        final_tick = tick + 1 == batch.process_ticks
        if boundary or final_tick:
            if pending_loss is None or pending_predictions == 0:
                raise RuntimeError("TBPTT chunk contains no predictions")
            normalized_loss = pending_loss / pending_predictions
            if not torch.isfinite(normalized_loss):
                raise FloatingPointError(f"non-finite training loss: {normalized_loss}")
            normalized_loss.backward()
            clip_grad_norm_(model.parameters(), max_grad_norm)
            optimizer.step()
            optimizer.zero_grad(set_to_none=True)
            optimizer_steps += 1
            state = detach_state(state)
            pending_loss = None
            pending_predictions = 0

    return BatchResult(
        mean_loss=reported_total / max(reported_count, 1),
        optimizer_steps=optimizer_steps,
        metrics={
            name: value / metric_counts[name] for name, value in metric_totals.items()
        },
    )


def evaluate_model(
    model: HierarchicalImitationController,
    batches: Iterable[SequenceBatch],
    device: torch.device,
    use_bf16: bool,
    loss_weights: LossWeights | None = None,
) -> dict[str, float]:
    """Evaluate causal windows without gradients or optimizer updates."""

    cfg = model.cfg
    weights = loss_weights or LossWeights()
    total_loss = 0.0
    total_prediction_terms = 0
    metric_totals: dict[str, float] = {}
    metric_counts: dict[str, int] = {}
    movement_confusion = torch.zeros(2, 3, 3, dtype=torch.int64)
    button_true_positives = 0
    button_false_positives = 0
    button_false_negatives = 0
    button_correct = 0
    button_total = 0
    immediate_axis_absolute_error = 0.0
    immediate_axis_values = 0
    windows = 0

    model.eval()
    with torch.inference_mode():
        for batch in batches:
            windows += batch.batch_size
            state = model.initial_state(batch.batch_size, device)
            for tick in range(batch.process_ticks):
                with _autocast_context(device, use_bf16):
                    fast_prediction, slow_prediction, state = _run_tick(
                        model, batch, tick, state, device
                    )
                    if tick < batch.history_ticks:
                        continue

                    fast_targets = FastTargets(
                        axes=batch.axes[:, tick : tick + cfg.fast_horizon].to(
                            device, non_blocking=True
                        )
                    )
                    fast_losses = fast_axis_loss(fast_prediction, fast_targets, cfg)
                    tick_loss = weighted_fast_total(fast_losses, weights)
                    prediction_terms = 1
                    all_losses = dict(fast_losses)

                    immediate_axis_absolute_error += float(
                        (fast_prediction.immediate_axes - fast_targets.axes[:, 0])
                        .abs()
                        .sum()
                    )
                    immediate_axis_values += fast_targets.axes[:, 0].numel()

                    if slow_prediction is not None:
                        slow_index = tick // cfg.fast_ticks_per_slow
                        slow_targets = SlowTargets(
                            movement=batch.movement[
                                :, slow_index : slow_index + cfg.slow_horizon
                            ].to(device, non_blocking=True),
                            buttons=batch.buttons[
                                :, slow_index : slow_index + cfg.slow_horizon
                            ].to(device, non_blocking=True),
                        )
                        slow_losses = slow_control_loss(
                            slow_prediction, slow_targets, cfg
                        )
                        tick_loss = tick_loss + weighted_slow_total(
                            slow_losses, weights
                        )
                        prediction_terms += 1
                        all_losses.update(slow_losses)

                        movement_prediction = (
                            slow_prediction.immediate_movement_logits.argmax(dim=-1)
                        )
                        movement_target = slow_targets.movement[:, 0].long()
                        for axis in range(2):
                            encoded = (
                                movement_target[:, axis] * 3
                                + movement_prediction[:, axis]
                            )
                            movement_confusion[axis] += torch.bincount(
                                encoded.cpu(), minlength=9
                            ).reshape(3, 3)

                        button_prediction = slow_prediction.immediate_button_logits >= 0
                        button_target = slow_targets.buttons[:, 0].bool()
                        button_true_positives += int(
                            (button_prediction & button_target).sum()
                        )
                        button_false_positives += int(
                            (button_prediction & ~button_target).sum()
                        )
                        button_false_negatives += int(
                            (~button_prediction & button_target).sum()
                        )
                        button_correct += int(
                            (button_prediction == button_target).sum()
                        )
                        button_total += button_target.numel()

                batch_size = batch.batch_size
                total_loss += float(tick_loss) * batch_size
                total_prediction_terms += prediction_terms * batch_size
                for name, value in all_losses.items():
                    metric_totals[name] = metric_totals.get(name, 0.0) + (
                        float(value) * batch_size
                    )
                    metric_counts[name] = metric_counts.get(name, 0) + batch_size

    if windows == 0 or total_prediction_terms == 0:
        raise ValueError("validation loader produced no usable windows")

    movement_correct = int(movement_confusion.diagonal(dim1=1, dim2=2).sum())
    movement_total = int(movement_confusion.sum())
    movement_f1_values: list[float] = []
    for axis in range(2):
        for class_index in range(3):
            true_positive = int(movement_confusion[axis, class_index, class_index])
            false_positive = (
                int(movement_confusion[axis, :, class_index].sum()) - true_positive
            )
            false_negative = (
                int(movement_confusion[axis, class_index, :].sum()) - true_positive
            )
            denominator = 2 * true_positive + false_positive + false_negative
            if denominator:
                movement_f1_values.append(2 * true_positive / denominator)

    button_f1_denominator = (
        2 * button_true_positives + button_false_positives + button_false_negatives
    )
    metrics = {
        "loss": total_loss / total_prediction_terms,
        "movement_accuracy": movement_correct / max(movement_total, 1),
        "movement_macro_f1": sum(movement_f1_values) / max(len(movement_f1_values), 1),
        "button_accuracy": button_correct / max(button_total, 1),
        "button_f1": (
            2 * button_true_positives / button_f1_denominator
            if button_f1_denominator
            else 0.0
        ),
        "immediate_axis_mae": immediate_axis_absolute_error
        / max(immediate_axis_values, 1),
        "windows": float(windows),
    }
    metrics.update(
        {name: value / metric_counts[name] for name, value in metric_totals.items()}
    )
    if not all(math.isfinite(value) for value in metrics.values()):
        raise FloatingPointError("validation produced non-finite metrics")
    return metrics


def save_checkpoint(
    path: Path,
    model: HierarchicalImitationController,
    optimizer: torch.optim.Optimizer,
    epoch: int,
    global_step: int,
    training_cfg: TrainingConfig,
    axis_normalization: dict[str, Any],
    control_profile_sha256: str,
    telemetry: dict[str, Any],
    epoch_complete: bool,
    batches_completed_in_epoch: int,
    data_loader_generator_state: torch.Tensor,
    best_validation_loss: float | None = None,
    validation_metrics: dict[str, float] | None = None,
) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    torch.save(
        {
            "format_version": 3,
            "model_config": model.cfg.to_dict(),
            "training_config": asdict(training_cfg),
            "axis_normalization": axis_normalization,
            "control_profile_sha256": control_profile_sha256,
            "telemetry": telemetry,
            "model": model.state_dict(),
            "optimizer": optimizer.state_dict(),
            "epoch": epoch,
            "epoch_complete": epoch_complete,
            "batches_completed_in_epoch": batches_completed_in_epoch,
            "data_loader_generator_state": data_loader_generator_state,
            "torch_rng_state": torch.get_rng_state(),
            "cuda_rng_state_all": torch.cuda.get_rng_state_all(),
            "global_step": global_step,
            "best_validation_loss": best_validation_loss,
            "validation_metrics": validation_metrics,
        },
        temporary,
    )
    temporary.replace(path)


def load_checkpoint(
    path: Path,
    model: HierarchicalImitationController,
    optimizer: torch.optim.Optimizer,
    *,
    axis_normalization: dict[str, Any],
    control_profile_sha256: str,
    telemetry: dict[str, Any],
) -> tuple[
    int,
    int,
    int | None,
    torch.Tensor | None,
    float,
    dict[str, float] | None,
]:
    checkpoint_data: dict[str, Any] = torch.load(
        path, map_location="cpu", weights_only=True
    )
    if checkpoint_data.get("format_version") != 3:
        raise ValueError("unsupported checkpoint format")
    if checkpoint_data.get("model_config") != model.cfg.to_dict():
        raise ValueError("checkpoint model configuration does not match")
    if checkpoint_data.get("axis_normalization") != axis_normalization:
        raise ValueError("checkpoint axis normalization does not match dataset")
    if checkpoint_data.get("control_profile_sha256") != control_profile_sha256:
        raise ValueError("checkpoint control profile does not match dataset")
    checkpoint_telemetry = checkpoint_data.get(
        "telemetry", {"provider": "zero", "sha256": None}
    )
    if checkpoint_telemetry != telemetry:
        raise ValueError("checkpoint telemetry configuration does not match dataset")
    model.load_state_dict(checkpoint_data["model"])
    optimizer.load_state_dict(checkpoint_data["optimizer"])
    torch_rng_state = checkpoint_data.get("torch_rng_state")
    if isinstance(torch_rng_state, torch.Tensor):
        torch.set_rng_state(torch_rng_state)
    cuda_rng_state_all = checkpoint_data.get("cuda_rng_state_all")
    if isinstance(cuda_rng_state_all, list) and cuda_rng_state_all:
        torch.cuda.set_rng_state_all(cuda_rng_state_all)
    checkpoint_epoch = int(checkpoint_data["epoch"])
    epoch_complete = bool(checkpoint_data.get("epoch_complete", False))
    resume_epoch = checkpoint_epoch + int(epoch_complete)
    batches_completed = (
        0 if epoch_complete else checkpoint_data.get("batches_completed_in_epoch")
    )
    if batches_completed is not None:
        batches_completed = int(batches_completed)
    generator_state = checkpoint_data.get("data_loader_generator_state")
    if generator_state is not None and not isinstance(generator_state, torch.Tensor):
        raise TypeError("checkpoint data-loader generator state is invalid")
    best_validation_loss_value = checkpoint_data.get("best_validation_loss")
    if best_validation_loss_value is None:
        best_validation_loss = math.inf
    elif not isinstance(best_validation_loss_value, (int, float)) or not math.isfinite(
        best_validation_loss_value
    ):
        raise ValueError("checkpoint best validation loss is invalid")
    else:
        best_validation_loss = float(best_validation_loss_value)
    validation_metrics_value = checkpoint_data.get("validation_metrics")
    if validation_metrics_value is None:
        validation_metrics = None
    elif not isinstance(validation_metrics_value, dict) or not all(
        isinstance(name, str)
        and isinstance(value, (int, float))
        and math.isfinite(value)
        for name, value in validation_metrics_value.items()
    ):
        raise ValueError("checkpoint validation metrics are invalid")
    else:
        validation_metrics = {
            name: float(value) for name, value in validation_metrics_value.items()
        }
    return (
        resume_epoch,
        int(checkpoint_data["global_step"]),
        batches_completed,
        generator_state,
        best_validation_loss,
        validation_metrics,
    )


def _legacy_batches_completed_in_epoch(
    global_step: int,
    epoch: int,
    batches_per_epoch: int,
    optimizer_steps_per_batch: int,
) -> int:
    if global_step % optimizer_steps_per_batch:
        raise ValueError(
            "legacy checkpoint global step does not align with a completed batch"
        )
    completed = global_step // optimizer_steps_per_batch - epoch * batches_per_epoch
    if not 0 <= completed <= batches_per_epoch:
        raise ValueError(
            "legacy checkpoint does not contain enough state for an exact resume"
        )
    return completed


def _make_optimizer(
    model: torch.nn.Module, cfg: TrainingConfig, device: torch.device
) -> AdamW:
    kwargs: dict[str, Any] = {
        "lr": cfg.learning_rate,
        "weight_decay": cfg.weight_decay,
        "betas": (0.9, 0.95),
    }
    if device.type == "cuda":
        kwargs["fused"] = True
    return AdamW(model.parameters(), **kwargs)


def run_training(
    manifest: Path,
    validation_manifest: Path,
    output_dir: Path,
    model_cfg: ModelConfig,
    training_cfg: TrainingConfig,
    resume: Path | None = None,
    max_batches: int | None = None,
) -> None:
    if not torch.cuda.is_available():
        raise RuntimeError("training requires an NVIDIA CUDA device")
    device = torch.device("cuda")
    if training_cfg.bf16 and not torch.cuda.is_bf16_supported():
        raise RuntimeError("the selected CUDA device does not support bfloat16")
    configure_runtime(device, training_cfg.seed)

    (
        dataset,
        validation_dataset,
        manifest_payload,
        _,
    ) = _build_training_datasets(
        manifest,
        validation_manifest,
        model_cfg,
        training_cfg,
    )
    axis_normalization = manifest_payload.get("axis_normalization")
    if not isinstance(axis_normalization, dict):
        raise TypeError("training manifest is missing axis_normalization")
    control_profile_sha256 = manifest_payload.get("control_profile_sha256")
    if not isinstance(control_profile_sha256, str) or not control_profile_sha256:
        raise TypeError("training manifest is missing control_profile_sha256")
    # Pre-HUD format-2 manifests unambiguously used the zero provider.
    telemetry = manifest_payload.get("telemetry", {"provider": "zero", "sha256": None})
    if not isinstance(telemetry, dict) or telemetry.get("provider") not in {
        "zero",
        "hud_telemetry",
    }:
        raise TypeError("training manifest is missing telemetry configuration")
    generator = torch.Generator().manual_seed(training_cfg.seed)
    loader = DataLoader(
        dataset,
        batch_size=training_cfg.batch_size,
        shuffle=True,
        num_workers=training_cfg.num_workers,
        collate_fn=dataset.collate,
        persistent_workers=training_cfg.num_workers > 0,
        generator=generator,
    )
    validation_loader = DataLoader(
        validation_dataset,
        batch_size=training_cfg.batch_size,
        shuffle=False,
        num_workers=training_cfg.num_workers,
        collate_fn=validation_dataset.collate,
        persistent_workers=training_cfg.num_workers > 0,
    )
    model = HierarchicalImitationController(model_cfg).to(device)
    optimizer = _make_optimizer(model, training_cfg, device)
    start_epoch = 0
    global_step = 0
    resume_batches: int | None = 0
    best_validation_loss = math.inf
    last_validation_metrics: dict[str, float] | None = None
    if resume is not None:
        (
            start_epoch,
            global_step,
            resume_batches,
            generator_state,
            best_validation_loss,
            last_validation_metrics,
        ) = load_checkpoint(
            resume,
            model,
            optimizer,
            axis_normalization=axis_normalization,
            control_profile_sha256=control_profile_sha256,
            telemetry=telemetry,
        )
        if generator_state is not None:
            generator.set_state(generator_state)
    if training_cfg.compile_vision:
        _compile_vision_for_training(model)

    tbptt_ticks = round(training_cfg.tbptt_seconds * model_cfg.fast_hz)
    if tbptt_ticks <= 0:
        raise ValueError("tbptt_seconds must span at least one fast tick")
    optimizer_steps_per_batch = (
        dataset.optimization_ticks + tbptt_ticks - 1
    ) // tbptt_ticks
    if resume is not None and resume_batches is None:
        resume_batches = _legacy_batches_completed_in_epoch(
            global_step,
            start_epoch,
            len(loader),
            optimizer_steps_per_batch,
        )
    assert resume_batches is not None
    output_dir.mkdir(parents=True, exist_ok=True)
    model_cfg.to_json(output_dir / "model_config.json")
    (output_dir / "training_config.json").write_text(
        json.dumps(asdict(training_cfg), indent=2) + "\n", encoding="utf-8"
    )
    device_properties = torch.cuda.get_device_properties(0)
    print(
        f"device={device_properties.name} "
        f"vram_gib={device_properties.total_memory / 2**30:.1f} "
        f"parameters={parameter_count(model):,} windows={len(dataset)} "
        f"validation_windows={len(validation_dataset)} "
        f"bf16={training_cfg.bf16}"
    )

    for epoch in range(start_epoch, training_cfg.epochs):
        epoch_start = time.perf_counter()
        epoch_generator_state = generator.get_state()
        epoch_losses: list[float] = []
        skipped_batches = resume_batches if epoch == start_epoch else 0
        processed_batches = skipped_batches
        newly_processed_batches = 0
        for batch_index, batch in enumerate(loader):
            if batch_index < skipped_batches:
                continue
            result = train_sequence_batch(
                model,
                batch,
                optimizer,
                device,
                tbptt_ticks,
                use_bf16=training_cfg.bf16,
                max_grad_norm=training_cfg.max_grad_norm,
            )
            global_step += result.optimizer_steps
            epoch_losses.append(result.mean_loss)
            processed_batches = batch_index + 1
            newly_processed_batches += 1
            print(
                f"epoch={epoch + 1}/{training_cfg.epochs} batch={batch_index + 1}/"
                f"{len(loader)} step={global_step} loss={result.mean_loss:.5f}"
            )
            if (
                global_step % training_cfg.checkpoint_every_steps
                < result.optimizer_steps
            ):
                save_checkpoint(
                    output_dir / "checkpoint_last.pt",
                    model,
                    optimizer,
                    epoch,
                    global_step,
                    training_cfg,
                    axis_normalization,
                    control_profile_sha256,
                    telemetry,
                    epoch_complete=False,
                    batches_completed_in_epoch=processed_batches,
                    data_loader_generator_state=epoch_generator_state,
                    best_validation_loss=(
                        None
                        if math.isinf(best_validation_loss)
                        else best_validation_loss
                    ),
                    validation_metrics=last_validation_metrics,
                )
            if max_batches is not None and newly_processed_batches >= max_batches:
                break

        mean_epoch_loss = sum(epoch_losses) / max(len(epoch_losses), 1)
        print(
            f"epoch={epoch + 1} mean_loss={mean_epoch_loss:.5f} "
            f"seconds={time.perf_counter() - epoch_start:.1f}"
        )
        epoch_complete = processed_batches == len(loader)
        if not epoch_complete:
            save_checkpoint(
                output_dir / "checkpoint_last.pt",
                model,
                optimizer,
                epoch,
                global_step,
                training_cfg,
                axis_normalization,
                control_profile_sha256,
                telemetry,
                epoch_complete=False,
                batches_completed_in_epoch=processed_batches,
                data_loader_generator_state=epoch_generator_state,
                best_validation_loss=(
                    None if math.isinf(best_validation_loss) else best_validation_loss
                ),
                validation_metrics=last_validation_metrics,
            )
            return

        validation_start = time.perf_counter()
        validation_metrics = evaluate_model(
            model,
            validation_loader,
            device,
            use_bf16=training_cfg.bf16,
        )
        validation_loss = validation_metrics["loss"]
        improved = validation_loss < best_validation_loss
        if improved:
            best_validation_loss = validation_loss
        last_validation_metrics = validation_metrics
        print(
            f"validation epoch={epoch + 1} loss={validation_loss:.5f} "
            f"movement_accuracy={validation_metrics['movement_accuracy']:.4f} "
            f"movement_macro_f1={validation_metrics['movement_macro_f1']:.4f} "
            f"button_accuracy={validation_metrics['button_accuracy']:.4f} "
            f"button_f1={validation_metrics['button_f1']:.4f} "
            f"axis_mae={validation_metrics['immediate_axis_mae']:.5f} "
            f"seconds={time.perf_counter() - validation_start:.1f}"
        )
        with (output_dir / "metrics.jsonl").open("a", encoding="utf-8") as metrics_file:
            metrics_file.write(
                json.dumps(
                    {
                        "epoch": epoch + 1,
                        "global_step": global_step,
                        "training_loss": mean_epoch_loss,
                        "validation": validation_metrics,
                        "best_validation_loss": best_validation_loss,
                        "is_best": improved,
                    },
                    sort_keys=True,
                )
                + "\n"
            )
        save_checkpoint(
            output_dir / "checkpoint_last.pt",
            model,
            optimizer,
            epoch,
            global_step,
            training_cfg,
            axis_normalization,
            control_profile_sha256,
            telemetry,
            epoch_complete=True,
            batches_completed_in_epoch=processed_batches,
            data_loader_generator_state=generator.get_state(),
            best_validation_loss=best_validation_loss,
            validation_metrics=validation_metrics,
        )
        if improved:
            save_checkpoint(
                output_dir / "checkpoint_best.pt",
                model,
                optimizer,
                epoch,
                global_step,
                training_cfg,
                axis_normalization,
                control_profile_sha256,
                telemetry,
                epoch_complete=True,
                batches_completed_in_epoch=processed_batches,
                data_loader_generator_state=generator.get_state(),
                best_validation_loss=best_validation_loss,
                validation_metrics=validation_metrics,
            )
        resume_batches = 0


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument(
        "--validation-manifest",
        type=Path,
        help="held-out validation manifest (required for training)",
    )
    parser.add_argument("--output", type=Path, default=Path("runs/default"))
    parser.add_argument("--model-config", type=Path)
    parser.add_argument(
        "--tiny", action="store_true", help="use the CPU-test model shape"
    )
    parser.add_argument("--epochs", type=int, default=10)
    parser.add_argument("--batch-size", type=int, default=1)
    parser.add_argument("--learning-rate", type=float, default=3.0e-4)
    parser.add_argument("--weight-decay", type=float, default=0.05)
    parser.add_argument("--history-seconds", type=float, default=30.0)
    parser.add_argument("--optimization-seconds", type=float, default=2.0)
    parser.add_argument("--stride-seconds", type=float, default=2.0)
    parser.add_argument("--tbptt-seconds", type=float, default=0.2)
    parser.add_argument("--num-workers", type=int, default=2)
    parser.add_argument("--seed", type=int, default=1337)
    parser.add_argument("--bf16", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--compile-vision", action="store_true")
    parser.add_argument("--checkpoint-every-steps", type=int, default=250)
    parser.add_argument("--resume", type=Path)
    parser.add_argument("--validate-only", action="store_true")
    parser.add_argument("--max-batches", type=int)
    return parser


def main() -> None:
    args = build_parser().parse_args()
    if args.tiny and args.model_config:
        raise SystemExit("--tiny and --model-config are mutually exclusive")
    model_cfg = (
        ModelConfig.tiny()
        if args.tiny
        else (
            ModelConfig.from_json(args.model_config)
            if args.model_config
            else ModelConfig()
        )
    )
    if args.validate_only:
        summary = dataset_summary(args.manifest, model_cfg)
        window_dataset = DemonstrationWindowDataset(
            args.manifest,
            model_cfg,
            history_seconds=args.history_seconds,
            optimization_seconds=args.optimization_seconds,
            stride_seconds=args.stride_seconds,
        )
        summary["usable_windows"] = len(window_dataset)
        summary["episodes_with_usable_windows"] = len(
            {record_index for record_index, _ in window_dataset.windows}
        )
        print(json.dumps(summary, indent=2))
        return
    if args.validation_manifest is None:
        raise SystemExit("--validation-manifest is required for training")
    training_cfg = TrainingConfig(
        epochs=args.epochs,
        batch_size=args.batch_size,
        learning_rate=args.learning_rate,
        weight_decay=args.weight_decay,
        history_seconds=args.history_seconds,
        optimization_seconds=args.optimization_seconds,
        stride_seconds=args.stride_seconds,
        tbptt_seconds=args.tbptt_seconds,
        num_workers=args.num_workers,
        seed=args.seed,
        bf16=args.bf16,
        compile_vision=args.compile_vision,
        checkpoint_every_steps=args.checkpoint_every_steps,
    )
    run_training(
        args.manifest,
        args.validation_manifest,
        args.output,
        model_cfg,
        training_cfg,
        resume=args.resume,
        max_batches=args.max_batches,
    )


if __name__ == "__main__":
    main()
