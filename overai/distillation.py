"""Distill a trained OverAI checkpoint into a smaller compatible model."""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import time
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

import torch
from torch.nn import functional as F
from torch.nn.utils import clip_grad_norm_
from torch.utils.data import DataLoader

from .config import ModelConfig
from .data import SequenceBatch
from .losses import (
    LossWeights,
    fast_axis_loss,
    horizon_weights,
    slow_control_loss,
    weighted_fast_total,
    weighted_mean,
    weighted_slow_total,
)
from .model import HierarchicalImitationController
from .training import (
    BatchResult,
    TrainingConfig,
    _autocast_context,
    _build_training_datasets,
    _compile_vision_for_training,
    _legacy_batches_completed_in_epoch,
    _make_optimizer,
    _run_tick,
    configure_runtime,
    evaluate_model,
    load_checkpoint,
    parameter_count,
    save_checkpoint,
)
from .types import (
    FastPrediction,
    FastTargets,
    SlowPrediction,
    SlowTargets,
    detach_state,
)


@dataclass(frozen=True, slots=True)
class DistillationConfig:
    """Weights controlling the hard-label and frozen-teacher objectives."""

    teacher_weight: float = 0.7
    label_weight: float = 0.3
    temperature: float = 2.0

    def __post_init__(self) -> None:
        if self.teacher_weight < 0 or self.label_weight < 0:
            raise ValueError("distillation weights cannot be negative")
        if self.teacher_weight + self.label_weight <= 0:
            raise ValueError("at least one distillation weight must be positive")
        if self.temperature <= 0:
            raise ValueError("distillation temperature must be positive")


_COMPATIBILITY_FIELDS = (
    "image_height",
    "image_width",
    "input_channels",
    "channel_order",
    "video_hz",
    "slow_hz",
    "fast_hz",
    "slow_horizon",
    "fast_horizon",
    "num_buttons",
)


def validate_distillation_configs(
    teacher_cfg: ModelConfig, student_cfg: ModelConfig
) -> None:
    """Require identical observable inputs, schedules, and output shapes."""

    mismatches = [
        name
        for name in _COMPATIBILITY_FIELDS
        if getattr(teacher_cfg, name) != getattr(student_cfg, name)
    ]
    if mismatches:
        raise ValueError(
            "teacher and student configs are incompatible: " + ", ".join(mismatches)
        )


def _checkpoint_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as checkpoint_file:
        while chunk := checkpoint_file.read(1024 * 1024):
            digest.update(chunk)
    return digest.hexdigest()


def _load_teacher_payload(path: Path) -> dict[str, Any]:
    payload: dict[str, Any] = torch.load(path, map_location="cpu", weights_only=True)
    if payload.get("format_version") != 4:
        raise ValueError("distillation requires a format-4 teacher checkpoint")
    if not isinstance(payload.get("model_config"), dict):
        raise TypeError("teacher checkpoint is missing model_config")
    if not isinstance(payload.get("model"), dict):
        raise TypeError("teacher checkpoint is missing model weights")
    return payload


def load_teacher_checkpoint(
    path: Path,
    student_cfg: ModelConfig,
    device: torch.device,
    *,
    axis_normalization: dict[str, Any],
    control_profile_sha256: str,
) -> tuple[HierarchicalImitationController, dict[str, Any]]:
    payload = _load_teacher_payload(path)
    teacher_cfg = ModelConfig(**payload["model_config"])
    validate_distillation_configs(teacher_cfg, student_cfg)
    if payload.get("axis_normalization") != axis_normalization:
        raise ValueError("teacher checkpoint axis normalization does not match dataset")
    if payload.get("control_profile_sha256") != control_profile_sha256:
        raise ValueError("teacher checkpoint control profile does not match dataset")
    model = HierarchicalImitationController(teacher_cfg)
    model.load_state_dict(payload["model"])
    model.requires_grad_(False)
    return model.to(device).eval(), payload


def _soft_cross_entropy(
    student_logits: torch.Tensor,
    teacher_logits: torch.Tensor,
    temperature: float,
) -> torch.Tensor:
    return (
        F.kl_div(
            F.log_softmax(student_logits / temperature, dim=-1),
            F.softmax(teacher_logits / temperature, dim=-1),
            reduction="none",
        ).sum(dim=-1)
        * temperature**2
    )


def slow_distillation_loss(
    student: SlowPrediction,
    teacher: SlowPrediction,
    cfg: ModelConfig,
    temperature: float,
) -> torch.Tensor:
    weights = horizon_weights(
        cfg.slow_horizon,
        student.trajectory_movement_logits.device,
        student.trajectory_movement_logits.dtype,
    )
    movement = weighted_mean(
        _soft_cross_entropy(
            student.trajectory_movement_logits,
            teacher.trajectory_movement_logits,
            temperature,
        ).sum(dim=-1),
        weights,
    )
    immediate_movement = _soft_cross_entropy(
        student.immediate_movement_logits,
        teacher.immediate_movement_logits,
        temperature,
    ).mean()
    buttons = weighted_mean(
        F.binary_cross_entropy_with_logits(
            student.trajectory_button_logits / temperature,
            torch.sigmoid(teacher.trajectory_button_logits / temperature),
            reduction="none",
        )
        * temperature**2,
        weights,
    )
    immediate_buttons = (
        F.binary_cross_entropy_with_logits(
            student.immediate_button_logits / temperature,
            torch.sigmoid(teacher.immediate_button_logits / temperature),
        )
        * temperature**2
    )
    return movement + immediate_movement + buttons + immediate_buttons


def fast_distillation_loss(
    student: FastPrediction,
    teacher: FastPrediction,
    cfg: ModelConfig,
) -> torch.Tensor:
    weights = horizon_weights(
        cfg.fast_horizon,
        student.axis_trajectory.device,
        student.axis_trajectory.dtype,
    )
    trajectory = weighted_mean(
        F.huber_loss(
            student.axis_trajectory, teacher.axis_trajectory, reduction="none"
        ),
        weights,
    )
    immediate = F.huber_loss(student.immediate_axes, teacher.immediate_axes)
    student_velocity = student.axis_trajectory[:, 1:] - student.axis_trajectory[:, :-1]
    teacher_velocity = teacher.axis_trajectory[:, 1:] - teacher.axis_trajectory[:, :-1]
    velocity = F.huber_loss(student_velocity, teacher_velocity)
    return trajectory + immediate + 0.1 * velocity


def _run_teacher_student_tick(
    teacher: HierarchicalImitationController,
    student: HierarchicalImitationController,
    batch: SequenceBatch,
    tick: int,
    teacher_state,
    student_state,
    device: torch.device,
):
    frame = None
    if tick % student.cfg.fast_ticks_per_video == 0:
        frame = batch.load_frame(tick // student.cfg.fast_ticks_per_video, device)
    with torch.no_grad():
        teacher_fast, teacher_slow, teacher_state = _run_tick(
            teacher, batch, tick, teacher_state, device, frame=frame
        )
    student_fast, student_slow, student_state = _run_tick(
        student, batch, tick, student_state, device, frame=frame
    )
    return (
        teacher_fast,
        teacher_slow,
        teacher_state,
        student_fast,
        student_slow,
        student_state,
    )


def train_distillation_batch(
    teacher: HierarchicalImitationController,
    student: HierarchicalImitationController,
    batch: SequenceBatch,
    optimizer: torch.optim.Optimizer,
    device: torch.device,
    tbptt_ticks: int,
    use_bf16: bool,
    distillation_cfg: DistillationConfig | None = None,
    max_grad_norm: float = 1.0,
    loss_weights: LossWeights | None = None,
) -> BatchResult:
    """Train one streaming student batch against labels and teacher outputs."""

    cfg = student.cfg
    distillation_cfg = distillation_cfg or DistillationConfig()
    weights = loss_weights or LossWeights()
    teacher.eval()
    student.train()
    teacher_state = teacher.initial_state(batch.batch_size, device)
    student_state = student.initial_state(batch.batch_size, device)
    optimizer.zero_grad(set_to_none=True)

    for tick in range(batch.history_ticks):
        with torch.no_grad(), _autocast_context(device, use_bf16):
            (
                _,
                _,
                teacher_state,
                _,
                _,
                student_state,
            ) = _run_teacher_student_tick(
                teacher,
                student,
                batch,
                tick,
                teacher_state,
                student_state,
                device,
            )
    student_state = detach_state(student_state)

    pending_loss: torch.Tensor | None = None
    pending_predictions = 0
    optimizer_steps = 0
    reported_total = 0.0
    reported_count = 0
    hard_total = 0.0
    teacher_total = 0.0

    for tick in range(batch.history_ticks, batch.process_ticks):
        with _autocast_context(device, use_bf16):
            (
                teacher_fast,
                teacher_slow,
                teacher_state,
                student_fast,
                student_slow,
                student_state,
            ) = _run_teacher_student_tick(
                teacher,
                student,
                batch,
                tick,
                teacher_state,
                student_state,
                device,
            )
            fast_targets = FastTargets(
                axes=batch.axes[:, tick : tick + cfg.fast_horizon].to(
                    device, non_blocking=True
                )
            )
            hard_loss = weighted_fast_total(
                fast_axis_loss(student_fast, fast_targets, cfg), weights
            )
            teacher_loss = fast_distillation_loss(student_fast, teacher_fast, cfg)
            prediction_terms = 1

            if student_slow is not None:
                if teacher_slow is None:
                    raise RuntimeError("teacher and student slow schedules diverged")
                slow_index = tick // cfg.fast_ticks_per_slow
                slow_targets = SlowTargets(
                    movement=batch.movement[
                        :, slow_index : slow_index + cfg.slow_horizon
                    ].to(device, non_blocking=True),
                    buttons=batch.buttons[
                        :, slow_index : slow_index + cfg.slow_horizon
                    ].to(device, non_blocking=True),
                )
                hard_loss = hard_loss + weighted_slow_total(
                    slow_control_loss(student_slow, slow_targets, cfg), weights
                )
                teacher_loss = teacher_loss + slow_distillation_loss(
                    student_slow,
                    teacher_slow,
                    cfg,
                    distillation_cfg.temperature,
                )
                prediction_terms += 1

            tick_loss = (
                distillation_cfg.label_weight * hard_loss
                + distillation_cfg.teacher_weight * teacher_loss
            )

        pending_loss = tick_loss if pending_loss is None else pending_loss + tick_loss
        pending_predictions += prediction_terms
        reported_total += float(tick_loss.detach())
        hard_total += float(hard_loss.detach())
        teacher_total += float(teacher_loss.detach())
        reported_count += prediction_terms

        ticks_in_optimization = tick - batch.history_ticks + 1
        boundary = ticks_in_optimization % tbptt_ticks == 0
        final_tick = tick + 1 == batch.process_ticks
        if boundary or final_tick:
            if pending_loss is None or pending_predictions == 0:
                raise RuntimeError("TBPTT chunk contains no predictions")
            normalized_loss = pending_loss / pending_predictions
            if not torch.isfinite(normalized_loss):
                raise FloatingPointError(
                    f"non-finite distillation loss: {normalized_loss}"
                )
            normalized_loss.backward()
            clip_grad_norm_(student.parameters(), max_grad_norm)
            optimizer.step()
            optimizer.zero_grad(set_to_none=True)
            optimizer_steps += 1
            student_state = detach_state(student_state)
            pending_loss = None
            pending_predictions = 0

    denominator = max(reported_count, 1)
    return BatchResult(
        mean_loss=reported_total / denominator,
        optimizer_steps=optimizer_steps,
        metrics={
            "hard_label_loss": hard_total / denominator,
            "teacher_loss": teacher_total / denominator,
        },
    )


def _distillation_metadata(
    teacher_checkpoint: Path,
    teacher_payload: dict[str, Any],
    distillation_cfg: DistillationConfig,
) -> dict[str, Any]:
    return {
        "teacher_checkpoint_sha256": _checkpoint_sha256(teacher_checkpoint),
        "teacher_model_config": teacher_payload["model_config"],
        "config": asdict(distillation_cfg),
    }


def _save_student_checkpoint(
    path: Path,
    student: HierarchicalImitationController,
    optimizer: torch.optim.Optimizer,
    epoch: int,
    global_step: int,
    training_cfg: TrainingConfig,
    axis_normalization: dict[str, Any],
    control_profile_sha256: str,
    epoch_complete: bool,
    batches_completed_in_epoch: int,
    generator_state: torch.Tensor,
    distillation_metadata: dict[str, Any],
    best_validation_loss: float | None,
    validation_metrics: dict[str, float] | None,
) -> None:
    save_checkpoint(
        path,
        student,
        optimizer,
        epoch,
        global_step,
        training_cfg,
        axis_normalization,
        control_profile_sha256,
        epoch_complete,
        batches_completed_in_epoch,
        generator_state,
        best_validation_loss,
        validation_metrics,
        extra={"distillation": distillation_metadata},
    )


def run_distillation(
    teacher_checkpoint: Path,
    manifest: Path,
    validation_manifest: Path,
    output_dir: Path,
    student_cfg: ModelConfig,
    training_cfg: TrainingConfig,
    distillation_cfg: DistillationConfig,
    resume: Path | None = None,
    max_batches: int | None = None,
) -> None:
    if not torch.cuda.is_available():
        raise RuntimeError("distillation requires an NVIDIA CUDA device")
    device = torch.device("cuda")
    if training_cfg.bf16 and not torch.cuda.is_bf16_supported():
        raise RuntimeError("the selected CUDA device does not support bfloat16")
    configure_runtime(device, training_cfg.seed)

    dataset, validation_dataset, manifest_payload, _ = _build_training_datasets(
        manifest, validation_manifest, student_cfg, training_cfg
    )
    axis_normalization = manifest_payload.get("axis_normalization")
    if not isinstance(axis_normalization, dict):
        raise TypeError("training manifest is missing axis_normalization")
    control_profile_sha256 = manifest_payload.get("control_profile_sha256")
    if not isinstance(control_profile_sha256, str) or not control_profile_sha256:
        raise TypeError("training manifest is missing control_profile_sha256")

    teacher, teacher_payload = load_teacher_checkpoint(
        teacher_checkpoint,
        student_cfg,
        device,
        axis_normalization=axis_normalization,
        control_profile_sha256=control_profile_sha256,
    )
    metadata = _distillation_metadata(
        teacher_checkpoint, teacher_payload, distillation_cfg
    )
    student = HierarchicalImitationController(student_cfg).to(device)
    optimizer = _make_optimizer(student, training_cfg, device)
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

    start_epoch = 0
    global_step = 0
    resume_batches: int | None = 0
    best_validation_loss = math.inf
    last_validation_metrics: dict[str, float] | None = None
    if resume is not None:
        resume_payload = torch.load(resume, map_location="cpu", weights_only=True)
        if resume_payload.get("distillation") != metadata:
            raise ValueError("resume checkpoint distillation metadata does not match")
        (
            start_epoch,
            global_step,
            resume_batches,
            generator_state,
            best_validation_loss,
            last_validation_metrics,
        ) = load_checkpoint(
            resume,
            student,
            optimizer,
            axis_normalization=axis_normalization,
            control_profile_sha256=control_profile_sha256,
        )
        if generator_state is not None:
            generator.set_state(generator_state)
    if training_cfg.compile_vision:
        _compile_vision_for_training(student)

    tbptt_ticks = round(training_cfg.tbptt_seconds * student_cfg.fast_hz)
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
    student_cfg.to_json(output_dir / "model_config.json")
    (output_dir / "training_config.json").write_text(
        json.dumps(asdict(training_cfg), indent=2) + "\n", encoding="utf-8"
    )
    (output_dir / "distillation_config.json").write_text(
        json.dumps(metadata, indent=2) + "\n", encoding="utf-8"
    )
    properties = torch.cuda.get_device_properties(0)
    print(
        f"device={properties.name} vram_gib={properties.total_memory / 2**30:.1f} "
        f"teacher_parameters={parameter_count(teacher):,} "
        f"student_parameters={parameter_count(student):,} windows={len(dataset)} "
        f"validation_windows={len(validation_dataset)} bf16={training_cfg.bf16}"
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
            result = train_distillation_batch(
                teacher,
                student,
                batch,
                optimizer,
                device,
                tbptt_ticks,
                training_cfg.bf16,
                distillation_cfg,
                training_cfg.max_grad_norm,
            )
            global_step += result.optimizer_steps
            epoch_losses.append(result.mean_loss)
            processed_batches = batch_index + 1
            newly_processed_batches += 1
            print(
                f"epoch={epoch + 1}/{training_cfg.epochs} batch={batch_index + 1}/"
                f"{len(loader)} step={global_step} loss={result.mean_loss:.5f} "
                f"hard={result.metrics['hard_label_loss']:.5f} "
                f"teacher={result.metrics['teacher_loss']:.5f}"
            )
            if (
                global_step % training_cfg.checkpoint_every_steps
                < result.optimizer_steps
            ):
                _save_student_checkpoint(
                    output_dir / "checkpoint_last.pt",
                    student,
                    optimizer,
                    epoch,
                    global_step,
                    training_cfg,
                    axis_normalization,
                    control_profile_sha256,
                    False,
                    processed_batches,
                    epoch_generator_state,
                    metadata,
                    None if math.isinf(best_validation_loss) else best_validation_loss,
                    last_validation_metrics,
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
            _save_student_checkpoint(
                output_dir / "checkpoint_last.pt",
                student,
                optimizer,
                epoch,
                global_step,
                training_cfg,
                axis_normalization,
                control_profile_sha256,
                False,
                processed_batches,
                epoch_generator_state,
                metadata,
                None if math.isinf(best_validation_loss) else best_validation_loss,
                last_validation_metrics,
            )
            return

        validation_metrics = evaluate_model(
            student, validation_loader, device, use_bf16=training_cfg.bf16
        )
        validation_loss = validation_metrics["loss"]
        improved = validation_loss < best_validation_loss
        if improved:
            best_validation_loss = validation_loss
        last_validation_metrics = validation_metrics
        print(
            f"validation epoch={epoch + 1} loss={validation_loss:.5f} "
            f"movement_accuracy={validation_metrics['movement_accuracy']:.4f} "
            f"button_f1={validation_metrics['button_f1']:.4f} "
            f"axis_mae={validation_metrics['immediate_axis_mae']:.5f}"
        )
        with (output_dir / "metrics.jsonl").open("a", encoding="utf-8") as metrics_file:
            metrics_file.write(
                json.dumps(
                    {
                        "epoch": epoch + 1,
                        "global_step": global_step,
                        "distillation_loss": mean_epoch_loss,
                        "validation": validation_metrics,
                        "best_validation_loss": best_validation_loss,
                        "is_best": improved,
                    },
                    sort_keys=True,
                )
                + "\n"
            )
        for checkpoint_name in (
            "checkpoint_last.pt",
            *(("checkpoint_best.pt",) if improved else ()),
        ):
            _save_student_checkpoint(
                output_dir / checkpoint_name,
                student,
                optimizer,
                epoch,
                global_step,
                training_cfg,
                axis_normalization,
                control_profile_sha256,
                True,
                processed_batches,
                generator.get_state(),
                metadata,
                best_validation_loss,
                validation_metrics,
            )
        resume_batches = 0


def _validate_only(
    teacher_checkpoint: Path,
    student_cfg: ModelConfig,
    manifest: Path,
    validation_manifest: Path,
    training_cfg: TrainingConfig,
) -> None:
    dataset, validation_dataset, manifest_payload, _ = _build_training_datasets(
        manifest, validation_manifest, student_cfg, training_cfg
    )
    payload = _load_teacher_payload(teacher_checkpoint)
    teacher_cfg = ModelConfig(**payload["model_config"])
    validate_distillation_configs(teacher_cfg, student_cfg)
    if payload.get("axis_normalization") != manifest_payload.get("axis_normalization"):
        raise ValueError("teacher checkpoint axis normalization does not match dataset")
    if payload.get("control_profile_sha256") != manifest_payload.get(
        "control_profile_sha256"
    ):
        raise ValueError("teacher checkpoint control profile does not match dataset")
    print(
        json.dumps(
            {
                "teacher_model_config": teacher_cfg.to_dict(),
                "student_model_config": student_cfg.to_dict(),
                "training_windows": len(dataset),
                "validation_windows": len(validation_dataset),
                "compatible": True,
            },
            indent=2,
        )
    )


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--teacher-checkpoint", type=Path, required=True)
    parser.add_argument("--student-config", type=Path, required=True)
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--validation-manifest", type=Path, required=True)
    parser.add_argument("--output", type=Path, default=Path("runs/distilled"))
    parser.add_argument("--epochs", type=int, default=10)
    parser.add_argument("--batch-size", type=int, default=1)
    parser.add_argument("--learning-rate", type=float, default=3.0e-4)
    parser.add_argument("--weight-decay", type=float, default=0.05)
    parser.add_argument("--history-seconds", type=float, default=5.0)
    parser.add_argument("--optimization-seconds", type=float, default=2.0)
    parser.add_argument("--stride-seconds", type=float, default=2.0)
    parser.add_argument("--tbptt-seconds", type=float, default=0.1)
    parser.add_argument("--num-workers", type=int, default=2)
    parser.add_argument("--seed", type=int, default=1337)
    parser.add_argument("--bf16", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--compile-vision", action="store_true")
    parser.add_argument("--checkpoint-every-steps", type=int, default=120)
    parser.add_argument("--teacher-weight", type=float, default=0.7)
    parser.add_argument("--label-weight", type=float, default=0.3)
    parser.add_argument("--temperature", type=float, default=2.0)
    parser.add_argument("--resume", type=Path)
    parser.add_argument("--validate-only", action="store_true")
    parser.add_argument("--max-batches", type=int)
    return parser


def main() -> None:
    args = build_parser().parse_args()
    student_cfg = ModelConfig.from_json(args.student_config)
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
    distillation_cfg = DistillationConfig(
        teacher_weight=args.teacher_weight,
        label_weight=args.label_weight,
        temperature=args.temperature,
    )
    if args.validate_only:
        _validate_only(
            args.teacher_checkpoint,
            student_cfg,
            args.manifest,
            args.validation_manifest,
            training_cfg,
        )
        return
    run_distillation(
        args.teacher_checkpoint,
        args.manifest,
        args.validation_manifest,
        args.output,
        student_cfg,
        training_cfg,
        distillation_cfg,
        resume=args.resume,
        max_batches=args.max_batches,
    )


if __name__ == "__main__":
    main()
