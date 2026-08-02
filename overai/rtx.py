"""RTX 4080-only ONNX/TensorRT-RTX artifact export and execution CLIs."""

from __future__ import annotations

import argparse
import ctypes
import hashlib
import json
import os
import shutil
import subprocess
import time
from pathlib import Path
from typing import Any, Literal

import torch
from torch import nn

from .blocks import ExportableAttention, ExportableLayerNorm
from .config import ModelConfig
from .model import HierarchicalImitationController
from .recording import (
    AxisDenormalizer,
    AxisNormalization,
    ControlProfile,
    create_native_backend,
)
from .telemetry import TelemetryWorker, coerce_captured_frame, create_telemetry_worker
from .types import (
    ControllerState,
    ExecutedActions,
    FastControllerState,
    HierarchicalMemoryState,
    ObservationContext,
    TimingContext,
)

RTX_GPU_NAME = "NVIDIA GeForce RTX 4080"
RTX_COMPUTE_CAPABILITY = (8, 9)
ARTIFACT_VERSION = 2


def _torch_dtype(trt_dtype: Any) -> torch.dtype:
    import tensorrt_rtx as trt  # type: ignore[import-not-found]

    mapping = {
        trt.DataType.FLOAT: torch.float32,
        trt.DataType.HALF: torch.float16,
        trt.DataType.BF16: torch.bfloat16,
        trt.DataType.INT8: torch.int8,
        trt.DataType.INT32: torch.int32,
        trt.DataType.INT64: torch.int64,
        trt.DataType.BOOL: torch.bool,
        trt.DataType.UINT8: torch.uint8,
    }
    if trt_dtype not in mapping:
        raise TypeError(f"unsupported TensorRT tensor dtype: {trt_dtype}")
    return mapping[trt_dtype]


class TensorRtxEngine:
    """Persistent-buffer TensorRT-RTX engine with whole-graph CUDA capture."""

    def __init__(self, engine_path: Path, device: torch.device) -> None:
        import tensorrt_rtx as trt  # type: ignore[import-not-found]

        logger = trt.Logger(trt.Logger.WARNING)
        runtime = trt.Runtime(logger)
        engine = runtime.deserialize_cuda_engine(engine_path.read_bytes())
        if engine is None:
            raise RuntimeError(f"failed to deserialize TensorRT engine: {engine_path}")
        runtime_config = engine.create_runtime_config()
        runtime_config.cuda_graph_strategy = trt.CudaGraphStrategy.WHOLE_GRAPH_CAPTURE
        runtime_config.set_execution_context_allocation_strategy(
            trt.ExecutionContextAllocationStrategy.USER_MANAGED
        )
        context = engine.create_execution_context(runtime_config)
        if context is None:
            raise RuntimeError(f"failed to create TensorRT context: {engine_path}")
        self.runtime = runtime
        self.engine = engine
        self.context = context
        self.device = torch.device(
            f"cuda:{torch.cuda.current_device()}" if device.index is None else device
        )
        self.device_memory = torch.empty(
            engine.device_memory_size_v2, dtype=torch.uint8, device=device
        )
        context.device_memory = self.device_memory.data_ptr()
        self.input_names: list[str] = []
        self.output_names: list[str] = []
        self.output_buffers: dict[str, torch.Tensor] = {}
        self.stream = torch.cuda.Stream(device=self.device)
        for index in range(engine.num_io_tensors):
            name = engine.get_tensor_name(index)
            shape = tuple(engine.get_tensor_shape(name))
            if any(dimension < 0 for dimension in shape):
                raise ValueError(
                    f"dynamic TensorRT shape is not allowed: {name}={shape}"
                )
            if engine.get_tensor_mode(name) == trt.TensorIOMode.INPUT:
                self.input_names.append(name)
            else:
                self.output_names.append(name)
                self.output_buffers[name] = torch.empty(
                    shape,
                    dtype=_torch_dtype(engine.get_tensor_dtype(name)),
                    device=device,
                )
                if not context.set_tensor_address(
                    name, self.output_buffers[name].data_ptr()
                ):
                    raise RuntimeError(f"failed to bind TensorRT output {name}")

    def execute(self, inputs: dict[str, torch.Tensor]) -> dict[str, torch.Tensor]:
        if set(inputs) != set(self.input_names):
            missing = set(self.input_names) - set(inputs)
            extra = set(inputs) - set(self.input_names)
            raise ValueError(
                f"TensorRT input mismatch; missing={missing}, extra={extra}"
            )
        for name in self.input_names:
            tensor = inputs[name]
            expected_shape = tuple(self.engine.get_tensor_shape(name))
            expected_dtype = _torch_dtype(self.engine.get_tensor_dtype(name))
            if tuple(tensor.shape) != expected_shape or tensor.dtype != expected_dtype:
                raise ValueError(
                    f"{name}: got {tuple(tensor.shape)} {tensor.dtype}, expected "
                    f"{expected_shape} {expected_dtype}"
                )
            if tensor.device != self.device or not tensor.is_contiguous():
                raise ValueError(f"{name} must be contiguous on {self.device}")
            if not self.context.set_tensor_address(name, tensor.data_ptr()):
                raise RuntimeError(f"failed to bind TensorRT input {name}")
        caller_stream = torch.cuda.current_stream(self.device)
        self.stream.wait_stream(caller_stream)
        if not self.context.execute_async_v3(self.stream.cuda_stream):
            raise RuntimeError("TensorRT execution failed")
        caller_stream.wait_stream(self.stream)
        return self.output_buffers


class Rtx4080Controller:
    """Deterministic fixed-rate state machine over the TensorRT engines."""

    def __init__(self, artifact_dir: Path, device: torch.device) -> None:
        manifest = json.loads(
            (artifact_dir / "artifact.json").read_text(encoding="utf-8")
        )
        self.cfg = ModelConfig(**manifest["model_config"])
        self.device = device
        self.engines = {
            name: TensorRtxEngine(artifact_dir / graph["engine"], device)
            for name, graph in manifest["graphs"].items()
        }
        state = HierarchicalImitationController(self.cfg).initial_state(
            1, device, dtype=torch.float16
        )
        self.state: dict[str, torch.Tensor] = {
            "recent": state.memory.recent,
            "intermediate": state.memory.intermediate,
            "long": state.memory.long,
            "recent_valid": state.memory.recent_valid,
            "intermediate_valid": state.memory.intermediate_valid,
            "long_valid": state.memory.long_valid,
            "fast_hidden": state.fast.hidden,
            "fast_previous_trajectory": state.fast.previous_trajectory,
            "previous_axis_trajectory": state.previous_axis_trajectory,
            "previous_slow_trajectory": state.previous_slow_trajectory,
            "shared_tokens": torch.zeros(
                1,
                self.cfg.control_query_tokens,
                self.cfg.model_dim,
                dtype=torch.float16,
                device=device,
            ),
        }
        self.video_frame_index = 0

    def _metadata(self, values: dict[str, torch.Tensor]) -> dict[str, torch.Tensor]:
        return {name: values[name] for name in METADATA_INPUT_NAMES}

    def step(
        self,
        tick: int,
        metadata: dict[str, torch.Tensor],
        frame: torch.Tensor | None,
    ) -> tuple[torch.Tensor, torch.Tensor | None, torch.Tensor | None]:
        slow_due = tick % self.cfg.fast_ticks_per_slow == 0
        if frame is not None:
            self.video_frame_index += 1
            frames_per_long = (
                self.cfg.frames_per_intermediate * self.cfg.intermediate_per_long
            )
            if self.video_frame_index % frames_per_long == 0:
                name = "video_long"
            elif self.video_frame_index % self.cfg.frames_per_intermediate == 0:
                name = "video_intermediate"
            else:
                name = "video_ordinary"
            inputs = {
                "frame": frame,
                **self._metadata(metadata),
                **{
                    key: self.state[key]
                    for key in (
                        "recent",
                        "intermediate",
                        "long",
                        "recent_valid",
                        "intermediate_valid",
                        "long_valid",
                        "fast_hidden",
                        "fast_previous_trajectory",
                        "previous_axis_trajectory",
                        "previous_slow_trajectory",
                    )
                },
            }
            outputs = self.engines[name].execute(inputs)
            for key in (
                "recent",
                "intermediate",
                "long",
                "recent_valid",
                "intermediate_valid",
                "long_valid",
                "fast_hidden",
                "fast_previous_trajectory",
                "previous_axis_trajectory",
                "previous_slow_trajectory",
            ):
                self.state[key].copy_(outputs[f"next_{key}"])
            self.state["shared_tokens"].copy_(outputs["shared_tokens"])
            axes = outputs["immediate_axes"]
        else:
            inputs = {
                **self._metadata(metadata),
                "shared_tokens": self.state["shared_tokens"],
                "fast_hidden": self.state["fast_hidden"],
                "fast_previous_trajectory": self.state["fast_previous_trajectory"],
                "previous_axis_trajectory": self.state["previous_axis_trajectory"],
                "previous_slow_trajectory": self.state["previous_slow_trajectory"],
            }
            outputs = self.engines["fast_tick"].execute(inputs)
            self.state["fast_hidden"].copy_(outputs["next_fast_hidden"])
            self.state["fast_previous_trajectory"].copy_(
                outputs["next_fast_previous_trajectory"]
            )
            self.state["previous_axis_trajectory"].copy_(outputs["axis_trajectory"])
            axes = outputs["immediate_axes"]

        if not slow_due:
            return axes, None, None
        slow_outputs = self.engines["slow_tick"].execute(
            {"shared_tokens": self.state["shared_tokens"]}
        )
        self.state["previous_slow_trajectory"].copy_(
            slow_outputs["next_previous_slow_trajectory"]
        )
        return (
            axes,
            slow_outputs["immediate_movement_logits"],
            slow_outputs["immediate_button_logits"],
        )


def _checkpoint_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as checkpoint:
        while chunk := checkpoint.read(1024 * 1024):
            digest.update(chunk)
    return digest.hexdigest()


def _load_checkpoint(
    path: Path,
) -> tuple[HierarchicalImitationController, dict[str, Any]]:
    payload: dict[str, Any] = torch.load(path, map_location="cpu", weights_only=True)
    if payload.get("format_version") != 3:
        raise ValueError("RTX export requires a format-3 RB checkpoint")
    cfg = ModelConfig(**payload["model_config"])
    model = HierarchicalImitationController(cfg)
    model.load_state_dict(payload["model"])
    return model.eval(), payload


def _metadata_inputs(
    cfg: ModelConfig,
    dtype: torch.dtype,
) -> tuple[torch.Tensor, ...]:
    one = lambda: torch.zeros(1, 1, dtype=dtype)
    return (
        one(),  # health
        one(),  # damage
        one(),  # kill
        one(),  # charge
        torch.ones(1, 2, dtype=torch.long),  # movement
        torch.zeros(1, cfg.num_buttons, dtype=dtype),  # buttons
        torch.zeros(1, 2, dtype=dtype),  # executed axes
        one(),  # absolute time
        one(),  # since video
        one(),  # since slow
        one(),  # delta
    )


METADATA_INPUT_NAMES = [
    "health",
    "damage_event",
    "kill_event",
    "charge",
    "movement",
    "buttons",
    "executed_axes",
    "absolute_time",
    "since_video_frame",
    "since_slow_update",
    "fast_delta_time",
]


def _typed_metadata(
    values: tuple[torch.Tensor, ...],
) -> tuple[ObservationContext, ExecutedActions, TimingContext]:
    return (
        ObservationContext(values[0], values[1], values[2], values[3]),
        ExecutedActions(values[4], values[5], values[6]),
        TimingContext(values[7], values[8], values[9], values[10]),
    )


class VideoPath(nn.Module):
    """A fixed-branch video graph suitable for a static TensorRT engine."""

    def __init__(
        self,
        model: HierarchicalImitationController,
        variant: Literal["ordinary", "intermediate", "long"],
    ) -> None:
        super().__init__()
        self.model = model
        self.variant = variant

    def forward(
        self,
        frame: torch.Tensor,
        health: torch.Tensor,
        damage_event: torch.Tensor,
        kill_event: torch.Tensor,
        charge: torch.Tensor,
        movement: torch.Tensor,
        buttons: torch.Tensor,
        executed_axes: torch.Tensor,
        absolute_time: torch.Tensor,
        since_video_frame: torch.Tensor,
        since_slow_update: torch.Tensor,
        fast_delta_time: torch.Tensor,
        recent: torch.Tensor,
        intermediate: torch.Tensor,
        long: torch.Tensor,
        recent_valid: torch.Tensor,
        intermediate_valid: torch.Tensor,
        long_valid: torch.Tensor,
        fast_hidden: torch.Tensor,
        fast_previous_trajectory: torch.Tensor,
        previous_axis_trajectory: torch.Tensor,
        previous_slow_trajectory: torch.Tensor,
    ) -> tuple[torch.Tensor, ...]:
        metadata = _typed_metadata(
            (
                health,
                damage_event,
                kill_event,
                charge,
                movement,
                buttons,
                executed_axes,
                absolute_time,
                since_video_frame,
                since_slow_update,
                fast_delta_time,
            )
        )
        if self.variant == "ordinary":
            frame_counter, intermediate_counter = 0, 0
        elif self.variant == "intermediate":
            frame_counter = self.model.cfg.frames_per_intermediate - 1
            intermediate_counter = 0
        else:
            frame_counter = (
                self.model.cfg.frames_per_intermediate
                * self.model.cfg.intermediate_per_long
                - 1
            )
            intermediate_counter = self.model.cfg.intermediate_per_long - 1
        state = ControllerState(
            memory=HierarchicalMemoryState(
                recent,
                intermediate,
                long,
                recent_valid,
                intermediate_valid,
                long_valid,
                frame_counter,
                intermediate_counter,
            ),
            fast=FastControllerState(fast_hidden, fast_previous_trajectory),
            current_grid=None,
            shared_tokens=None,
            previous_axis_trajectory=previous_axis_trajectory,
            previous_slow_trajectory=previous_slow_trajectory,
        )
        output = self.model.on_video_frame(
            frame,
            metadata[0],
            metadata[1],
            metadata[2],
            state,
            run_slow_decoder=False,
        )
        next_state = output.state
        shared = next_state.shared_tokens
        assert shared is not None
        common = (
            next_state.memory.recent,
            next_state.memory.intermediate,
            next_state.memory.long,
            next_state.memory.recent_valid,
            next_state.memory.intermediate_valid,
            next_state.memory.long_valid,
            next_state.fast.hidden,
            next_state.fast.previous_trajectory,
            next_state.previous_axis_trajectory,
            next_state.previous_slow_trajectory,
            shared,
            output.fast.immediate_axes,
            output.fast.axis_trajectory,
        )
        return common


VIDEO_INPUT_NAMES = [
    "frame",
    *METADATA_INPUT_NAMES,
    "recent",
    "intermediate",
    "long",
    "recent_valid",
    "intermediate_valid",
    "long_valid",
    "fast_hidden",
    "fast_previous_trajectory",
    "previous_axis_trajectory",
    "previous_slow_trajectory",
]
VIDEO_COMMON_OUTPUT_NAMES = [
    "next_recent",
    "next_intermediate",
    "next_long",
    "next_recent_valid",
    "next_intermediate_valid",
    "next_long_valid",
    "next_fast_hidden",
    "next_fast_previous_trajectory",
    "next_previous_axis_trajectory",
    "next_previous_slow_trajectory",
    "shared_tokens",
    "immediate_axes",
    "axis_trajectory",
]


class FastPath(nn.Module):
    def __init__(self, model: HierarchicalImitationController) -> None:
        super().__init__()
        self.model = model

    def forward(
        self,
        health: torch.Tensor,
        damage_event: torch.Tensor,
        kill_event: torch.Tensor,
        charge: torch.Tensor,
        movement: torch.Tensor,
        buttons: torch.Tensor,
        executed_axes: torch.Tensor,
        absolute_time: torch.Tensor,
        since_video_frame: torch.Tensor,
        since_slow_update: torch.Tensor,
        fast_delta_time: torch.Tensor,
        shared_tokens: torch.Tensor,
        fast_hidden: torch.Tensor,
        fast_previous_trajectory: torch.Tensor,
        previous_axis_trajectory: torch.Tensor,
        previous_slow_trajectory: torch.Tensor,
    ) -> tuple[torch.Tensor, ...]:
        context, actions, timing = _typed_metadata(
            (
                health,
                damage_event,
                kill_event,
                charge,
                movement,
                buttons,
                executed_axes,
                absolute_time,
                since_video_frame,
                since_slow_update,
                fast_delta_time,
            )
        )
        context_embedding = self.model.observation_encoder(context)
        action_embedding = self.model.executed_action_encoder(actions)
        timing_embedding = self.model.time_encoder(
            torch.cat(
                (
                    timing.absolute_time,
                    timing.since_video_frame,
                    timing.since_slow_update,
                    timing.fast_delta_time,
                ),
                dim=-1,
            )
        )
        trajectory_tokens = self.model.trajectory_encoder(
            previous_axis_trajectory, previous_slow_trajectory
        )
        _ = torch.cat(
            (
                torch.stack(
                    (
                        context_embedding,
                        self.model.action_to_model(action_embedding),
                        timing_embedding,
                    ),
                    dim=1,
                ),
                trajectory_tokens,
            ),
            dim=1,
        )
        prediction = self.model.fast_decoder(
            shared_tokens,
            action_embedding,
            timing_embedding,
            FastControllerState(fast_hidden, fast_previous_trajectory),
        )
        return (
            prediction.immediate_axes,
            prediction.axis_trajectory,
            prediction.next_state.hidden,
            prediction.next_state.previous_trajectory,
        )


class SlowPath(nn.Module):
    """Standalone slow decoder so discrete control follows the fast-tick phase."""

    def __init__(self, model: HierarchicalImitationController) -> None:
        super().__init__()
        self.model = model

    def forward(
        self, shared_tokens: torch.Tensor
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        prediction = self.model.slow_decoder(shared_tokens)
        return (
            prediction.immediate_movement_logits,
            prediction.immediate_button_logits,
            self.model._slow_trajectory(prediction),
        )


def _example_video_inputs(
    cfg: ModelConfig, dtype: torch.dtype
) -> tuple[torch.Tensor, ...]:
    state = HierarchicalImitationController(cfg).initial_state(1, "cpu", dtype=dtype)
    return (
        torch.zeros(
            1,
            cfg.input_channels,
            cfg.image_height,
            cfg.image_width,
            dtype=torch.uint8,
        ),
        *_metadata_inputs(cfg, dtype),
        state.memory.recent,
        state.memory.intermediate,
        state.memory.long,
        state.memory.recent_valid,
        state.memory.intermediate_valid,
        state.memory.long_valid,
        state.fast.hidden,
        state.fast.previous_trajectory,
        state.previous_axis_trajectory,
        state.previous_slow_trajectory,
    )


def _export_onnx(
    module: nn.Module,
    inputs: tuple[torch.Tensor, ...],
    output: Path,
    input_names: list[str],
    output_names: list[str],
) -> None:
    try:
        import onnx  # pyright: ignore[reportMissingImports]  # noqa: F401
        import onnxscript  # pyright: ignore[reportMissingImports]  # noqa: F401
    except ImportError as error:
        raise RuntimeError(
            "required ONNX dependencies are missing; reinstall the project"
        ) from error
    module.eval()
    torch.onnx.export(
        module,
        inputs,
        output,
        input_names=input_names,
        output_names=output_names,
        opset_version=22,
        dynamo=True,
        external_data=True,
        verbose=False,
    )


def export_rtx_artifact(
    checkpoint: Path,
    output_dir: Path,
    *,
    fp32_attention: bool = False,
    fp32_layernorm: bool = False,
) -> Path:
    if output_dir.exists() and any(output_dir.iterdir()):
        raise FileExistsError(f"artifact output is not empty: {output_dir}")
    output_dir.mkdir(parents=True, exist_ok=True)
    model, payload = _load_checkpoint(checkpoint)
    model = model.half().eval()
    for module in model.modules():
        if fp32_attention and isinstance(module, ExportableAttention):
            module.export_attention_fp32 = True
        elif fp32_layernorm and isinstance(module, ExportableLayerNorm):
            module.export_fp32 = True
    cfg = model.cfg
    video_inputs = _example_video_inputs(cfg, torch.float16)
    graphs: dict[str, dict[str, str]] = {}
    for variant in ("ordinary", "intermediate", "long"):
        graph_path = output_dir / f"video_{variant}.onnx"
        output_names = list(VIDEO_COMMON_OUTPUT_NAMES)
        _export_onnx(
            VideoPath(model, variant),
            video_inputs,
            graph_path,
            VIDEO_INPUT_NAMES,
            output_names,
        )
        graphs[f"video_{variant}"] = {"onnx": graph_path.name}

    metadata = _metadata_inputs(cfg, torch.float16)
    state = model.initial_state(1, "cpu", dtype=torch.float16)
    fast_inputs = (
        *metadata,
        torch.zeros(1, cfg.control_query_tokens, cfg.model_dim, dtype=torch.float16),
        state.fast.hidden,
        state.fast.previous_trajectory,
        state.previous_axis_trajectory,
        state.previous_slow_trajectory,
    )
    fast_path = output_dir / "fast_tick.onnx"
    _export_onnx(
        FastPath(model),
        fast_inputs,
        fast_path,
        METADATA_INPUT_NAMES
        + [
            "shared_tokens",
            "fast_hidden",
            "fast_previous_trajectory",
            "previous_axis_trajectory",
            "previous_slow_trajectory",
        ],
        [
            "immediate_axes",
            "axis_trajectory",
            "next_fast_hidden",
            "next_fast_previous_trajectory",
        ],
    )
    graphs["fast_tick"] = {"onnx": fast_path.name}
    slow_path = output_dir / "slow_tick.onnx"
    _export_onnx(
        SlowPath(model),
        (torch.zeros(1, cfg.control_query_tokens, cfg.model_dim, dtype=torch.float16),),
        slow_path,
        ["shared_tokens"],
        [
            "immediate_movement_logits",
            "immediate_button_logits",
            "next_previous_slow_trajectory",
        ],
    )
    graphs["slow_tick"] = {"onnx": slow_path.name}
    artifact = {
        "artifact_version": ARTIFACT_VERSION,
        "checkpoint_format": 3,
        "checkpoint_sha256": _checkpoint_sha256(checkpoint),
        "model_config": cfg.to_dict(),
        "training_config": payload.get("training_config"),
        "precision": "fp16",
        "precision_overrides": [
            name
            for enabled, name in (
                (fp32_attention, "attention_sdpa_softmax:fp32"),
                (fp32_layernorm, "layer_norm:fp32"),
            )
            if enabled
        ],
        "channels": ["R", "B"],
        "axis_normalization": payload.get("axis_normalization"),
        "control_profile_sha256": payload.get("control_profile_sha256"),
        # Every format-3 checkpoint predating this field used zero telemetry.
        "telemetry": payload.get(
            "telemetry", {"provider": "zero", "sha256": None}
        ),
        "required_gpu": RTX_GPU_NAME,
        "required_compute_capability": list(RTX_COMPUTE_CAPABILITY),
        "graphs": graphs,
    }
    manifest_path = output_dir / "artifact.json"
    manifest_path.write_text(json.dumps(artifact, indent=2) + "\n", encoding="utf-8")
    return manifest_path


def _rtx_executable() -> str | None:
    executable = shutil.which("tensorrt_rtx") or shutil.which("tensorrt_rtx.exe")
    return executable


def _build_engine_python(onnx_path: Path, engine_path: Path) -> str:
    try:
        import tensorrt_rtx as trt  # type: ignore[import-not-found]
    except ImportError as error:
        raise RuntimeError(
            "TensorRT-RTX SDK CLI or Python package is required to build engines"
        ) from error
    logger = trt.Logger(trt.Logger.WARNING)
    builder = trt.Builder(logger)
    network = builder.create_network(
        1 << int(trt.NetworkDefinitionCreationFlag.STRONGLY_TYPED)
    )
    config = builder.create_builder_config()
    config.set_flag(trt.BuilderFlag.REQUIRE_USER_ALLOCATION)
    parser = trt.OnnxParser(network, logger)
    if not parser.parse_from_file(str(onnx_path)):
        errors = "\n".join(
            str(parser.get_error(index)) for index in range(parser.num_errors)
        )
        raise RuntimeError(f"TensorRT-RTX could not parse {onnx_path}:\n{errors}")
    serialized = builder.build_serialized_network(network, config)
    if serialized is None:
        raise RuntimeError(f"TensorRT-RTX failed to build {onnx_path}")
    engine_path.write_bytes(bytes(serialized))
    return str(trt.__version__)


def build_engines(artifact_dir: Path) -> None:
    manifest_path = artifact_dir / "artifact.json"
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    executable = _rtx_executable()
    runtime_version = "sdk-cli"
    for name, graph in manifest["graphs"].items():
        onnx_path = artifact_dir / graph["onnx"]
        engine_path = artifact_dir / f"{name}.rtxplan"
        if executable is None:
            runtime_version = _build_engine_python(onnx_path, engine_path)
        else:
            subprocess.run(
                [
                    executable,
                    f"--onnx={onnx_path}",
                    f"--saveEngine={engine_path}",
                    "--profilingVerbosity=detailed",
                ],
                check=True,
            )
        graph["engine"] = engine_path.name
    try:
        import tensorrt_rtx as trt_rtx  # type: ignore[import-not-found]

        manifest["tensorrt_rtx_version"] = trt_rtx.__version__
    except (ImportError, AttributeError):
        manifest["tensorrt_rtx_version"] = runtime_version
    manifest_path.write_text(json.dumps(manifest, indent=2) + "\n", encoding="utf-8")


def validate_rtx4080() -> None:
    if not torch.cuda.is_available():
        raise RuntimeError("CUDA is unavailable")
    name = torch.cuda.get_device_name(0)
    capability = torch.cuda.get_device_capability(0)
    if name != RTX_GPU_NAME or capability != RTX_COMPUTE_CAPABILITY:
        raise RuntimeError(
            f"artifact requires {RTX_GPU_NAME} SM89; found {name} SM{capability[0]}{capability[1]}"
        )


def _load_artifact(artifact_dir: Path) -> dict[str, Any]:
    manifest: dict[str, Any] = json.loads(
        (artifact_dir / "artifact.json").read_text(encoding="utf-8")
    )
    if manifest.get("artifact_version") != ARTIFACT_VERSION:
        raise ValueError("unsupported RTX artifact version")
    if manifest.get("channels") != ["R", "B"]:
        raise ValueError("RTX artifact must use R/B input channels")
    telemetry = manifest.get("telemetry")
    if telemetry is None:
        # Artifact-v1 exports made before HUD support could only use zeros.
        telemetry = {"provider": "zero", "sha256": None}
        manifest["telemetry"] = telemetry
    if not isinstance(telemetry, dict) or telemetry.get("provider") not in {
        "zero",
        "hud_telemetry",
    }:
        raise ValueError("RTX artifact is missing telemetry configuration")
    if manifest.get("required_gpu") != RTX_GPU_NAME or manifest.get(
        "required_compute_capability"
    ) != list(RTX_COMPUTE_CAPABILITY):
        raise ValueError("artifact is not locked to an RTX 4080 SM89")
    for name in (
        "video_ordinary",
        "video_intermediate",
        "video_long",
        "fast_tick",
        "slow_tick",
    ):
        graph = manifest.get("graphs", {}).get(name)
        if not isinstance(graph, dict) or not graph.get("engine"):
            raise ValueError(f"artifact is missing built engine {name}")
    return manifest


def _runtime_metadata(
    cfg: ModelConfig, device: torch.device
) -> dict[str, torch.Tensor]:
    values = _metadata_inputs(cfg, torch.float16)
    return {
        name: value.to(device=device).contiguous()
        for name, value in zip(METADATA_INPUT_NAMES, values, strict=True)
    }


def _update_timing(
    metadata: dict[str, torch.Tensor],
    absolute_time: float,
    since_video: float,
    since_slow: float,
    fast_hz: int,
) -> None:
    metadata["absolute_time"].fill_(absolute_time)
    metadata["since_video_frame"].fill_(since_video)
    metadata["since_slow_update"].fill_(since_slow)
    metadata["fast_delta_time"].fill_(1.0 / fast_hz)


def _update_telemetry(
    metadata: dict[str, torch.Tensor], worker: TelemetryWorker, timestamp: float
) -> None:
    """Write one causal 5 Hz sample, then clear only the represented latches."""

    snapshot = worker.sample(timestamp)
    metadata["health"].fill_(snapshot.health)
    metadata["damage_event"].fill_(snapshot.damage_event)
    metadata["kill_event"].fill_(snapshot.kill_event)
    metadata["charge"].fill_(snapshot.charge)
    worker.acknowledge(snapshot)


class _HighResolutionTimer:
    """Windows high-resolution waitable timer with a short spin tail."""

    def __init__(self) -> None:
        self._kernel32 = None
        self._handle = None
        if os.name == "nt":
            kernel32 = ctypes.WinDLL("kernel32", use_last_error=True)
            kernel32.CreateWaitableTimerExW.restype = ctypes.c_void_p
            kernel32.SetWaitableTimerEx.argtypes = (
                ctypes.c_void_p,
                ctypes.POINTER(ctypes.c_longlong),
                ctypes.c_long,
                ctypes.c_void_p,
                ctypes.c_void_p,
                ctypes.c_void_p,
                ctypes.c_ulong,
            )
            kernel32.SetWaitableTimerEx.restype = ctypes.c_int
            kernel32.WaitForSingleObject.argtypes = (ctypes.c_void_p, ctypes.c_ulong)
            kernel32.CloseHandle.argtypes = (ctypes.c_void_p,)
            handle = kernel32.CreateWaitableTimerExW(None, None, 0x2, 0x1F0003)
            if handle:
                self._kernel32 = kernel32
                self._handle = handle

    def wait_until(self, deadline: float) -> None:
        remaining = deadline - time.perf_counter()
        if remaining > 0.0015 and self._kernel32 is not None and self._handle:
            due = ctypes.c_longlong(-max(1, int((remaining - 0.0005) * 10_000_000)))
            if self._kernel32.SetWaitableTimerEx(
                self._handle, ctypes.byref(due), 0, None, None, None, 0
            ):
                self._kernel32.WaitForSingleObject(self._handle, 0xFFFFFFFF)
        while time.perf_counter() < deadline:
            pass

    def close(self) -> None:
        if self._kernel32 is not None and self._handle:
            self._kernel32.CloseHandle(self._handle)
            self._handle = None


def _normalization_from_artifact(artifact: dict[str, Any]) -> AxisNormalization:
    data = artifact.get("axis_normalization")
    if not isinstance(data, dict):
        raise TypeError("artifact is missing frozen axis normalization")
    scales = data.get("scale_counts_per_second")
    if not isinstance(scales, list) or len(scales) != 2:
        raise RuntimeError("artifact axis normalization is invalid")
    return AxisNormalization(
        (float(scales[0]), float(scales[1])),
        float(data.get("percentile", 99.5)),
        str(data.get("method", "clipped_linear_velocity_p99_5")),
    )


def _gpu_telemetry() -> dict[str, str] | None:
    try:
        completed = subprocess.run(
            [
                "nvidia-smi",
                "--query-gpu=clocks.sm,temperature.gpu,pstate,power.draw",
                "--format=csv,noheader,nounits",
            ],
            check=True,
            capture_output=True,
            text=True,
            timeout=5,
        )
    except (OSError, subprocess.SubprocessError):
        return None
    lines = completed.stdout.splitlines()
    if not lines:
        return None
    fields = [field.strip() for field in lines[0].split(",")]
    if len(fields) != 4:
        return None
    return dict(zip(("sm_clock_mhz", "temperature_c", "pstate", "power_w"), fields))


def export_main() -> None:
    parser = argparse.ArgumentParser(description="Export an RTX 4080 TensorRT artifact")
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--skip-engine-build", action="store_true")
    parser.add_argument("--fp32-attention", action="store_true")
    parser.add_argument("--fp32-layernorm", action="store_true")
    args = parser.parse_args()
    manifest = export_rtx_artifact(
        args.checkpoint,
        args.output,
        fp32_attention=args.fp32_attention,
        fp32_layernorm=args.fp32_layernorm,
    )
    if not args.skip_engine_build:
        validate_rtx4080()
        build_engines(args.output)
    print(manifest)


def benchmark_main() -> None:
    parser = argparse.ArgumentParser(description="Benchmark RTX 4080 TensorRT engines")
    parser.add_argument("--artifact", type=Path, required=True)
    parser.add_argument("--profile", type=Path, required=True)
    parser.add_argument("--duration", type=float, default=600)
    parser.add_argument("--warmup", type=float, default=2.0)
    args = parser.parse_args()
    validate_rtx4080()
    if args.duration <= 0 or args.warmup < 0:
        raise ValueError("duration must be positive and warmup cannot be negative")
    profile = ControlProfile.from_json(args.profile)
    artifact = _load_artifact(args.artifact)
    expected_profile_hash = artifact.get("control_profile_sha256")
    if expected_profile_hash != profile.sha256():
        raise RuntimeError(
            "control profile does not match the artifact training profile"
        )
    if artifact.get("telemetry") != profile.telemetry_manifest():
        raise RuntimeError("telemetry configuration does not match the artifact")
    cfg = ModelConfig(**artifact["model_config"])
    device = torch.device("cuda:0")
    controller = Rtx4080Controller(args.artifact, device)
    metadata = _runtime_metadata(cfg, device)
    backend = create_native_backend(args.profile, cfg)
    telemetry = create_telemetry_worker(
        profile.telemetry_provider, profile.hud_telemetry
    )
    frame_host = torch.empty(
        (1, cfg.input_channels, cfg.image_height, cfg.image_width),
        dtype=torch.uint8,
        pin_memory=True,
    )
    frame_gpu = torch.empty_like(frame_host, device=device)
    latencies: list[float] = []
    engine_times: list[float] = []
    transfer_times: list[float] = []
    capture_ages: list[float] = []
    preprocessing_times: list[float] = []
    capture_stalls = 0
    fresh_frames = 0
    deadline_misses = 0
    transfer_start = torch.cuda.Event(enable_timing=True)
    transfer_end = torch.cuda.Event(enable_timing=True)
    engine_start = torch.cuda.Event(enable_timing=True)
    engine_end = torch.cuda.Event(enable_timing=True)
    torch.cuda.reset_peak_memory_stats(device)
    gpu_telemetry_start = _gpu_telemetry()
    timer = _HighResolutionTimer()
    telemetry.start()
    try:
        backend.start()
    except BaseException:
        telemetry.stop(drain=False)
        timer.close()
        raise
    try:
        initial = backend.latest_frame(timeout_ms=2000)
        if initial is None:
            raise RuntimeError("Windows Graphics Capture produced no initial frame")
        initial_frame = coerce_captured_frame(initial)
        telemetry.submit(initial_frame)
        captured_at, frame = initial_frame.timestamp, initial_frame.model_channels
        frame_host[0].copy_(frame)
        frame_gpu.copy_(frame_host, non_blocking=True)
        torch.cuda.synchronize(device)
        start = time.perf_counter()
        total_ticks = int((args.warmup + args.duration) * cfg.fast_hz)
        warmup_ticks = int(args.warmup * cfg.fast_hz)
        last_video = start
        last_slow = start
        for tick in range(total_ticks):
            deadline = start + tick / cfg.fast_hz
            timer.wait_until(deadline)
            if backend.emergency_stop_requested() or not backend.target_active():
                raise RuntimeError("benchmark stopped because focus or target was lost")
            tick_start = time.perf_counter()
            failure_duration = (
                None
                if profile.hud_telemetry is None
                else profile.hud_telemetry.failure_termination_seconds
            )
            if telemetry.should_terminate(tick_start, failure_duration):
                raise RuntimeError("benchmark stopped after sustained telemetry failure")
            video_frame: torch.Tensor | None = None
            latest: Any = None
            if tick % cfg.fast_ticks_per_video == 0:
                latest = initial_frame if tick == 0 else backend.latest_frame(timeout_ms=0)
                if latest is None:
                    capture_stalls += 1
                else:
                    captured_frame = coerce_captured_frame(latest)
                    captured_at, frame = (
                        captured_frame.timestamp,
                        captured_frame.model_channels,
                    )
                    if tick != 0:
                        telemetry.submit(captured_frame)
                    frame_host[0].copy_(frame)
                    transfer_start.record()
                    frame_gpu.copy_(frame_host, non_blocking=True)
                    transfer_end.record()
                    fresh_frames += 1
                video_frame = frame_gpu
                last_video = tick_start
            _update_timing(
                metadata,
                tick_start - start,
                tick_start - last_video,
                tick_start - last_slow,
                cfg.fast_hz,
            )
            if tick % cfg.fast_ticks_per_slow == 0:
                _update_telemetry(metadata, telemetry, tick_start)
            engine_start.record()
            axes, movement, buttons = controller.step(tick, metadata, video_frame)
            metadata["executed_axes"].copy_(axes)
            if movement is not None and buttons is not None:
                metadata["movement"].copy_(movement.argmax(dim=-1))
                metadata["buttons"].copy_(buttons >= 0)
                last_slow = tick_start
            engine_end.record()
            torch.cuda.synchronize(device)
            elapsed = (time.perf_counter() - tick_start) * 1000.0
            if tick >= warmup_ticks:
                latencies.append(elapsed)
                engine_times.append(engine_start.elapsed_time(engine_end))
                if video_frame is not None and latest is not None:
                    transfer_times.append(transfer_start.elapsed_time(transfer_end))
                    capture_ages.append(max(0.0, (tick_start - captured_at) * 1000.0))
                    diagnostics = getattr(backend, "capture_diagnostics", dict)()
                    preprocess = diagnostics.get("preprocessing_ms")
                    if isinstance(preprocess, (int, float)):
                        preprocessing_times.append(float(preprocess))
                if elapsed > 1000.0 / cfg.fast_hz:
                    deadline_misses += 1
    finally:
        try:
            backend.stop()
        finally:
            telemetry.stop()
            timer.close()
    values = torch.tensor(latencies, dtype=torch.float64)
    report = {
        "gpu": torch.cuda.get_device_name(device),
        "tensorrt_rtx_version": artifact.get("tensorrt_rtx_version"),
        "duration_seconds": args.duration,
        "ticks": len(latencies),
        "latency_ms": {
            "p50": torch.quantile(values, 0.50).item(),
            "p95": torch.quantile(values, 0.95).item(),
            "p99": torch.quantile(values, 0.99).item(),
            "maximum": values.max().item(),
        },
        "engine_ms_p99": torch.quantile(torch.tensor(engine_times), 0.99).item(),
        "transfer_ms_p99": (
            torch.quantile(torch.tensor(transfer_times), 0.99).item()
            if transfer_times
            else None
        ),
        "wgc_age_ms_p99": (
            torch.quantile(torch.tensor(capture_ages), 0.99).item()
            if capture_ages
            else None
        ),
        "preprocessing_ms_p99": (
            torch.quantile(torch.tensor(preprocessing_times), 0.99).item()
            if preprocessing_times
            else None
        ),
        "fresh_frames": fresh_frames,
        "capture_stalls": capture_stalls,
        "deadline_misses": deadline_misses,
        "peak_inference_vram_bytes": torch.cuda.max_memory_allocated(device),
        "gpu_telemetry_start": gpu_telemetry_start,
        "gpu_telemetry_end": _gpu_telemetry(),
        "hud_telemetry": telemetry.diagnostics(),
    }
    print(json.dumps(report, indent=2))
    if (
        report["latency_ms"]["p99"] > 13.33
        or report["latency_ms"]["maximum"] >= 1000.0 / cfg.fast_hz
        or deadline_misses
        or capture_stalls
        or report["peak_inference_vram_bytes"] >= 4 * 1024**3
    ):
        raise RuntimeError("RTX 4080 benchmark failed an acceptance gate")


def run_main() -> None:
    parser = argparse.ArgumentParser(description="Run native RTX 4080 inference")
    parser.add_argument("--artifact", type=Path, required=True)
    parser.add_argument("--profile", type=Path, required=True)
    args = parser.parse_args()
    validate_rtx4080()
    profile = ControlProfile.from_json(args.profile)
    artifact = _load_artifact(args.artifact)
    if artifact.get("control_profile_sha256") != profile.sha256():
        raise RuntimeError(
            "control profile does not match the artifact training profile"
        )
    if artifact.get("telemetry") != profile.telemetry_manifest():
        raise RuntimeError("telemetry configuration does not match the artifact")
    normalization = _normalization_from_artifact(artifact)
    cfg = ModelConfig(**artifact["model_config"])
    device = torch.device("cuda:0")
    controller = Rtx4080Controller(args.artifact, device)
    metadata = _runtime_metadata(cfg, device)
    denormalizer = AxisDenormalizer(normalization, profile.invert_axes)
    backend = create_native_backend(args.profile, cfg)
    telemetry = create_telemetry_worker(
        profile.telemetry_provider, profile.hud_telemetry
    )
    if not all(
        hasattr(backend, method)
        for method in ("apply_relative_mouse", "apply_discrete", "release_all")
    ):
        raise RuntimeError("capture backend does not provide SendInput control output")
    frame_host = torch.empty(
        (1, cfg.input_channels, cfg.image_height, cfg.image_width),
        dtype=torch.uint8,
        pin_memory=True,
    )
    frame_gpu = torch.empty_like(frame_host, device=device)
    axes_host = torch.empty((1, 2), dtype=torch.float16, pin_memory=True)
    timer = _HighResolutionTimer()
    telemetry.start()
    try:
        backend.start()
    except BaseException:
        telemetry.stop(drain=False)
        timer.close()
        raise
    start = time.perf_counter()
    last_video = start
    last_slow = start
    try:
        initial = backend.latest_frame(timeout_ms=2000)
        if initial is None:
            raise RuntimeError("Windows Graphics Capture produced no initial frame")
        initial_frame = coerce_captured_frame(initial)
        telemetry.submit(initial_frame)
        first_frame = initial_frame.model_channels
        frame_host[0].copy_(first_frame)
        frame_gpu.copy_(frame_host, non_blocking=True)
        torch.cuda.synchronize(device)
        tick = 0
        while True:
            timer.wait_until(start + tick / cfg.fast_hz)
            now = time.perf_counter()
            held = backend.held_inputs()
            if (
                backend.emergency_stop_requested()
                or not backend.target_active()
                or profile.pause_key in held
            ):
                break
            failure_duration = (
                None
                if profile.hud_telemetry is None
                else profile.hud_telemetry.failure_termination_seconds
            )
            if telemetry.should_terminate(now, failure_duration):
                break
            video_frame: torch.Tensor | None = None
            if tick % cfg.fast_ticks_per_video == 0:
                latest = (
                    initial_frame
                    if tick == 0
                    else backend.latest_frame(timeout_ms=0)
                )
                if latest is None:
                    break
                captured_frame = coerce_captured_frame(latest)
                frame = captured_frame.model_channels
                if tick != 0:
                    telemetry.submit(captured_frame)
                frame_host[0].copy_(frame)
                frame_gpu.copy_(frame_host, non_blocking=True)
                video_frame = frame_gpu
                last_video = now
            _update_timing(
                metadata,
                now - start,
                now - last_video,
                now - last_slow,
                cfg.fast_hz,
            )
            if tick % cfg.fast_ticks_per_slow == 0:
                _update_telemetry(metadata, telemetry, now)
            axes, movement, buttons = controller.step(tick, metadata, video_frame)
            axes_host.copy_(axes, non_blocking=True)
            torch.cuda.synchronize(device)
            axis_values = axes_host[0].float()
            delta = denormalizer.convert(axis_values, 1.0 / cfg.fast_hz)
            backend.apply_relative_mouse(*delta)  # type: ignore[attr-defined]
            metadata["executed_axes"].copy_(axes)
            if movement is not None and buttons is not None:
                discrete = tuple(
                    int(value) for value in movement.argmax(dim=-1)[0].tolist()
                )
                button_states = tuple(
                    bool(value) for value in (buttons[0] >= 0).tolist()
                )
                backend.apply_discrete(discrete, button_states)  # type: ignore[attr-defined]
                metadata["movement"].copy_(movement.argmax(dim=-1))
                metadata["buttons"].copy_(buttons >= 0)
                last_slow = now
            tick += 1
    finally:
        try:
            backend.release_all()  # type: ignore[attr-defined]
        finally:
            try:
                backend.stop()
            finally:
                telemetry.stop()
                timer.close()
