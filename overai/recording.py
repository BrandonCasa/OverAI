"""Windows demonstration recording, control profiles, and dataset finalization."""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import os
import queue
import re
import shutil
import threading
import time
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any, Protocol

import torch
from torchvision.io import write_jpeg

from .config import ModelConfig
from .data import CONTROL_KEYS


_ENCODER_STALL_BUDGET_SECONDS = 2.0
_CAPTURE_REUSE_BUDGET_SECONDS = 0.25
_TIMING_GAP_BUDGET_SECONDS = 0.25
_RECORDED_CONTROL_KEYS = {*CONTROL_KEYS, "raw_mouse_deltas", "fast_durations"}


@dataclass(frozen=True, slots=True)
class WindowMatch:
    process_name: str
    title_regex: str


@dataclass(frozen=True, slots=True)
class ControlProfile:
    window: WindowMatch
    movement: dict[str, str]
    buttons: tuple[str, ...]
    pause_key: str
    emergency_stop_key: str
    invert_axes: tuple[bool, bool] = (False, False)

    @classmethod
    def from_json(cls, path: str | Path) -> ControlProfile:
        payload: Any = json.loads(Path(path).read_text(encoding="utf-8"))
        if not isinstance(payload, dict):
            raise TypeError("control profile must be a JSON object")
        supported_fields = {
            "window",
            "movement",
            "buttons",
            "pause_key",
            "emergency_stop_key",
            "invert_axes",
        }
        unsupported_fields = sorted(set(payload) - supported_fields)
        if unsupported_fields:
            raise ValueError(
                "control profile contains unsupported fields: "
                + ", ".join(unsupported_fields)
            )
        window = payload.get("window")
        if not isinstance(window, dict):
            raise TypeError("control profile window must be an object")
        process_name = window.get("process_name")
        title_regex = window.get("title_regex")
        if not isinstance(process_name, str) or not process_name:
            raise ValueError("window.process_name must be a non-empty string")
        if not isinstance(title_regex, str) or not title_regex:
            raise ValueError("window.title_regex must be a non-empty string")
        re.compile(title_regex)
        movement = payload.get("movement")
        required_directions = {"left", "right", "forward", "reverse"}
        if not isinstance(movement, dict) or set(movement) != required_directions:
            raise ValueError(
                "movement must define exactly left, right, forward, and reverse"
            )
        if any(not isinstance(value, str) or not value for value in movement.values()):
            raise ValueError("movement bindings must be non-empty strings")
        buttons = payload.get("buttons")
        if (
            not isinstance(buttons, list)
            or not buttons
            or any(not isinstance(value, str) or not value for value in buttons)
        ):
            raise ValueError("buttons must contain at least one input name")
        invert_axes = payload.get("invert_axes", [False, False])
        if (
            not isinstance(invert_axes, list)
            or len(invert_axes) != 2
            or any(not isinstance(value, bool) for value in invert_axes)
        ):
            raise ValueError("invert_axes must contain two booleans")
        pause = payload.get("pause_key")
        emergency = payload.get("emergency_stop_key")
        if not isinstance(pause, str) or not pause:
            raise ValueError("pause_key must be a non-empty string")
        if not isinstance(emergency, str) or not emergency:
            raise ValueError("emergency_stop_key must be a non-empty string")
        return cls(
            window=WindowMatch(process_name, title_regex),
            movement={str(key): str(value) for key, value in movement.items()},
            buttons=tuple(buttons),
            pause_key=pause,
            emergency_stop_key=emergency,
            invert_axes=(invert_axes[0], invert_axes[1]),
        )

    def sha256(self) -> str:
        canonical = json.dumps(asdict(self), sort_keys=True, separators=(",", ":"))
        return hashlib.sha256(canonical.encode("utf-8")).hexdigest()

@dataclass(frozen=True, slots=True)
class AxisNormalization:
    scale_counts_per_second: tuple[float, float]
    percentile: float = 99.5
    method: str = "clipped_linear_velocity_p99_5"

    @classmethod
    def derive(
        cls, raw_mouse_deltas: torch.Tensor, fast_durations: torch.Tensor
    ) -> AxisNormalization:
        if raw_mouse_deltas.ndim != 2 or raw_mouse_deltas.shape[1] != 2:
            raise ValueError("raw mouse deltas must have shape [T, 2]")
        if fast_durations.shape != (raw_mouse_deltas.shape[0],):
            raise ValueError("fast durations must have shape [T]")
        if not torch.all(fast_durations > 0):
            raise ValueError("fast durations must be positive")
        velocities = raw_mouse_deltas.float() / fast_durations[:, None]
        scales: list[float] = []
        for axis in range(2):
            nonzero = velocities[:, axis].abs()
            nonzero = nonzero[nonzero > 0]
            scale = (
                float(torch.quantile(nonzero, 0.995)) if nonzero.numel() else 1.0
            )
            scales.append(max(scale, 1.0))
        return cls((scales[0], scales[1]))

    @classmethod
    def from_manifest(cls, payload: Any) -> AxisNormalization:
        if not isinstance(payload, dict):
            raise TypeError("axis_normalization must be an object")
        method = payload.get("method")
        percentile = payload.get("percentile")
        scales = payload.get("scale_counts_per_second")
        if method != "clipped_linear_velocity_p99_5":
            raise ValueError(f"unsupported axis normalization method: {method!r}")
        if not isinstance(percentile, (int, float)) or float(percentile) != 99.5:
            raise ValueError("axis normalization percentile must be 99.5")
        if (
            not isinstance(scales, list)
            or len(scales) != 2
            or any(not isinstance(value, (int, float)) or value <= 0 for value in scales)
        ):
            raise ValueError("axis normalization scales must contain two positive numbers")
        return cls((float(scales[0]), float(scales[1])), float(percentile), method)

    def normalize(
        self, raw_mouse_deltas: torch.Tensor, fast_durations: torch.Tensor
    ) -> torch.Tensor:
        scales = torch.tensor(self.scale_counts_per_second, dtype=torch.float32)
        velocity = raw_mouse_deltas.float() / fast_durations[:, None]
        return (velocity / scales).clamp(-1.0, 1.0)

    def to_manifest(self) -> dict[str, Any]:
        return {
            "method": self.method,
            "percentile": self.percentile,
            "scale_counts_per_second": list(self.scale_counts_per_second),
        }


class AxisDenormalizer:
    """Turn normalized model axes into integer relative mouse counts."""

    def __init__(
        self,
        normalization: AxisNormalization,
        invert_axes: tuple[bool, bool] = (False, False),
    ) -> None:
        self.normalization = normalization
        self.signs = (-1.0 if invert_axes[0] else 1.0, -1.0 if invert_axes[1] else 1.0)
        self.residual = [0.0, 0.0]

    def convert(self, axes: torch.Tensor, duration_seconds: float) -> tuple[int, int]:
        if tuple(axes.shape) != (2,):
            raise ValueError("axes must have shape [2]")
        result: list[int] = []
        for index in range(2):
            value = (
                float(axes[index].clamp(-1, 1))
                * self.normalization.scale_counts_per_second[index]
                * duration_seconds
                * self.signs[index]
                + self.residual[index]
            )
            integer = math.trunc(value)
            self.residual[index] = value - integer
            result.append(integer)
        return result[0], result[1]


class NativeRecorderBackend(Protocol):
    """Interface supplied by the optional native WGC/Raw Input extension."""

    def start(self) -> None: ...
    def stop(self) -> None: ...
    def latest_frame(self, timeout_ms: int) -> Any: ...
    def drain_mouse_deltas(self, start: float, end: float) -> tuple[int, int]: ...
    def held_inputs(self) -> set[str]: ...
    def target_active(self) -> bool: ...
    def emergency_stop_requested(self) -> bool: ...


def create_native_backend(
    profile_path: Path, cfg: ModelConfig
) -> NativeRecorderBackend:
    try:
        from overai_native import WindowsCaptureInput  # type: ignore[import-not-found]
    except ImportError:
        from .windows_backend import WindowsCaptureBackend

        return WindowsCaptureBackend(profile_path, cfg)
    return WindowsCaptureInput(str(profile_path), cfg.image_width, cfg.image_height)


@dataclass(slots=True)
class _EncodedFrame:
    index: int
    channels: torch.Tensor


class EpisodeRecorder:
    def __init__(
        self,
        backend: NativeRecorderBackend,
        profile: ControlProfile,
        cfg: ModelConfig,
        output_dir: Path,
        split: str,
        episode_id: str,
    ) -> None:
        if split not in {"train", "validation"}:
            raise ValueError("split must be train or validation")
        if not episode_id or any(char in episode_id for char in '<>:"/\\|?*'):
            raise ValueError("episode id is empty or contains invalid filename characters")
        self.backend = backend
        self.profile = profile
        self.cfg = cfg
        self.output_dir = output_dir
        self.split = split
        self.episode_id = episode_id
        if len(profile.buttons) != cfg.num_buttons:
            raise ValueError(
                "control profile button count does not match model config: "
                f"{len(profile.buttons)} != {cfg.num_buttons}"
            )

    @staticmethod
    def _movement(held: set[str], bindings: dict[str, str]) -> tuple[int, int]:
        left = bindings["left"] in held
        right = bindings["right"] in held
        forward = bindings["forward"] in held
        reverse = bindings["reverse"] in held
        x = 1 if left == right else (0 if left else 2)
        y = 1 if reverse == forward else (0 if reverse else 2)
        return x, y

    def _emergency_stop_reason(self) -> str:
        reason_provider = getattr(self.backend, "emergency_stop_reason", None)
        if callable(reason_provider):
            reason = reason_provider()
            if isinstance(reason, str) and reason:
                return reason
        return "emergency_stop"

    def _capture_timeout_reason(self) -> str:
        reason_provider = getattr(self.backend, "capture_interruption_reason", None)
        if callable(reason_provider):
            reason = reason_provider()
            if isinstance(reason, str) and reason:
                return reason
        return "capture_frame_timeout"

    def record(self, duration_seconds: float | None) -> Path:
        final_dir = self.output_dir / "episodes" / self.episode_id
        temporary_dir = final_dir.with_name(final_dir.name + ".recording")
        if final_dir.exists() or temporary_dir.exists():
            raise FileExistsError(f"episode already exists: {final_dir}")
        red_dir = temporary_dir / "frames_r"
        blue_dir = temporary_dir / "frames_b"
        red_dir.mkdir(parents=True)
        blue_dir.mkdir(parents=True)
        # Keep capture cadence independent from ordinary filesystem/antivirus
        # stalls. At 720p, two seconds is still bounded to about 110 MiB while
        # covering much longer stalls than the old fixed 16-frame queue.
        encoder_queue_capacity = max(
            16, math.ceil(self.cfg.video_hz * _ENCODER_STALL_BUDGET_SECONDS)
        )
        encode_queue: queue.Queue[_EncodedFrame | None] = queue.Queue(
            maxsize=encoder_queue_capacity
        )
        encode_error: list[BaseException] = []
        encoder_timeout_seconds = 30.0

        def encode_worker() -> None:
            try:
                while True:
                    item = encode_queue.get()
                    if item is None:
                        return
                    write_jpeg(
                        item.channels[0:1].contiguous(),
                        str(red_dir / f"{item.index:06d}.jpg"),
                        quality=95,
                    )
                    write_jpeg(
                        item.channels[1:2].contiguous(),
                        str(blue_dir / f"{item.index:06d}.jpg"),
                        quality=95,
                    )
            except BaseException as error:  # propagate worker failures to recorder
                encode_error.append(error)

        worker = threading.Thread(target=encode_worker, name="overai-jpeg", daemon=True)
        worker.start()

        def enqueue_frame(item: _EncodedFrame) -> None:
            deadline = time.perf_counter() + encoder_timeout_seconds
            while True:
                if encode_error or not worker.is_alive():
                    cause = encode_error[0] if encode_error else None
                    raise RuntimeError("JPEG encoder failed") from cause
                try:
                    encode_queue.put(item, timeout=0.05)
                    return
                except queue.Full:
                    if time.perf_counter() >= deadline:
                        raise RuntimeError("JPEG encoder stopped consuming frames")

        def finish_encoder() -> None:
            deadline = time.perf_counter() + encoder_timeout_seconds
            sentinel_sent = False
            while worker.is_alive() and time.perf_counter() < deadline:
                try:
                    encode_queue.put(None, timeout=0.05)
                    sentinel_sent = True
                    break
                except queue.Full:
                    continue
            if sentinel_sent:
                worker.join(timeout=max(0.0, deadline - time.perf_counter()))
            else:
                worker.join(timeout=0)
            if worker.is_alive():
                encode_error.append(
                    RuntimeError("JPEG encoder did not shut down within 30 seconds")
                )
        fast_deltas: list[tuple[int, int]] = []
        fast_durations: list[float] = []
        fast_timestamps: list[float] = []
        frame_timestamps: list[float] = []
        slow_timestamps: list[float] = []
        movement: list[tuple[int, int]] = []
        buttons: list[list[int]] = []
        try:
            self.backend.start()
        except BaseException:
            finish_encoder()
            shutil.rmtree(temporary_dir, ignore_errors=True)
            raise
        maximum_ticks = (
            None
            if duration_seconds is None
            else max(0, math.ceil(duration_seconds * self.cfg.fast_hz))
        )
        initial_frame: tuple[float, torch.Tensor] | None = None
        if maximum_ticks is None or maximum_ticks > 0:
            try:
                captured = self.backend.latest_frame(timeout_ms=2000)
                if captured is None:
                    raise RuntimeError("no fresh Windows Graphics Capture frame")
                initial_frame = captured
            except BaseException:
                try:
                    self.backend.stop()
                finally:
                    finish_encoder()
                    shutil.rmtree(temporary_dir, ignore_errors=True)
                raise
            # The initial frame establishes readiness, not elapsed recording time.
            # Treat it as the observation at t=0 after all startup work is complete.
            start = time.perf_counter()
        else:
            start = time.perf_counter()
        previous_fast = start
        schedule_start = start
        frame_index = 0
        fast_index = 0
        capture_timeout_ms = max(1, math.ceil(1000 / self.cfg.video_hz))
        maximum_reused_frames = max(
            1, math.ceil(self.cfg.video_hz * _CAPTURE_REUSE_BUDGET_SECONDS)
        )
        consecutive_reused_frames = 0
        reused_video_frames = 0
        last_frame: torch.Tensor | None = None
        failure: BaseException | None = None
        termination_reason: str | None = None
        try:
            while maximum_ticks is None or fast_index < maximum_ticks:
                if self.backend.emergency_stop_requested():
                    termination_reason = self._emergency_stop_reason()
                    break
                deadline = schedule_start + fast_index / self.cfg.fast_hz
                remaining = deadline - time.perf_counter()
                if remaining > 0:
                    time.sleep(remaining)
                now = time.perf_counter()
                if not self.backend.target_active():
                    termination_reason = "target_focus_lost"
                    break
                held = self.backend.held_inputs()
                if self.profile.pause_key in held:
                    termination_reason = "pause_key"
                    break
                lateness = now - deadline
                if lateness > _TIMING_GAP_BUDGET_SECONDS:
                    termination_reason = "timing_gap"
                    break
                if lateness > 2.0 / self.cfg.fast_hz:
                    # Preserve the real elapsed timestamps/duration, but move
                    # future deadlines forward so a brief OS stall does not
                    # cause a burst of artificial catch-up samples.
                    schedule_start += lateness
                fast_timestamps.append(now - start)
                fast_durations.append(max(now - previous_fast, 1.0 / self.cfg.fast_hz))
                fast_deltas.append(self.backend.drain_mouse_deltas(previous_fast, now))
                previous_fast = now
                if fast_index % self.cfg.fast_ticks_per_video == 0:
                    # Startup capture happens before the recording clock begins.
                    # Give later samples one video interval to absorb normal WGC
                    # callback jitter. A zero-timeout poll can miss a frame that
                    # arrives moments later and used to silently truncate an episode.
                    captured_frame = initial_frame if fast_index == 0 else None
                    if captured_frame is None:
                        captured = self.backend.latest_frame(
                            timeout_ms=capture_timeout_ms
                        )
                        if captured is None:
                            if self.backend.emergency_stop_requested():
                                termination_reason = self._emergency_stop_reason()
                                break
                            consecutive_reused_frames += 1
                            if consecutive_reused_frames > maximum_reused_frames:
                                termination_reason = self._capture_timeout_reason()
                                break
                            if last_frame is None:
                                raise RuntimeError(
                                    "Windows Graphics Capture lost its initial frame"
                                )
                            captured_frame = (time.perf_counter(), last_frame)
                            reused_video_frames += 1
                        else:
                            captured_frame = captured
                            consecutive_reused_frames = 0
                    _captured_at, frame = captured_frame
                    if tuple(frame.shape) != (
                        self.cfg.input_channels,
                        self.cfg.image_height,
                        self.cfg.image_width,
                    ) or frame.dtype != torch.uint8:
                        raise ValueError("native capture returned an invalid RB frame")
                    last_frame = frame
                    enqueue_frame(_EncodedFrame(frame_index, frame.cpu()))
                    # Frames are observations consumed at this scheduled video
                    # tick. A WGC callback may carry an older capture timestamp
                    # after a reused-frame timeout; using the causal sample time
                    # keeps the observation stream strictly ordered.
                    frame_timestamps.append(now - start)
                    frame_index += 1
                if fast_index % self.cfg.fast_ticks_per_slow == 0:
                    movement.append(self._movement(held, self.profile.movement))
                    buttons.append([int(binding in held) for binding in self.profile.buttons])
                    slow_timestamps.append(now - start)
                fast_index += 1
                if encode_error:
                    raise RuntimeError("JPEG encoder failed") from encode_error[0]
            if termination_reason is None:
                termination_reason = "duration_complete"
        except BaseException as error:
            failure = error
        finally:
            try:
                self.backend.stop()
            finally:
                finish_encoder()
        if failure is not None:
            shutil.rmtree(temporary_dir, ignore_errors=True)
            raise failure
        if encode_error:
            shutil.rmtree(temporary_dir, ignore_errors=True)
            raise RuntimeError("JPEG encoder failed") from encode_error[0]
        if not fast_timestamps:
            shutil.rmtree(temporary_dir, ignore_errors=True)
            raise RuntimeError("recording produced no samples")
        controls = {
            "fast_timestamps": torch.tensor(fast_timestamps, dtype=torch.float64),
            "frame_timestamps": torch.tensor(frame_timestamps, dtype=torch.float64),
            "slow_timestamps": torch.tensor(slow_timestamps, dtype=torch.float64),
            "raw_mouse_deltas": torch.tensor(fast_deltas, dtype=torch.int32),
            "fast_durations": torch.tensor(fast_durations, dtype=torch.float32),
            "axes": torch.zeros(len(fast_deltas), 2),
            "movement": torch.tensor(movement, dtype=torch.long),
            "buttons": torch.tensor(buttons, dtype=torch.uint8),
        }
        torch.save(controls, temporary_dir / "controls.pt")
        metadata = {
            "id": self.episode_id,
            "split": self.split,
            "profile_sha256": self.profile.sha256(),
            "termination_reason": termination_reason,
            "duration_seconds": float(fast_timestamps[-1]),
            "reused_video_frames": reused_video_frames,
            "finalized": False,
        }
        (temporary_dir / "episode.json").write_text(
            json.dumps(metadata, indent=2) + "\n", encoding="utf-8"
        )
        temporary_dir.replace(final_dir)
        return final_dir


def _episode_entry(root: Path, episode_dir: Path) -> dict[str, str]:
    relative = episode_dir.relative_to(root).as_posix()
    return {
        "id": episode_dir.name,
        "red_frames": f"{relative}/frames_r",
        "blue_frames": f"{relative}/frames_b",
        "controls": f"{relative}/controls.pt",
    }


def _validate_episode(
    episode: Path, expected_num_buttons: int | None = None
) -> tuple[torch.Tensor | None, torch.Tensor | None]:
    """Validate one closed segment before it can enter a manifest."""

    if episode.name.endswith(".recording"):
        raise ValueError(f"unfinished temporary episode cannot be finalized: {episode}")
    red = sorted((episode / "frames_r").glob("*.jpg"))
    blue = sorted((episode / "frames_b").glob("*.jpg"))
    if not red or [path.name for path in red] != [path.name for path in blue]:
        raise ValueError(f"{episode}: R/B JPEG pairs are missing or misaligned")
    controls = torch.load(
        episode / "controls.pt", map_location="cpu", weights_only=True
    )
    control_keys = set(controls)
    is_recorded = control_keys == _RECORDED_CONTROL_KEYS
    is_finalized = control_keys == set(CONTROL_KEYS)
    if not is_recorded and not is_finalized:
        raise ValueError(f"{episode}: controls keys do not match dataset format 3")
    fast_count = int(controls["fast_timestamps"].shape[0])
    slow_count = int(controls["slow_timestamps"].shape[0])
    if fast_count <= 0 or len(red) != int(controls["frame_timestamps"].shape[0]):
        raise ValueError(f"{episode}: frame/control counts are inconsistent")
    if controls["axes"].shape != (fast_count, 2):
        raise ValueError(f"{episode}: axes must have shape [T_fast, 2]")
    if is_recorded:
        if controls["raw_mouse_deltas"].shape != (fast_count, 2):
            raise ValueError(f"{episode}: raw mouse deltas must have shape [T_fast, 2]")
        if controls["fast_durations"].shape != (fast_count,):
            raise ValueError(f"{episode}: fast durations must have shape [T_fast]")
    if controls["movement"].shape != (slow_count, 2):
        raise ValueError(f"{episode}: movement must have shape [T_slow, 2]")
    buttons = controls["buttons"]
    if buttons.ndim != 2 or buttons.shape[0] != slow_count or buttons.shape[1] <= 0:
        raise ValueError(f"{episode}: buttons must have shape [T_slow, num_buttons]")
    if expected_num_buttons is not None and buttons.shape[1] != expected_num_buttons:
        raise ValueError(
            f"{episode}: button count {buttons.shape[1]} does not match "
            f"dataset button count {expected_num_buttons}"
        )
    for name in ("fast_timestamps", "frame_timestamps", "slow_timestamps"):
        values = controls[name]
        if not torch.isfinite(values).all() or (
            values.numel() > 1 and not torch.all(values[1:] > values[:-1])
        ):
            raise ValueError(f"{episode}: {name} must be finite and strictly monotonic")
    if not torch.isfinite(controls["axes"]).all():
        raise ValueError(f"{episode}: axes must be finite")
    if is_finalized and controls["axes"].abs().max() > 1.0001:
        raise ValueError(f"{episode}: finalized axes must be normalized to [-1, 1]")
    if is_recorded and (
        not torch.isfinite(controls["fast_durations"]).all()
        or not torch.all(controls["fast_durations"] > 0)
    ):
        raise ValueError(f"{episode}: fast durations must be finite and positive")
    if not torch.all((controls["movement"] >= 0) & (controls["movement"] <= 2)):
        raise ValueError(f"{episode}: movement contains a non-class value")
    if not torch.all((controls["buttons"] == 0) | (controls["buttons"] == 1)):
        raise ValueError(f"{episode}: buttons must be binary")
    if is_recorded:
        return controls["raw_mouse_deltas"], controls["fast_durations"]
    return None, None


def _existing_manifest(root: Path, split: str) -> dict[str, Any] | None:
    path = root / f"{split}.json"
    if not path.is_file():
        return None
    payload: Any = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(payload, dict) or payload.get("split") != split:
        raise ValueError(f"{path}: existing manifest split does not match {split}")
    return payload


def finalize_dataset(train_path: Path, validation_path: Path) -> AxisNormalization:
    roots = {train_path.resolve(), validation_path.resolve()}
    if len(roots) != 2:
        raise ValueError("training and validation outputs must be different directories")
    train_episodes = sorted(
        path for path in (train_path / "episodes").glob("*") if path.is_dir()
    )
    validation_episodes = sorted(
        path for path in (validation_path / "episodes").glob("*") if path.is_dir()
    )
    if not train_episodes or not validation_episodes:
        raise ValueError("both training and validation must contain episodes")
    duplicate_ids = {path.name for path in train_episodes} & {
        path.name for path in validation_episodes
    }
    if duplicate_ids:
        raise ValueError(f"episode ids overlap across splits: {sorted(duplicate_ids)}")
    raw_parts: list[torch.Tensor] = []
    duration_parts: list[torch.Tensor] = []
    num_buttons: int | None = None
    for episode in train_episodes:
        raw, durations = _validate_episode(episode, num_buttons)
        if num_buttons is None:
            controls = torch.load(
                episode / "controls.pt", map_location="cpu", weights_only=True
            )
            num_buttons = int(controls["buttons"].shape[1])
        if raw is not None and durations is not None:
            raw_parts.append(raw)
            duration_parts.append(durations)
    for episode in validation_episodes:
        _validate_episode(episode, num_buttons)
    train_manifest = _existing_manifest(train_path, "train")
    validation_manifest = _existing_manifest(validation_path, "validation")
    if validation_manifest is not None and train_manifest is None:
        raise ValueError("validation manifest exists without a training manifest")
    if train_manifest is not None:
        normalization = AxisNormalization.from_manifest(
            train_manifest.get("axis_normalization")
        )
        if validation_manifest is not None:
            validation_normalization = AxisNormalization.from_manifest(
                validation_manifest.get("axis_normalization")
            )
            if validation_normalization != normalization:
                raise ValueError("existing train and validation normalization differ")
    else:
        if len(raw_parts) != len(train_episodes):
            raise ValueError(
                "finalized training episodes require an existing training manifest"
            )
        normalization = AxisNormalization.derive(
            torch.cat(raw_parts), torch.cat(duration_parts)
        )
    profile_hashes = {
        str(
            json.loads((episode / "episode.json").read_text(encoding="utf-8")).get(
                "profile_sha256"
            )
        )
        for episode in (*train_episodes, *validation_episodes)
    }
    if len(profile_hashes) != 1:
        raise ValueError("all train and validation episodes must use one control profile")
    profile_hash = next(iter(profile_hashes))
    for existing in (train_manifest, validation_manifest):
        if existing is None:
            continue
        if existing.get("num_buttons") != num_buttons:
            raise ValueError("existing manifest button count does not match episodes")
        if existing.get("control_profile_sha256") != profile_hash:
            raise ValueError("existing manifest control profile does not match episodes")
    for root, split, episodes in (
        (train_path, "train", train_episodes),
        (validation_path, "validation", validation_episodes),
    ):
        entries: list[dict[str, str]] = []
        for episode in episodes:
            metadata = json.loads((episode / "episode.json").read_text(encoding="utf-8"))
            if metadata.get("split") != split:
                raise ValueError(f"{episode}: split metadata does not match {split}")
            controls_path = episode / "controls.pt"
            controls = torch.load(controls_path, map_location="cpu", weights_only=True)
            if set(controls) == _RECORDED_CONTROL_KEYS:
                controls["axes"] = normalization.normalize(
                    controls["raw_mouse_deltas"], controls["fast_durations"]
                )
                finalized_controls = {key: controls[key] for key in CONTROL_KEYS}
                temporary = controls_path.with_suffix(".pt.tmp")
                torch.save(finalized_controls, temporary)
                os.replace(temporary, controls_path)
            if not metadata.get("finalized"):
                metadata["finalized"] = True
                metadata_temporary = episode / "episode.json.tmp"
                metadata_temporary.write_text(
                    json.dumps(metadata, indent=2) + "\n", encoding="utf-8"
                )
                os.replace(metadata_temporary, episode / "episode.json")
            entries.append(_episode_entry(root, episode))
        manifest = {
            "version": 3,
            "split": split,
            "channels": ["R", "B"],
            "axis_normalization": normalization.to_manifest(),
            "num_buttons": num_buttons,
            "control_profile_sha256": profile_hash,
            "episodes": entries,
        }
        temporary_manifest = root / f"{split}.json.tmp"
        temporary_manifest.write_text(
            json.dumps(manifest, indent=2) + "\n", encoding="utf-8"
        )
        os.replace(temporary_manifest, root / f"{split}.json")
    return normalization


def record_main() -> None:
    parser = argparse.ArgumentParser(description="Record a Windows OverAI episode")
    parser.add_argument("--profile", type=Path, required=True)
    parser.add_argument("--split", choices=("train", "validation"), required=True)
    parser.add_argument("--episode-id", required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--duration", type=float)
    parser.add_argument("--model-config", type=Path, default=Path("configs/h100_720p.json"))
    args = parser.parse_args()
    profile = ControlProfile.from_json(args.profile)
    cfg = ModelConfig.from_json(args.model_config)
    backend = create_native_backend(args.profile, cfg)
    recorder = EpisodeRecorder(
        backend, profile, cfg, args.output, args.split, args.episode_id
    )
    episode = recorder.record(args.duration)
    metadata = json.loads((episode / "episode.json").read_text(encoding="utf-8"))
    print(
        "recording stopped: "
        f"reason={metadata['termination_reason']} "
        f"duration={metadata['duration_seconds']:.3f}s "
        f"reused_frames={metadata['reused_video_frames']}"
    )
    print(episode)


def finalize_main() -> None:
    parser = argparse.ArgumentParser(description="Finalize OverAI train/validation data")
    parser.add_argument("--train", type=Path, required=True)
    parser.add_argument("--validation", type=Path, required=True)
    args = parser.parse_args()
    print(json.dumps(finalize_dataset(args.train, args.validation).to_manifest(), indent=2))
