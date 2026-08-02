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
from .telemetry import (
    HudTelemetryConfig,
    TelemetryWorker,
    coerce_captured_frame,
    create_telemetry_worker,
)


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
    telemetry_provider: str = "zero"
    hud_telemetry: HudTelemetryConfig | None = None

    @classmethod
    def from_json(cls, path: str | Path) -> ControlProfile:
        payload: Any = json.loads(Path(path).read_text(encoding="utf-8"))
        if not isinstance(payload, dict):
            raise TypeError("control profile must be a JSON object")
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
        telemetry = payload.get("telemetry_provider", "zero")
        if telemetry not in {"zero", "hud_telemetry"}:
            raise ValueError("telemetry_provider must be zero or hud_telemetry")
        hud_payload = payload.get("hud_telemetry")
        hud_telemetry = (
            HudTelemetryConfig.from_mapping(hud_payload)
            if telemetry == "hud_telemetry"
            else None
        )
        if telemetry == "zero" and hud_payload is not None:
            raise ValueError("hud_telemetry configuration requires its provider")
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
            telemetry_provider=telemetry,
            hud_telemetry=hud_telemetry,
        )

    def sha256(self) -> str:
        canonical = json.dumps(asdict(self), sort_keys=True, separators=(",", ":"))
        return hashlib.sha256(canonical.encode("utf-8")).hexdigest()

    def telemetry_manifest(self) -> dict[str, Any]:
        if self.hud_telemetry is None:
            return {"provider": "zero", "sha256": None}
        return self.hud_telemetry.manifest()


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
        telemetry_worker: TelemetryWorker | None = None,
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
        self.telemetry_worker = telemetry_worker
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

    def record(self, duration_seconds: float | None) -> Path:
        final_dir = self.output_dir / "episodes" / self.episode_id
        temporary_dir = final_dir.with_name(final_dir.name + ".recording")
        if final_dir.exists() or temporary_dir.exists():
            raise FileExistsError(f"episode already exists: {final_dir}")
        red_dir = temporary_dir / "frames_r"
        blue_dir = temporary_dir / "frames_b"
        red_dir.mkdir(parents=True)
        blue_dir.mkdir(parents=True)
        encode_queue: queue.Queue[_EncodedFrame | None] = queue.Queue(maxsize=16)
        encode_error: list[BaseException] = []

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
        telemetry = self.telemetry_worker or create_telemetry_worker(
            self.profile.telemetry_provider, self.profile.hud_telemetry
        )
        fast_deltas: list[tuple[int, int]] = []
        fast_durations: list[float] = []
        fast_timestamps: list[float] = []
        frame_timestamps: list[float] = []
        slow_timestamps: list[float] = []
        movement: list[tuple[int, int]] = []
        buttons: list[list[int]] = []
        contexts: list[tuple[float, float, float, float]] = []
        telemetry.start()
        try:
            self.backend.start()
        except BaseException:
            telemetry.stop(drain=False)
            encode_queue.put(None)
            worker.join()
            shutil.rmtree(temporary_dir, ignore_errors=True)
            raise
        start = time.perf_counter()
        previous_fast = start
        frame_index = 0
        fast_index = 0
        slow_index = 0
        failure: BaseException | None = None
        try:
            maximum_ticks = (
                None
                if duration_seconds is None
                else max(0, math.ceil(duration_seconds * self.cfg.fast_hz))
            )
            while maximum_ticks is None or fast_index < maximum_ticks:
                if self.backend.emergency_stop_requested():
                    break
                deadline = start + fast_index / self.cfg.fast_hz
                remaining = deadline - time.perf_counter()
                if remaining > 0:
                    time.sleep(remaining)
                now = time.perf_counter()
                if not self.backend.target_active():
                    break
                held = self.backend.held_inputs()
                if self.profile.pause_key in held:
                    break
                if now - deadline > 2.0 / self.cfg.fast_hz:
                    break
                failure_duration = (
                    None
                    if self.profile.hud_telemetry is None
                    else self.profile.hud_telemetry.failure_termination_seconds
                )
                if telemetry.should_terminate(now, failure_duration):
                    break
                fast_timestamps.append(now - start)
                fast_durations.append(max(now - previous_fast, 1.0 / self.cfg.fast_hz))
                fast_deltas.append(self.backend.drain_mouse_deltas(previous_fast, now))
                previous_fast = now
                if fast_index % self.cfg.fast_ticks_per_video == 0:
                    # Windows Graphics Capture begins on a background thread.  Give
                    # its first frame a bounded startup window; later samples must
                    # remain non-blocking to preserve the requested cadence.
                    captured = self.backend.latest_frame(
                        timeout_ms=2000 if frame_index == 0 else 0
                    )
                    if captured is None:
                        if frame_index:
                            break
                        raise RuntimeError("no fresh Windows Graphics Capture frame")
                    captured_frame = coerce_captured_frame(captured)
                    captured_at = captured_frame.timestamp
                    frame = captured_frame.model_channels
                    if tuple(frame.shape) != (
                        self.cfg.input_channels,
                        self.cfg.image_height,
                        self.cfg.image_width,
                    ) or frame.dtype != torch.uint8:
                        raise ValueError("native capture returned an invalid RB frame")
                    telemetry.submit(captured_frame)
                    encode_queue.put(_EncodedFrame(frame_index, frame.cpu()))
                    frame_timestamps.append(captured_at - start)
                    frame_index += 1
                if fast_index % self.cfg.fast_ticks_per_slow == 0:
                    movement.append(self._movement(held, self.profile.movement))
                    buttons.append([int(binding in held) for binding in self.profile.buttons])
                    snapshot = telemetry.sample(now)
                    contexts.append(snapshot.values())
                    telemetry.acknowledge(snapshot)
                    slow_timestamps.append(now - start)
                    slow_index += 1
                fast_index += 1
                if encode_error:
                    raise RuntimeError("JPEG encoder failed") from encode_error[0]
        except BaseException as error:
            failure = error
        finally:
            try:
                self.backend.stop()
            finally:
                telemetry.stop()
                encode_queue.put(None)
                worker.join()
        if failure is not None:
            shutil.rmtree(temporary_dir, ignore_errors=True)
            raise failure
        if encode_error:
            shutil.rmtree(temporary_dir, ignore_errors=True)
            raise RuntimeError("JPEG encoder failed") from encode_error[0]
        if not fast_timestamps:
            shutil.rmtree(temporary_dir, ignore_errors=True)
            raise RuntimeError("recording produced no samples")
        context_tensor = torch.tensor(contexts, dtype=torch.float32)
        controls = {
            "fast_timestamps": torch.tensor(fast_timestamps, dtype=torch.float64),
            "frame_timestamps": torch.tensor(frame_timestamps, dtype=torch.float64),
            "slow_timestamps": torch.tensor(slow_timestamps, dtype=torch.float64),
            "raw_mouse_deltas": torch.tensor(fast_deltas, dtype=torch.int32),
            "fast_durations": torch.tensor(fast_durations, dtype=torch.float32),
            "axes": torch.zeros(len(fast_deltas), 2),
            "health": context_tensor[:, 0:1],
            "damage_events": context_tensor[:, 1:2],
            "kill_events": context_tensor[:, 2:3],
            "charge": context_tensor[:, 3:4],
            "movement": torch.tensor(movement, dtype=torch.long),
            "buttons": torch.tensor(buttons, dtype=torch.uint8),
        }
        torch.save(controls, temporary_dir / "controls.pt")
        metadata = {
            "id": self.episode_id,
            "split": self.split,
            "profile_sha256": self.profile.sha256(),
            "telemetry": {
                **self.profile.telemetry_manifest(),
                "diagnostics": telemetry.diagnostics(),
            },
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
) -> tuple[torch.Tensor, torch.Tensor]:
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
    required = {
        "fast_timestamps",
        "frame_timestamps",
        "slow_timestamps",
        "raw_mouse_deltas",
        "fast_durations",
        "axes",
        "health",
        "damage_events",
        "kill_events",
        "charge",
        "movement",
        "buttons",
    }
    if set(controls) != required:
        raise ValueError(f"{episode}: controls keys do not match dataset format 2")
    fast_count = int(controls["fast_timestamps"].shape[0])
    slow_count = int(controls["slow_timestamps"].shape[0])
    if fast_count <= 0 or len(red) != int(controls["frame_timestamps"].shape[0]):
        raise ValueError(f"{episode}: frame/control counts are inconsistent")
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
    for name in ("health", "damage_events", "kill_events", "charge"):
        if controls[name].shape != (slow_count, 1):
            raise ValueError(f"{episode}: {name} must have shape [T_slow, 1]")
    for name in ("fast_timestamps", "frame_timestamps", "slow_timestamps"):
        values = controls[name]
        if not torch.isfinite(values).all() or (
            values.numel() > 1 and not torch.all(values[1:] > values[:-1])
        ):
            raise ValueError(f"{episode}: {name} must be finite and strictly monotonic")
    if not torch.isfinite(controls["fast_durations"]).all() or not torch.all(
        controls["fast_durations"] > 0
    ):
        raise ValueError(f"{episode}: fast durations must be finite and positive")
    if not torch.all((controls["movement"] >= 0) & (controls["movement"] <= 2)):
        raise ValueError(f"{episode}: movement contains a non-class value")
    if not torch.all((controls["buttons"] == 0) | (controls["buttons"] == 1)):
        raise ValueError(f"{episode}: buttons must be binary")
    return controls["raw_mouse_deltas"], controls["fast_durations"]


def finalize_dataset(train_path: Path, validation_path: Path) -> AxisNormalization:
    roots = {train_path.resolve(), validation_path.resolve()}
    if len(roots) != 2:
        raise ValueError("training and validation outputs must be different directories")
    train_episodes = sorted((train_path / "episodes").glob("*"))
    validation_episodes = sorted((validation_path / "episodes").glob("*"))
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
        raw_parts.append(raw)
        duration_parts.append(durations)
    for episode in validation_episodes:
        _validate_episode(episode, num_buttons)
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
    telemetry_manifests = {
        json.dumps(
            {
                key: value
                for key, value in json.loads(
                    (episode / "episode.json").read_text(encoding="utf-8")
                ).get("telemetry", {"provider": "zero", "sha256": None}).items()
                if key != "diagnostics"
            },
            sort_keys=True,
        )
        for episode in (*train_episodes, *validation_episodes)
    }
    if len(telemetry_manifests) != 1:
        raise ValueError("all episodes must use one telemetry configuration")
    telemetry_manifest = json.loads(next(iter(telemetry_manifests)))
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
            controls["axes"] = normalization.normalize(
                controls["raw_mouse_deltas"], controls["fast_durations"]
            )
            temporary = controls_path.with_suffix(".pt.tmp")
            torch.save(controls, temporary)
            os.replace(temporary, controls_path)
            metadata["finalized"] = True
            metadata_temporary = episode / "episode.json.tmp"
            metadata_temporary.write_text(
                json.dumps(metadata, indent=2) + "\n", encoding="utf-8"
            )
            os.replace(metadata_temporary, episode / "episode.json")
            entries.append(_episode_entry(root, episode))
        manifest = {
            "version": 2,
            "split": split,
            "channels": ["R", "B"],
            "axis_normalization": normalization.to_manifest(),
            "num_buttons": num_buttons,
            "control_profile_sha256": next(iter(profile_hashes)),
            "telemetry": telemetry_manifest,
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
    parser.add_argument("--model-config", type=Path, default=Path("configs/h100_1080p.json"))
    args = parser.parse_args()
    profile = ControlProfile.from_json(args.profile)
    cfg = ModelConfig.from_json(args.model_config)
    backend = create_native_backend(args.profile, cfg)
    recorder = EpisodeRecorder(
        backend, profile, cfg, args.output, args.split, args.episode_id
    )
    print(recorder.record(args.duration))


def finalize_main() -> None:
    parser = argparse.ArgumentParser(description="Finalize OverAI train/validation data")
    parser.add_argument("--train", type=Path, required=True)
    parser.add_argument("--validation", type=Path, required=True)
    args = parser.parse_args()
    print(json.dumps(finalize_dataset(args.train, args.validation).to_manifest(), indent=2))
