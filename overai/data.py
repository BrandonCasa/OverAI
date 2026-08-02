"""Timestamp-aligned, streaming demonstration dataset support.

Frames remain compressed image files on disk and are decoded one timestep at a
time.  This avoids the pseudocode's impractical multi-gigabyte in-memory frame
tensor while preserving timestamp order.
"""

from __future__ import annotations

import json
import math
from collections.abc import Sequence
from dataclasses import dataclass
from functools import lru_cache
from pathlib import Path
from typing import Any

import torch
from torch.utils.data import Dataset
from torchvision.io import ImageReadMode, decode_image

from .config import ModelConfig
from .telemetry import SIMILARITY_FORMULA, SIMILARITY_METRIC
from .types import ExecutedActions, ObservationContext, TimingContext

CONTROL_KEYS = (
    "fast_timestamps",
    "frame_timestamps",
    "slow_timestamps",
    "health",
    "damage_events",
    "kill_events",
    "charge",
    "axes",
    "movement",
    "buttons",
)
IMAGE_SUFFIXES = {".jpg", ".jpeg", ".png"}


@dataclass(frozen=True, slots=True)
class EpisodeRecord:
    episode_id: str
    frame_pairs: tuple[tuple[Path, Path], ...]
    controls_path: Path
    fast_count: int
    slow_count: int


@dataclass(slots=True)
class SequenceBatch:
    frame_pairs: list[list[tuple[Path, Path]]]
    fast_timestamps: torch.Tensor
    frame_timestamps: torch.Tensor
    slow_timestamps: torch.Tensor
    health: torch.Tensor
    damage_events: torch.Tensor
    kill_events: torch.Tensor
    charge: torch.Tensor
    axes: torch.Tensor
    movement: torch.Tensor
    buttons: torch.Tensor
    initial_axes: torch.Tensor
    initial_movement: torch.Tensor
    initial_buttons: torch.Tensor
    history_ticks: int
    optimization_ticks: int
    fast_ticks_per_video: int
    fast_ticks_per_slow: int
    fast_hz: int
    image_height: int
    image_width: int

    @property
    def batch_size(self) -> int:
        return self.axes.shape[0]

    @property
    def process_ticks(self) -> int:
        return self.history_ticks + self.optimization_ticks

    def load_frame(self, frame_index: int, device: torch.device) -> torch.Tensor:
        frames = []
        for pairs in self.frame_pairs:
            red_path, blue_path = pairs[frame_index]
            red = decode_image(str(red_path), mode=ImageReadMode.GRAY)
            blue = decode_image(str(blue_path), mode=ImageReadMode.GRAY)
            frames.append(torch.cat((red, blue), dim=0))
        expected = (2, self.image_height, self.image_width)
        for frame in frames:
            if tuple(frame.shape) != expected:
                raise ValueError(
                    f"decoded frame has shape {tuple(frame.shape)}, expected {expected}"
                )
        return torch.stack(frames).to(device=device, non_blocking=True)

    def observation_context(
        self, tick: int, device: torch.device
    ) -> ObservationContext:
        slow_tick = tick // self.fast_ticks_per_slow
        return ObservationContext(
            health=self.health[:, slow_tick].to(device, non_blocking=True),
            damage_event=self.damage_events[:, slow_tick].to(device, non_blocking=True),
            kill_event=self.kill_events[:, slow_tick].to(device, non_blocking=True),
            charge=self.charge[:, slow_tick].to(device, non_blocking=True),
        )

    def executed_actions(self, tick: int, device: torch.device) -> ExecutedActions:
        previous_fast = max(tick - 1, 0)
        current_slow = tick // self.fast_ticks_per_slow
        axes = self.initial_axes if tick == 0 else self.axes[:, previous_fast]
        if current_slow == 0:
            movement = self.initial_movement
            buttons = self.initial_buttons
        else:
            movement = self.movement[:, current_slow - 1]
            buttons = self.buttons[:, current_slow - 1]
        return ExecutedActions(
            movement=movement.to(device, non_blocking=True),
            buttons=buttons.to(device, non_blocking=True),
            axes=axes.to(device, non_blocking=True),
        )

    def timing_context(self, tick: int, device: torch.device) -> TimingContext:
        current = self.fast_timestamps[:, tick : tick + 1]
        frame_index = tick // self.fast_ticks_per_video
        slow_index = tick // self.fast_ticks_per_slow
        if tick == 0:
            delta = torch.full_like(current, 1.0 / self.fast_hz)
        else:
            delta = current - self.fast_timestamps[:, tick - 1 : tick]
        return TimingContext(
            absolute_time=current.to(device, non_blocking=True),
            since_video_frame=(
                current - self.frame_timestamps[:, frame_index : frame_index + 1]
            ).to(device, non_blocking=True),
            since_slow_update=(
                current - self.slow_timestamps[:, slow_index : slow_index + 1]
            ).to(device, non_blocking=True),
            fast_delta_time=delta.to(device, non_blocking=True),
        )


def _load_controls(path: Path) -> dict[str, torch.Tensor]:
    stat = path.stat()
    return _load_controls_cached(path, stat.st_mtime_ns, stat.st_size)


@lru_cache(maxsize=32)
def _load_controls_cached(
    path: Path, modified_ns: int, size: int
) -> dict[str, torch.Tensor]:
    # File metadata is intentionally part of the cache key. Dataset windows reuse
    # the same control tensors heavily, while a rewritten controls.pt must not
    # continue serving the previous tensors from a long-lived worker process.
    del modified_ns, size
    loaded = torch.load(path, map_location="cpu", weights_only=True)
    if not isinstance(loaded, dict):
        raise TypeError(f"{path} must contain a tensor dictionary")
    missing = [key for key in CONTROL_KEYS if key not in loaded]
    if missing:
        raise ValueError(f"{path} is missing controls: {', '.join(missing)}")
    controls: dict[str, torch.Tensor] = {}
    for key in CONTROL_KEYS:
        value = loaded[key]
        if not isinstance(value, torch.Tensor):
            raise TypeError(f"{path}: {key} must be a tensor")
        controls[key] = value.contiguous()
    return controls


def _validate_controls(
    path: Path,
    controls: dict[str, torch.Tensor],
    frame_count: int,
    cfg: ModelConfig,
) -> tuple[int, int]:
    fast_count = controls["axes"].shape[0]
    slow_count = controls["movement"].shape[0]
    expected_fast_shapes = {
        "fast_timestamps": (fast_count,),
        "axes": (fast_count, 2),
    }
    expected_slow_shapes = {
        "slow_timestamps": (slow_count,),
        "health": (slow_count, 1),
        "damage_events": (slow_count, 1),
        "kill_events": (slow_count, 1),
        "charge": (slow_count, 1),
        "movement": (slow_count, 2),
        "buttons": (slow_count, cfg.num_buttons),
    }
    expected_frame_shapes = {"frame_timestamps": (frame_count,)}
    for key, shape in {
        **expected_fast_shapes,
        **expected_slow_shapes,
        **expected_frame_shapes,
    }.items():
        if tuple(controls[key].shape) != shape:
            raise ValueError(
                f"{path}: {key} has shape {tuple(controls[key].shape)}, expected {shape}"
            )
    if fast_count < cfg.fast_horizon + cfg.fast_hz:
        raise ValueError(f"{path}: episode is too short for one training window")
    expected_frames = math.ceil(fast_count / cfg.fast_ticks_per_video)
    expected_slow = math.ceil(fast_count / cfg.fast_ticks_per_slow)
    if frame_count != expected_frames:
        raise ValueError(
            f"{path}: found {frame_count} frames, expected {expected_frames} "
            f"for {fast_count} fast ticks"
        )
    if slow_count != expected_slow:
        raise ValueError(
            f"{path}: found {slow_count} slow controls, expected {expected_slow}"
        )
    for key in ("fast_timestamps", "frame_timestamps", "slow_timestamps"):
        values = controls[key].float()
        if not torch.isfinite(values).all() or (
            values.numel() > 1 and not torch.all(values[1:] > values[:-1])
        ):
            raise ValueError(f"{path}: {key} must be finite and strictly increasing")
    movement = controls["movement"]
    if movement.dtype.is_floating_point or movement.min() < 0 or movement.max() > 2:
        raise ValueError(f"{path}: movement must contain integer classes 0, 1, or 2")
    if not torch.isfinite(controls["axes"]).all():
        raise ValueError(f"{path}: axes contains non-finite values")
    for key in ("health", "damage_events", "kill_events", "charge"):
        if not torch.isfinite(controls[key]).all():
            raise ValueError(f"{path}: {key} contains non-finite values")
    if controls["axes"].abs().max() > 1.0001:
        raise ValueError(f"{path}: axes must be normalized to [-1, 1]")
    buttons = controls["buttons"]
    if not torch.all((buttons == 0) | (buttons == 1)):
        raise ValueError(f"{path}: buttons must be binary states")
    return fast_count, slow_count


def load_manifest(path: str | Path, cfg: ModelConfig) -> list[EpisodeRecord]:
    manifest_path = Path(path).resolve()
    data: Any = json.loads(manifest_path.read_text(encoding="utf-8"))
    if not isinstance(data, dict):
        raise TypeError("dataset manifest must be an object")
    if data.get("version") != 2:
        raise ValueError("dataset manifest must have version 2")
    if data.get("channels") != ["R", "B"]:
        raise ValueError("dataset manifest channels must be ['R', 'B']")
    if data.get("split") not in {"train", "validation"}:
        raise ValueError("dataset manifest split must be train or validation")
    profile_hash = data.get("control_profile_sha256")
    if not isinstance(profile_hash, str) or not profile_hash:
        raise ValueError("dataset manifest must contain a control profile hash")
    # Format-2 manifests created before HUD support could only contain zeros.
    telemetry = data.get("telemetry", {"provider": "zero", "sha256": None})
    if not isinstance(telemetry, dict) or telemetry.get("provider") not in {
        "zero",
        "hud_telemetry",
    }:
        raise ValueError("dataset manifest must contain a telemetry configuration")
    if telemetry.get("provider") == "hud_telemetry":
        if not isinstance(telemetry.get("sha256"), str) or not telemetry["sha256"]:
            raise ValueError("HUD telemetry manifest must contain its configuration hash")
        if telemetry.get("similarity_metric") != SIMILARITY_METRIC or telemetry.get(
            "similarity_formula"
        ) != SIMILARITY_FORMULA:
            raise ValueError("HUD telemetry manifest uses an unsupported color metric")
    normalization = data.get("axis_normalization")
    if not isinstance(normalization, dict):
        raise TypeError("dataset manifest axis_normalization must be an object")
    scales = normalization.get("scale_counts_per_second")
    if (
        not isinstance(scales, list)
        or len(scales) != 2
        or any(not isinstance(value, (int, float)) or value <= 0 for value in scales)
    ):
        raise ValueError("axis normalization must contain two positive scales")
    episodes = data.get("episodes")
    if not isinstance(episodes, list):
        raise TypeError("dataset manifest episodes must be a list")
    if not episodes:
        raise ValueError("dataset manifest must contain at least one episode")

    records: list[EpisodeRecord] = []
    seen_ids: set[str] = set()
    for item in episodes:
        if not isinstance(item, dict):
            raise TypeError("each manifest episode must be an object")
        episode_id = item.get("id")
        if not isinstance(episode_id, str):
            raise TypeError("episode ids must be strings")
        if not episode_id or episode_id in seen_ids:
            raise ValueError("episode ids must be non-empty and unique")
        seen_ids.add(episode_id)
        red_frame_dir = (
            manifest_path.parent / str(item.get("red_frames", ""))
        ).resolve()
        blue_frame_dir = (
            manifest_path.parent / str(item.get("blue_frames", ""))
        ).resolve()
        controls_path = (manifest_path.parent / str(item.get("controls", ""))).resolve()
        if not red_frame_dir.is_dir():
            raise FileNotFoundError(f"red frame directory does not exist: {red_frame_dir}")
        if not blue_frame_dir.is_dir():
            raise FileNotFoundError(
                f"blue frame directory does not exist: {blue_frame_dir}"
            )
        if not controls_path.is_file():
            raise FileNotFoundError(f"controls file does not exist: {controls_path}")
        def channel_paths(directory: Path) -> tuple[Path, ...]:
            return tuple(
                sorted(
                    candidate
                    for candidate in directory.iterdir()
                    if candidate.is_file()
                    and candidate.suffix.lower() in IMAGE_SUFFIXES
                )
            )

        red_paths = channel_paths(red_frame_dir)
        blue_paths = channel_paths(blue_frame_dir)
        if len(red_paths) != len(blue_paths):
            raise ValueError(
                f"{episode_id}: red/blue frame counts differ: "
                f"{len(red_paths)} != {len(blue_paths)}"
            )
        if [path.name for path in red_paths] != [path.name for path in blue_paths]:
            raise ValueError(f"{episode_id}: red/blue frame names do not match")
        frame_pairs = tuple(zip(red_paths, blue_paths, strict=True))
        controls = _load_controls(controls_path)
        fast_count, slow_count = _validate_controls(
            controls_path, controls, len(frame_pairs), cfg
        )
        records.append(
            EpisodeRecord(
                episode_id, frame_pairs, controls_path, fast_count, slow_count
            )
        )
    return records


class DemonstrationWindowDataset(Dataset[dict[str, Any]]):
    """Causal episode windows with history, optimization span, and horizons."""

    def __init__(
        self,
        manifest_path: str | Path,
        cfg: ModelConfig,
        history_seconds: float = 30.0,
        optimization_seconds: float = 2.0,
        stride_seconds: float = 2.0,
    ) -> None:
        self.cfg = cfg
        self.records = load_manifest(manifest_path, cfg)
        self.history_ticks = self._duration_ticks(history_seconds, "history_seconds")
        self.optimization_ticks = self._duration_ticks(
            optimization_seconds, "optimization_seconds"
        )
        self.stride_ticks = self._duration_ticks(stride_seconds, "stride_seconds")
        for name, ticks in (
            ("history_seconds", self.history_ticks),
            ("optimization_seconds", self.optimization_ticks),
            ("stride_seconds", self.stride_ticks),
        ):
            if ticks % cfg.fast_ticks_per_slow:
                raise ValueError(f"{name} must align to a {cfg.slow_hz} Hz boundary")

        self.windows: list[tuple[int, int]] = []
        for record_index, record in enumerate(self.records):
            maximum_anchor = (
                record.fast_count - self.optimization_ticks - cfg.fast_horizon
            )
            for anchor in range(
                self.history_ticks, maximum_anchor + 1, self.stride_ticks
            ):
                last_slow = (
                    anchor + self.optimization_ticks - 1
                ) // cfg.fast_ticks_per_slow
                if last_slow + cfg.slow_horizon <= record.slow_count:
                    self.windows.append((record_index, anchor))
        if not self.windows:
            raise ValueError(
                "dataset has no usable windows; provide longer episodes or reduce history"
            )

    def _duration_ticks(self, seconds: float, name: str) -> int:
        ticks = round(seconds * self.cfg.fast_hz)
        if seconds < 0 or (name != "history_seconds" and ticks <= 0):
            raise ValueError(f"{name} must be positive")
        if not math.isclose(ticks / self.cfg.fast_hz, seconds, abs_tol=1e-9):
            raise ValueError(f"{name} must align to a {self.cfg.fast_hz} Hz tick")
        return ticks

    def __len__(self) -> int:
        return len(self.windows)

    def __getitem__(self, index: int) -> dict[str, Any]:
        record_index, anchor = self.windows[index]
        record = self.records[record_index]
        controls = _load_controls(record.controls_path)
        base_tick = anchor - self.history_ticks
        process_end = anchor + self.optimization_ticks
        fast_end = process_end + self.cfg.fast_horizon
        base_frame = base_tick // self.cfg.fast_ticks_per_video
        frame_end = math.ceil(process_end / self.cfg.fast_ticks_per_video)
        base_slow = base_tick // self.cfg.fast_ticks_per_slow
        last_slow = (process_end - 1) // self.cfg.fast_ticks_per_slow
        slow_end = last_slow + self.cfg.slow_horizon

        initial_axes = (
            torch.zeros_like(controls["axes"][0])
            if base_tick == 0
            else controls["axes"][base_tick - 1]
        )
        initial_movement = (
            torch.ones_like(controls["movement"][0])
            if base_slow == 0
            else controls["movement"][base_slow - 1]
        )
        initial_buttons = (
            torch.zeros_like(controls["buttons"][0])
            if base_slow == 0
            else controls["buttons"][base_slow - 1]
        )

        return {
            "frame_pairs": list(record.frame_pairs[base_frame:frame_end]),
            "fast_timestamps": controls["fast_timestamps"][base_tick:fast_end],
            "frame_timestamps": controls["frame_timestamps"][base_frame:frame_end],
            "slow_timestamps": controls["slow_timestamps"][base_slow:slow_end],
            "health": controls["health"][base_slow:slow_end],
            "damage_events": controls["damage_events"][base_slow:slow_end],
            "kill_events": controls["kill_events"][base_slow:slow_end],
            "charge": controls["charge"][base_slow:slow_end],
            "axes": controls["axes"][base_tick:fast_end],
            "movement": controls["movement"][base_slow:slow_end],
            "buttons": controls["buttons"][base_slow:slow_end],
            "initial_axes": initial_axes,
            "initial_movement": initial_movement,
            "initial_buttons": initial_buttons,
        }

    def collate(self, samples: Sequence[dict[str, Any]]) -> SequenceBatch:
        if not samples:
            raise ValueError("cannot collate an empty batch")

        def stack(key: str) -> torch.Tensor:
            return torch.stack([sample[key] for sample in samples])

        return SequenceBatch(
            frame_pairs=[sample["frame_pairs"] for sample in samples],
            fast_timestamps=stack("fast_timestamps"),
            frame_timestamps=stack("frame_timestamps"),
            slow_timestamps=stack("slow_timestamps"),
            health=stack("health"),
            damage_events=stack("damage_events"),
            kill_events=stack("kill_events"),
            charge=stack("charge"),
            axes=stack("axes"),
            movement=stack("movement"),
            buttons=stack("buttons"),
            initial_axes=stack("initial_axes"),
            initial_movement=stack("initial_movement"),
            initial_buttons=stack("initial_buttons"),
            history_ticks=self.history_ticks,
            optimization_ticks=self.optimization_ticks,
            fast_ticks_per_video=self.cfg.fast_ticks_per_video,
            fast_ticks_per_slow=self.cfg.fast_ticks_per_slow,
            fast_hz=self.cfg.fast_hz,
            image_height=self.cfg.image_height,
            image_width=self.cfg.image_width,
        )


def dataset_summary(path: str | Path, cfg: ModelConfig) -> dict[str, int]:
    records = load_manifest(path, cfg)
    return {
        "episodes": len(records),
        "frames": sum(len(record.frame_pairs) for record in records),
        "fast_ticks": sum(record.fast_count for record in records),
        "slow_ticks": sum(record.slow_count for record in records),
    }
