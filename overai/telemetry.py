"""Shared, causal HUD telemetry analysis for recording and inference."""

from __future__ import annotations

import hashlib
import json
import math
import queue
import re
import threading
import time
from dataclasses import asdict, dataclass
from typing import Any, Protocol, cast

import torch
from torch.nn import functional as F

SIMILARITY_METRIC = "normalized_rgb_euclidean_v1"
SIMILARITY_FORMULA = "1-rgb_euclidean_distance/sqrt(3*255^2)"


def _tuple_of_ints(value: Any, length: int, name: str) -> tuple[int, ...]:
    if (
        not isinstance(value, list)
        or len(value) != length
        or any(not isinstance(item, int) or isinstance(item, bool) for item in value)
    ):
        raise ValueError(f"{name} must contain exactly {length} integers")
    return tuple(value)


def _positive_float(value: Any, name: str, *, allow_zero: bool = False) -> float:
    if not isinstance(value, (int, float)) or isinstance(value, bool):
        raise TypeError(f"{name} must be numeric")
    result = float(value)
    if not math.isfinite(result) or result < 0 or (result == 0 and not allow_zero):
        qualifier = "non-negative" if allow_zero else "positive"
        raise ValueError(f"{name} must be finite and {qualifier}")
    return result


@dataclass(frozen=True, slots=True)
class OcrPreprocessing:
    threshold: int = 160
    scale: float = 4.0
    allowed_glyphs: str = "0123456789.%/"
    polarity: str = "light_on_dark"
    minimum_confidence: float = 0.72

    @classmethod
    def from_mapping(cls, value: Any) -> OcrPreprocessing:
        if not isinstance(value, dict):
            raise TypeError("hud_telemetry.ocr must be an object")
        threshold = value.get("threshold", 160)
        if (
            not isinstance(threshold, int)
            or isinstance(threshold, bool)
            or not 0 <= threshold <= 255
        ):
            raise ValueError(
                "hud_telemetry.ocr.threshold must be an integer in [0,255]"
            )
        scale = _positive_float(value.get("scale", 4.0), "hud_telemetry.ocr.scale")
        glyphs = value.get("allowed_glyphs", "0123456789.%/")
        if (
            not isinstance(glyphs, str)
            or not glyphs
            or any(ord(char) > 127 for char in glyphs)
        ):
            raise ValueError("hud_telemetry.ocr.allowed_glyphs must be non-empty ASCII")
        polarity = value.get("polarity", "light_on_dark")
        if polarity not in {"light_on_dark", "dark_on_light", "auto"}:
            raise ValueError("hud_telemetry.ocr.polarity is invalid")
        confidence = _positive_float(
            value.get("minimum_confidence", 0.72),
            "hud_telemetry.ocr.minimum_confidence",
            allow_zero=True,
        )
        if confidence > 1:
            raise ValueError("hud_telemetry.ocr.minimum_confidence must be in [0,1]")
        return cls(threshold, scale, glyphs, polarity, confidence)


@dataclass(frozen=True, slots=True)
class ContinuousRegion:
    bbox: tuple[int, int, int, int]
    maximum: float
    max_stale_seconds: float


@dataclass(frozen=True, slots=True)
class EventRegion:
    points: tuple[tuple[int, int], ...]
    colors: tuple[tuple[int, int, int], ...]
    debounce_seconds: float
    rearm_seconds: float


@dataclass(frozen=True, slots=True)
class HudTelemetryConfig:
    reference_resolution: tuple[int, int]
    ocr: OcrPreprocessing
    health: ContinuousRegion
    charge: ContinuousRegion
    hitmarker: EventRegion
    kill: EventRegion
    similarity_threshold: float = 0.90
    similarity_metric: str = SIMILARITY_METRIC
    failure_termination_seconds: float | None = None

    @staticmethod
    def _continuous(
        value: Any, name: str, *, maximum: float | None
    ) -> ContinuousRegion:
        if not isinstance(value, dict):
            raise TypeError(f"hud_telemetry.{name} must be an object")
        bbox = cast(
            tuple[int, int, int, int],
            _tuple_of_ints(value.get("bbox"), 4, f"hud_telemetry.{name}.bbox"),
        )
        if bbox[0] < 0 or bbox[1] < 0 or bbox[2] <= 0 or bbox[3] <= 0:
            raise ValueError(f"hud_telemetry.{name}.bbox must be [x,y,width,height]")
        if maximum is not None and "maximum" in value and value["maximum"] != maximum:
            raise ValueError(f"hud_telemetry.{name}.maximum must be {maximum:g}")
        configured_maximum = value.get("maximum", maximum)
        result_maximum = _positive_float(
            configured_maximum, f"hud_telemetry.{name}.maximum"
        )
        stale = _positive_float(
            value.get("max_stale_seconds", 0.5),
            f"hud_telemetry.{name}.max_stale_seconds",
            allow_zero=True,
        )
        return ContinuousRegion(bbox, result_maximum, stale)

    @staticmethod
    def _event(
        value: Any, name: str, *, point_count: int, color_count: int
    ) -> EventRegion:
        if not isinstance(value, dict):
            raise TypeError(f"hud_telemetry.{name} must be an object")
        raw_points = value.get("points")
        if not isinstance(raw_points, list) or len(raw_points) != point_count:
            raise ValueError(
                f"hud_telemetry.{name}.points must contain {point_count} points"
            )
        points = cast(
            tuple[tuple[int, int], ...],
            tuple(
                _tuple_of_ints(point, 2, f"hud_telemetry.{name}.points[{index}]")
                for index, point in enumerate(raw_points)
            ),
        )
        if any(x < 0 or y < 0 for x, y in points):
            raise ValueError(f"hud_telemetry.{name}.points must be non-negative")
        raw_colors = value.get("colors")
        if not isinstance(raw_colors, list) or len(raw_colors) != color_count:
            raise ValueError(
                f"hud_telemetry.{name}.colors must contain {color_count} RGB colors"
            )
        colors = cast(
            tuple[tuple[int, int, int], ...],
            tuple(
                _tuple_of_ints(color, 3, f"hud_telemetry.{name}.colors[{index}]")
                for index, color in enumerate(raw_colors)
            ),
        )
        if any(channel < 0 or channel > 255 for color in colors for channel in color):
            raise ValueError(f"hud_telemetry.{name}.colors channels must be in [0,255]")
        debounce = _positive_float(
            value.get("debounce_seconds", 0.0),
            f"hud_telemetry.{name}.debounce_seconds",
            allow_zero=True,
        )
        rearm = _positive_float(
            value.get("rearm_seconds", 0.1),
            f"hud_telemetry.{name}.rearm_seconds",
            allow_zero=True,
        )
        return EventRegion(points, colors, debounce, rearm)

    @classmethod
    def from_mapping(cls, value: Any) -> HudTelemetryConfig:
        if not isinstance(value, dict):
            raise TypeError("hud_telemetry must be an object")
        resolution = cast(
            tuple[int, int],
            _tuple_of_ints(
                value.get("reference_resolution", [1920, 1080]),
                2,
                "hud_telemetry.reference_resolution",
            ),
        )
        if resolution[0] <= 0 or resolution[1] <= 0:
            raise ValueError("hud_telemetry.reference_resolution must be positive")
        threshold = _positive_float(
            value.get("similarity_threshold", 0.90),
            "hud_telemetry.similarity_threshold",
            allow_zero=True,
        )
        if threshold > 1:
            raise ValueError("hud_telemetry.similarity_threshold must be in [0,1]")
        metric = value.get("similarity_metric", SIMILARITY_METRIC)
        if metric != SIMILARITY_METRIC:
            raise ValueError(f"similarity_metric must be {SIMILARITY_METRIC!r}")
        failure = value.get("failure_termination_seconds")
        failure_seconds = (
            None
            if failure is None
            else _positive_float(failure, "hud_telemetry.failure_termination_seconds")
        )
        result = cls(
            resolution,
            OcrPreprocessing.from_mapping(value.get("ocr", {})),
            cls._continuous(value.get("health"), "health", maximum=None),
            cls._continuous(value.get("charge"), "charge", maximum=100.0),
            cls._event(
                value.get("hitmarker"), "hitmarker", point_count=4, color_count=2
            ),
            cls._event(value.get("kill"), "kill", point_count=7, color_count=1),
            threshold,
            metric,
            failure_seconds,
        )
        reference_width, reference_height = result.reference_resolution
        for name, region in (("health", result.health), ("charge", result.charge)):
            x, y, width, height = region.bbox
            if x + width > reference_width or y + height > reference_height:
                raise ValueError(
                    f"hud_telemetry.{name}.bbox exceeds reference resolution"
                )
        for name, region in (("hitmarker", result.hitmarker), ("kill", result.kill)):
            if any(
                x >= reference_width or y >= reference_height for x, y in region.points
            ):
                raise ValueError(
                    f"hud_telemetry.{name}.points exceed reference resolution"
                )
        return result

    def sha256(self) -> str:
        canonical = json.dumps(asdict(self), sort_keys=True, separators=(",", ":"))
        return hashlib.sha256(canonical.encode("utf-8")).hexdigest()

    def manifest(self) -> dict[str, Any]:
        return {
            "provider": "hud_telemetry",
            "sha256": self.sha256(),
            "configuration": json.loads(json.dumps(asdict(self))),
            "reference_resolution": list(self.reference_resolution),
            "similarity_metric": self.similarity_metric,
            "similarity_formula": SIMILARITY_FORMULA,
            "similarity_threshold": self.similarity_threshold,
        }


@dataclass(frozen=True, slots=True)
class CapturedFrame:
    timestamp: float
    model_channels: torch.Tensor
    bgra_surface: torch.Tensor | None = None


def coerce_captured_frame(value: Any) -> CapturedFrame:
    if isinstance(value, CapturedFrame):
        return value
    if isinstance(value, tuple) and len(value) in {2, 3}:
        timestamp, channels = value[:2]
        surface = value[2] if len(value) == 3 else None
        if (
            isinstance(timestamp, (int, float))
            and isinstance(channels, torch.Tensor)
            and surface is None
            or isinstance(surface, torch.Tensor)
        ):
            return CapturedFrame(float(timestamp), channels, surface)
    raise TypeError("capture backend returned an invalid frame object")


@dataclass(frozen=True, slots=True)
class OcrResult:
    text: str
    confidence: float


class OcrEngine(Protocol):
    def read(
        self,
        bgra: torch.Tensor,
        bbox: tuple[int, int, int, int],
        config: OcrPreprocessing,
    ) -> OcrResult: ...


_GLYPHS = {
    "0": ("11111", "10001", "10011", "10101", "11001", "10001", "11111"),
    "1": ("00100", "01100", "00100", "00100", "00100", "00100", "01110"),
    "2": ("11110", "00001", "00001", "11110", "10000", "10000", "11111"),
    "3": ("11110", "00001", "00001", "01110", "00001", "00001", "11110"),
    "4": ("10010", "10010", "10010", "11111", "00010", "00010", "00010"),
    "5": ("11111", "10000", "10000", "11110", "00001", "00001", "11110"),
    "6": ("01111", "10000", "10000", "11110", "10001", "10001", "01110"),
    "7": ("11111", "00001", "00010", "00100", "01000", "01000", "01000"),
    "8": ("01110", "10001", "10001", "01110", "10001", "10001", "01110"),
    "9": ("01110", "10001", "10001", "01111", "00001", "00001", "11110"),
    ".": ("00000", "00000", "00000", "00000", "00000", "00110", "00110"),
    "%": ("11001", "11010", "00100", "01000", "10110", "00110", "00000"),
    "/": ("00001", "00010", "00010", "00100", "01000", "01000", "10000"),
}


class BitmapGlyphOcr:
    """Small deterministic OCR for fixed bitmap HUD digits and synthetic tests.

    Game-specific screenshots are still required before its glyph templates can be
    replaced or augmented for a production HUD font.
    """

    @staticmethod
    def _mask(crop: torch.Tensor, config: OcrPreprocessing) -> torch.Tensor:
        if crop.ndim != 3 or crop.shape[0] != 4 or crop.dtype != torch.uint8:
            raise ValueError("OCR expects a uint8 BGRA [4,H,W] surface")
        blue, green, red = crop[0].float(), crop[1].float(), crop[2].float()
        gray = red * 0.299 + green * 0.587 + blue * 0.114
        if config.scale != 1:
            gray = F.interpolate(
                gray[None, None], scale_factor=config.scale, mode="nearest"
            )[0, 0]
        light = gray >= config.threshold
        dark = gray <= config.threshold
        if config.polarity == "light_on_dark":
            return light
        if config.polarity == "dark_on_light":
            return dark
        light_count = int(light.sum())
        dark_count = int(dark.sum())
        if not light_count:
            return dark
        if not dark_count:
            return light
        return light if light_count <= dark_count else dark

    def read(
        self,
        bgra: torch.Tensor,
        bbox: tuple[int, int, int, int],
        config: OcrPreprocessing,
    ) -> OcrResult:
        x, y, width, height = bbox
        crop = bgra[:, y : y + height, x : x + width]
        if crop.shape[1] != height or crop.shape[2] != width:
            return OcrResult("", 0.0)
        mask = self._mask(crop, config)
        active_columns = mask.any(dim=0)
        runs: list[tuple[int, int]] = []
        start: int | None = None
        for index, active in enumerate(active_columns.tolist() + [False]):
            if active and start is None:
                start = index
            elif not active and start is not None:
                runs.append((start, index))
                start = None
        if not runs:
            return OcrResult("", 0.0)
        allowed = [char for char in config.allowed_glyphs if char in _GLYPHS]
        text = ""
        confidences: list[float] = []
        for left, right in runs:
            glyph = mask[:, left:right]
            active_rows = glyph.any(dim=1)
            glyph = glyph[active_rows].float()[None, None]
            normalized = F.interpolate(glyph, size=(14, 10), mode="nearest")[0, 0]
            best_char = ""
            best_score = -1.0
            for char in allowed:
                template = torch.tensor(
                    [[cell == "1" for cell in row] for row in _GLYPHS[char]],
                    dtype=torch.bool,
                )
                template = template[template.any(dim=1)][:, template.any(dim=0)]
                resized_template = F.interpolate(
                    template.float()[None, None], size=(14, 10), mode="nearest"
                )[0, 0]
                score = 1.0 - float((normalized - resized_template).abs().mean())
                if score > best_score:
                    best_char, best_score = char, score
            if not best_char:
                return OcrResult("", 0.0)
            text += best_char
            confidences.append(best_score)
        confidence = min(confidences)
        if confidence < config.minimum_confidence:
            return OcrResult(text, confidence)
        return OcrResult(text, confidence)


def rgb_similarity(
    left: tuple[int | float, int | float, int | float],
    right: tuple[int | float, int | float, int | float],
) -> float:
    distance = math.sqrt(sum((a - b) ** 2 for a, b in zip(left, right, strict=True)))
    return 1.0 - distance / math.sqrt(3.0 * 255.0**2)


@dataclass(slots=True)
class _EventState:
    armed: bool = True
    match_started_at: float | None = None
    clear_started_at: float | None = None
    produced: int = 0
    consumed: int = 0


@dataclass(frozen=True, slots=True)
class TelemetrySnapshot:
    timestamp: float
    health: float
    damage_event: float
    kill_event: float
    charge: float
    damage_token: int = 0
    kill_token: int = 0

    def values(self) -> tuple[float, float, float, float]:
        return self.health, self.damage_event, self.kill_event, self.charge


class TelemetryProvider(Protocol):
    def analyze_frame(self, timestamp: float, bgra: torch.Tensor) -> None: ...
    def sample(self, timestamp: float | None = None) -> TelemetrySnapshot: ...
    def acknowledge(self, snapshot: TelemetrySnapshot) -> None: ...
    def diagnostics(self) -> dict[str, Any]: ...


class ZeroTelemetryProvider:
    def analyze_frame(self, timestamp: float, bgra: torch.Tensor) -> None:
        return None

    def sample(self, timestamp: float | None = None) -> TelemetrySnapshot:
        return TelemetrySnapshot(
            time.perf_counter() if timestamp is None else timestamp, 0, 0, 0, 0
        )

    def acknowledge(self, snapshot: TelemetrySnapshot) -> None:
        return None

    def diagnostics(self) -> dict[str, Any]:
        return {"provider": "zero"}


class HudTelemetryProvider:
    def __init__(
        self, config: HudTelemetryConfig, ocr: OcrEngine | None = None
    ) -> None:
        self.config = config
        self.ocr = BitmapGlyphOcr() if ocr is None else ocr
        self._lock = threading.Lock()
        self._health: tuple[float, float] | None = None
        self._charge: tuple[float, float] | None = None
        self._damage = _EventState()
        self._kill = _EventState()
        self._last_frame_timestamp = float("-inf")
        self._counters: dict[str, int] = {
            "frames_analyzed": 0,
            "out_of_order_frames": 0,
            "health_ocr_calls": 0,
            "health_valid": 0,
            "health_invalid": 0,
            "health_stale_samples": 0,
            "health_expired_samples": 0,
            "charge_ocr_calls": 0,
            "charge_valid": 0,
            "charge_invalid": 0,
            "charge_stale_samples": 0,
            "charge_expired_samples": 0,
            "damage_events": 0,
            "kill_events": 0,
        }
        self._last_ocr: dict[str, dict[str, Any]] = {
            "health": {"text": "", "confidence": 0.0},
            "charge": {"text": "", "confidence": 0.0},
        }

    def _scaled_bbox(
        self, bbox: tuple[int, int, int, int], width: int, height: int
    ) -> tuple[int, int, int, int]:
        ref_width, ref_height = self.config.reference_resolution
        x, y, box_width, box_height = bbox
        left = round(x * width / ref_width)
        top = round(y * height / ref_height)
        right = round((x + box_width) * width / ref_width)
        bottom = round((y + box_height) * height / ref_height)
        return left, top, max(1, right - left), max(1, bottom - top)

    def _scaled_point(
        self, point: tuple[int, int], width: int, height: int
    ) -> tuple[int, int]:
        ref_width, ref_height = self.config.reference_resolution
        x = min(
            width - 1, max(0, round(point[0] * (width - 1) / max(1, ref_width - 1)))
        )
        y = min(
            height - 1, max(0, round(point[1] * (height - 1) / max(1, ref_height - 1)))
        )
        return x, y

    @staticmethod
    def _parse_number(
        result: OcrResult, maximum: float, minimum_confidence: float
    ) -> float | None:
        if result.confidence < minimum_confidence:
            return None
        match = re.search(r"(?:\d+(?:\.\d*)?|\.\d+)", result.text)
        if match is None:
            return None
        try:
            value = float(match.group(0))
        except ValueError:
            return None
        if not math.isfinite(value) or not 0 <= value <= maximum:
            return None
        return value / maximum

    def _all_points_match(self, surface: torch.Tensor, region: EventRegion) -> bool:
        _, height, width = surface.shape
        for reference_point in region.points:
            x, y = self._scaled_point(reference_point, width, height)
            sampled = (
                int(surface[2, y, x]),
                int(surface[1, y, x]),
                int(surface[0, y, x]),
            )
            if not any(
                rgb_similarity(sampled, accepted) >= self.config.similarity_threshold
                for accepted in region.colors
            ):
                return False
        return True

    @staticmethod
    def _update_event(
        state: _EventState, matched: bool, timestamp: float, config: EventRegion
    ) -> bool:
        if matched:
            state.clear_started_at = None
            if not state.armed:
                return False
            if state.match_started_at is None:
                state.match_started_at = timestamp
            if timestamp - state.match_started_at >= config.debounce_seconds:
                state.armed = False
                state.match_started_at = None
                state.produced += 1
                return True
            return False
        state.match_started_at = None
        if state.armed:
            return False
        if state.clear_started_at is None:
            state.clear_started_at = timestamp
        if timestamp - state.clear_started_at >= config.rearm_seconds:
            state.armed = True
            state.clear_started_at = None
        return False

    def analyze_frame(self, timestamp: float, bgra: torch.Tensor) -> None:
        if bgra.ndim != 3 or bgra.shape[0] != 4 or bgra.dtype != torch.uint8:
            raise ValueError("HUD telemetry requires uint8 BGRA [4,H,W]")
        with self._lock:
            if timestamp <= self._last_frame_timestamp:
                self._counters["out_of_order_frames"] += 1
                return
            self._last_frame_timestamp = timestamp
            self._counters["frames_analyzed"] += 1
        _, height, width = bgra.shape
        health_result = self.ocr.read(
            bgra,
            self._scaled_bbox(self.config.health.bbox, width, height),
            self.config.ocr,
        )
        charge_result = self.ocr.read(
            bgra,
            self._scaled_bbox(self.config.charge.bbox, width, height),
            self.config.ocr,
        )
        health = self._parse_number(
            health_result,
            self.config.health.maximum,
            self.config.ocr.minimum_confidence,
        )
        charge = self._parse_number(
            charge_result, 100.0, self.config.ocr.minimum_confidence
        )
        damage_match = self._all_points_match(bgra, self.config.hitmarker)
        kill_match = self._all_points_match(bgra, self.config.kill)
        with self._lock:
            self._counters["health_ocr_calls"] += 1
            self._counters["charge_ocr_calls"] += 1
            self._last_ocr["health"] = asdict(health_result)
            self._last_ocr["charge"] = asdict(charge_result)
            if health is None:
                self._counters["health_invalid"] += 1
            else:
                self._health = (health, timestamp)
                self._counters["health_valid"] += 1
            if charge is None:
                self._counters["charge_invalid"] += 1
            else:
                self._charge = (charge, timestamp)
                self._counters["charge_valid"] += 1
            if self._update_event(
                self._damage, damage_match, timestamp, self.config.hitmarker
            ):
                self._counters["damage_events"] += 1
            if self._update_event(self._kill, kill_match, timestamp, self.config.kill):
                self._counters["kill_events"] += 1

    def _continuous_value(
        self,
        name: str,
        value: tuple[float, float] | None,
        timestamp: float,
        stale: float,
    ) -> float:
        if value is None or timestamp - value[1] > stale:
            self._counters[f"{name}_expired_samples"] += 1
            return 0.0
        if timestamp > value[1]:
            self._counters[f"{name}_stale_samples"] += 1
        return value[0]

    def sample(self, timestamp: float | None = None) -> TelemetrySnapshot:
        now = time.perf_counter() if timestamp is None else timestamp
        with self._lock:
            return TelemetrySnapshot(
                now,
                self._continuous_value(
                    "health", self._health, now, self.config.health.max_stale_seconds
                ),
                float(self._damage.produced > self._damage.consumed),
                float(self._kill.produced > self._kill.consumed),
                self._continuous_value(
                    "charge", self._charge, now, self.config.charge.max_stale_seconds
                ),
                self._damage.produced,
                self._kill.produced,
            )

    def acknowledge(self, snapshot: TelemetrySnapshot) -> None:
        with self._lock:
            self._damage.consumed = max(self._damage.consumed, snapshot.damage_token)
            self._kill.consumed = max(self._kill.consumed, snapshot.kill_token)

    def diagnostics(self) -> dict[str, Any]:
        with self._lock:
            return {
                "provider": "hud_telemetry",
                "config_sha256": self.config.sha256(),
                "counters": dict(self._counters),
                "last_ocr": json.loads(json.dumps(self._last_ocr)),
                "last_frame_timestamp": (
                    None
                    if self._last_frame_timestamp == float("-inf")
                    else self._last_frame_timestamp
                ),
            }


class TelemetryWorker:
    """Bounded latest-frame worker; inference and capture never wait for OCR."""

    def __init__(self, provider: TelemetryProvider) -> None:
        self.provider = provider
        self._queue: queue.Queue[tuple[float, torch.Tensor] | None] = queue.Queue(
            maxsize=1
        )
        self._thread: threading.Thread | None = None
        self._lock = threading.Lock()
        self._submitted = 0
        self._dropped = 0
        self._errors = 0
        self._failure_started_at: float | None = None
        self._last_error: str | None = None

    def start(self) -> None:
        if self._thread is not None:
            return

        def run() -> None:
            while True:
                item = self._queue.get()
                if item is None:
                    return
                timestamp, surface = item
                try:
                    self.provider.analyze_frame(timestamp, surface)
                except BaseException as error:
                    with self._lock:
                        self._errors += 1
                        if self._failure_started_at is None:
                            self._failure_started_at = timestamp
                        self._last_error = f"{type(error).__name__}: {error}"
                else:
                    with self._lock:
                        self._failure_started_at = None
                        self._last_error = None

        self._thread = threading.Thread(
            target=run, name="overai-telemetry", daemon=True
        )
        self._thread.start()

    def submit(self, frame: CapturedFrame) -> None:
        with self._lock:
            self._submitted += 1
        if frame.bgra_surface is None:
            item = (frame.timestamp, torch.empty(0, dtype=torch.uint8))
        else:
            item = (frame.timestamp, frame.bgra_surface)
        try:
            self._queue.put_nowait(item)
        except queue.Full:
            try:
                self._queue.get_nowait()
            except queue.Empty:
                pass
            with self._lock:
                self._dropped += 1
            self._queue.put_nowait(item)

    def sample(self, timestamp: float | None = None) -> TelemetrySnapshot:
        return self.provider.sample(timestamp)

    def acknowledge(self, snapshot: TelemetrySnapshot) -> None:
        self.provider.acknowledge(snapshot)

    def should_terminate(self, timestamp: float, duration: float | None) -> bool:
        if duration is None:
            return False
        with self._lock:
            return (
                self._failure_started_at is not None
                and timestamp - self._failure_started_at >= duration
            )

    def stop(self, *, drain: bool = True) -> None:
        if self._thread is None:
            return
        if not drain:
            try:
                self._queue.get_nowait()
            except queue.Empty:
                pass
        self._queue.put(None)
        self._thread.join()
        self._thread = None

    def diagnostics(self) -> dict[str, Any]:
        with self._lock:
            worker = {
                "frames_submitted": self._submitted,
                "frames_dropped": self._dropped,
                "worker_errors": self._errors,
                "last_worker_error": self._last_error,
            }
        return {**self.provider.diagnostics(), "worker": worker}


def create_telemetry_worker(
    provider_name: str, config: HudTelemetryConfig | None
) -> TelemetryWorker:
    if provider_name == "zero":
        return TelemetryWorker(ZeroTelemetryProvider())
    if provider_name == "hud_telemetry" and config is not None:
        return TelemetryWorker(HudTelemetryProvider(config))
    raise ValueError("invalid telemetry provider configuration")
