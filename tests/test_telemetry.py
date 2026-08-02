from __future__ import annotations

import time
import unittest
from dataclasses import replace

import torch

from overai.rtx import _update_telemetry
from overai.telemetry import (
    _GLYPHS,
    SIMILARITY_METRIC,
    CapturedFrame,
    HudTelemetryConfig,
    HudTelemetryProvider,
    OcrResult,
    TelemetrySnapshot,
    TelemetryWorker,
    rgb_similarity,
)


def _config(**overrides) -> HudTelemetryConfig:
    payload = {
        "reference_resolution": [100, 60],
        "ocr": {
            "threshold": 128,
            "scale": 1,
            "allowed_glyphs": "0123456789%",
            "polarity": "light_on_dark",
            "minimum_confidence": 0.72,
        },
        "health": {"bbox": [2, 2, 40, 9], "maximum": 200, "max_stale_seconds": 0.5},
        "charge": {"bbox": [2, 16, 40, 9], "max_stale_seconds": 0.5},
        "hitmarker": {
            "points": [[55, 5], [60, 5], [55, 10], [60, 10]],
            "colors": [[255, 20, 10], [10, 240, 20]],
            "debounce_seconds": 0,
            "rearm_seconds": 0.05,
        },
        "kill": {
            "points": [[70, 30], [73, 30], [76, 30], [70, 33], [73, 33], [76, 33], [79, 33]],
            "colors": [[20, 30, 250]],
            "debounce_seconds": 0,
            "rearm_seconds": 0.05,
        },
        "similarity_threshold": 0.90,
        "similarity_metric": SIMILARITY_METRIC,
    }
    payload.update(overrides)
    return HudTelemetryConfig.from_mapping(payload)


def _surface(width: int = 100, height: int = 60) -> torch.Tensor:
    result = torch.zeros(4, height, width, dtype=torch.uint8)
    result[3].fill_(255)
    return result


def _set_rgb(frame: torch.Tensor, x: int, y: int, color: tuple[int, int, int]) -> None:
    frame[0, y, x] = color[2]
    frame[1, y, x] = color[1]
    frame[2, y, x] = color[0]


def _draw_text(
    frame: torch.Tensor, bbox: tuple[int, int, int, int], text: str, *, pixel_scale: int = 1
) -> None:
    x, y, _, _ = bbox
    cursor = x
    for char in text:
        pattern = _GLYPHS[char]
        for row, values in enumerate(pattern):
            for column, value in enumerate(values):
                if value == "1":
                    frame[
                        0:3,
                        y + row * pixel_scale : y + (row + 1) * pixel_scale,
                        cursor + column * pixel_scale : cursor + (column + 1) * pixel_scale,
                    ] = 255
        cursor += (len(pattern[0]) + 1) * pixel_scale


def _mark_event(frame: torch.Tensor, points, color) -> None:
    for x, y in points:
        _set_rgb(frame, x, y, color)


class HudTelemetryTests(unittest.TestCase):
    def test_health_and_charge_bitmap_ocr_and_normalization(self) -> None:
        config = _config()
        provider = HudTelemetryProvider(config)
        frame = _surface()
        _draw_text(frame, config.health.bbox, "150")
        _draw_text(frame, config.charge.bbox, "87%")
        provider.analyze_frame(1.0, frame)
        sample = provider.sample(1.0)
        self.assertAlmostEqual(sample.health, 0.75)
        self.assertAlmostEqual(sample.charge, 0.87)
        diagnostics = provider.diagnostics()
        self.assertEqual(diagnostics["last_ocr"]["health"]["text"], "150")
        self.assertEqual(diagnostics["last_ocr"]["charge"]["text"], "87%")

    def test_invalid_missing_and_obscured_ocr_retains_then_expires(self) -> None:
        config = _config()
        provider = HudTelemetryProvider(config)
        valid = _surface()
        _draw_text(valid, config.health.bbox, "100")
        _draw_text(valid, config.charge.bbox, "50")
        provider.analyze_frame(1.0, valid)
        obscured = valid.clone()
        x, y, _, height = config.health.bbox
        obscured[0:3, y : y + height, x + 1] = 0
        x, y, width, height = config.charge.bbox
        obscured[:, y : y + height, x : x + width] = 0
        provider.analyze_frame(1.2, obscured)
        stale = provider.sample(1.3)
        self.assertEqual(stale.health, 0.5)
        self.assertEqual(stale.charge, 0.5)
        expired = provider.sample(1.51)
        self.assertEqual(expired.health, 0.0)
        self.assertEqual(expired.charge, 0.0)

    def test_out_of_range_ocr_is_rejected(self) -> None:
        class Results:
            def __init__(self) -> None:
                self.calls = 0

            def read(self, bgra, bbox, config):
                self.calls += 1
                return OcrResult("201" if self.calls % 2 else "101%", 1.0)

        provider = HudTelemetryProvider(_config(), Results())
        provider.analyze_frame(1.0, _surface())
        sample = provider.sample(1.0)
        self.assertEqual(sample.health, 0.0)
        self.assertEqual(sample.charge, 0.0)

    def test_each_hitmarker_point_is_required_and_either_color_matches(self) -> None:
        config = _config()
        for failed_index in range(4):
            provider = HudTelemetryProvider(config)
            frame = _surface()
            for index, point in enumerate(config.hitmarker.points):
                if index != failed_index:
                    _set_rgb(frame, *point, config.hitmarker.colors[0])
            provider.analyze_frame(1.0, frame)
            self.assertEqual(provider.sample(1.0).damage_event, 0.0)
        for color in config.hitmarker.colors:
            provider = HudTelemetryProvider(config)
            frame = _surface()
            _mark_event(frame, config.hitmarker.points, color)
            provider.analyze_frame(1.0, frame)
            self.assertEqual(provider.sample(1.0).damage_event, 1.0)

    def test_each_kill_point_is_required(self) -> None:
        config = _config()
        for failed_index in range(7):
            provider = HudTelemetryProvider(config)
            frame = _surface()
            for index, point in enumerate(config.kill.points):
                if index != failed_index:
                    _set_rgb(frame, *point, config.kill.colors[0])
            provider.analyze_frame(1.0, frame)
            self.assertEqual(provider.sample(1.0).kill_event, 0.0)
        provider = HudTelemetryProvider(config)
        frame = _surface()
        _mark_event(frame, config.kill.points, config.kill.colors[0])
        provider.analyze_frame(1.0, frame)
        self.assertEqual(provider.sample(1.0).kill_event, 1.0)

    def test_similarity_below_at_and_above_threshold(self) -> None:
        reference = (0.0, 0.0, 0.0)
        at = (25.5, 25.5, 25.5)
        self.assertLess(rgb_similarity(reference, (26.0, 26.0, 26.0)), 0.90)
        self.assertAlmostEqual(rgb_similarity(reference, at), 0.90)
        self.assertGreater(rgb_similarity(reference, (25.0, 25.0, 25.0)), 0.90)

    def test_persistent_events_debounce_rearm_latch_and_clear(self) -> None:
        config = _config()
        provider = HudTelemetryProvider(config)
        marked = _surface()
        _mark_event(marked, config.hitmarker.points, config.hitmarker.colors[0])
        _mark_event(marked, config.kill.points, config.kill.colors[0])
        provider.analyze_frame(1.01, marked)
        provider.analyze_frame(1.04, marked)
        latched = provider.sample(1.20)
        self.assertEqual(latched.damage_event, 1.0)
        self.assertEqual(latched.kill_event, 1.0)
        self.assertEqual(provider.sample(1.21).damage_event, 1.0)
        provider.acknowledge(latched)
        self.assertEqual(provider.sample(1.22).damage_event, 0.0)
        clear = _surface()
        provider.analyze_frame(1.23, clear)
        provider.analyze_frame(1.29, clear)
        provider.analyze_frame(1.30, marked)
        second = provider.sample(1.40)
        self.assertEqual(second.damage_event, 1.0)
        self.assertEqual(second.kill_event, 1.0)
        self.assertEqual(provider.diagnostics()["counters"]["damage_events"], 2)

    def test_events_between_slow_samples_are_combined(self) -> None:
        config = _config()
        provider = HudTelemetryProvider(config)
        hit = _surface()
        _mark_event(hit, config.hitmarker.points, config.hitmarker.colors[1])
        provider.analyze_frame(1.01, hit)
        blank = _surface()
        provider.analyze_frame(1.08, blank)
        kill = _surface()
        _mark_event(kill, config.kill.points, config.kill.colors[0])
        provider.analyze_frame(1.15, kill)
        sample = provider.sample(1.20)
        self.assertEqual(sample.damage_event, 1.0)
        self.assertEqual(sample.kill_event, 1.0)

    def test_resolution_normalization_uses_reference_coordinates(self) -> None:
        config = _config()
        provider = HudTelemetryProvider(config)
        frame = _surface(200, 120)
        scaled_bbox = provider._scaled_bbox(config.health.bbox, 200, 120)
        _draw_text(frame, scaled_bbox, "100", pixel_scale=2)
        charge_bbox = provider._scaled_bbox(config.charge.bbox, 200, 120)
        _draw_text(frame, charge_bbox, "25", pixel_scale=2)
        for point in config.hitmarker.points:
            _set_rgb(
                frame,
                *provider._scaled_point(point, 200, 120),
                config.hitmarker.colors[0],
            )
        provider.analyze_frame(1.0, frame)
        sample = provider.sample(1.0)
        self.assertEqual(sample.health, 0.5)
        self.assertEqual(sample.charge, 0.25)
        self.assertEqual(sample.damage_event, 1.0)

    def test_duplicate_capture_timestamp_is_not_analyzed_twice(self) -> None:
        provider = HudTelemetryProvider(_config())
        frame = _surface()
        provider.analyze_frame(1.0, frame)
        provider.analyze_frame(1.0, frame)
        counters = provider.diagnostics()["counters"]
        self.assertEqual(counters["frames_analyzed"], 1)
        self.assertEqual(counters["health_ocr_calls"], 1)
        self.assertEqual(counters["out_of_order_frames"], 1)

    def test_config_hash_and_manifest_cover_similarity_and_ocr(self) -> None:
        config = _config()
        changed = replace(config, similarity_threshold=0.91)
        self.assertNotEqual(config.sha256(), changed.sha256())
        manifest = config.manifest()
        self.assertEqual(manifest["sha256"], config.sha256())
        self.assertEqual(manifest["similarity_metric"], SIMILARITY_METRIC)
        self.assertEqual(manifest["configuration"]["ocr"]["threshold"], 128)

    def test_config_rejects_out_of_bounds_regions(self) -> None:
        with self.assertRaisesRegex(ValueError, "exceeds reference"):
            _config(health={"bbox": [99, 59, 2, 2], "maximum": 100})

    def test_configured_match_debounce_is_honored(self) -> None:
        config = _config()
        config = replace(
            config,
            hitmarker=replace(config.hitmarker, debounce_seconds=0.05),
        )
        provider = HudTelemetryProvider(config)
        frame = _surface()
        _mark_event(frame, config.hitmarker.points, config.hitmarker.colors[0])
        provider.analyze_frame(1.0, frame)
        self.assertEqual(provider.sample(1.0).damage_event, 0.0)
        provider.analyze_frame(1.049, frame)
        self.assertEqual(provider.sample(1.049).damage_event, 0.0)
        provider.analyze_frame(1.05, frame)
        self.assertEqual(provider.sample(1.05).damage_event, 1.0)

    def test_identical_provider_sequence_produces_identical_context(self) -> None:
        config = _config()
        providers = [HudTelemetryProvider(config), HudTelemetryProvider(config)]
        frame = _surface()
        _draw_text(frame, config.health.bbox, "120")
        _draw_text(frame, config.charge.bbox, "75")
        _mark_event(frame, config.hitmarker.points, config.hitmarker.colors[0])
        for provider in providers:
            provider.analyze_frame(1.0, frame.clone())
        self.assertEqual(providers[0].sample(1.2).values(), providers[1].sample(1.2).values())

        worker = TelemetryWorker(providers[0])
        metadata = {
            name: torch.zeros(1, 1)
            for name in ("health", "damage_event", "kill_event", "charge")
        }
        _update_telemetry(metadata, worker, 1.2)
        self.assertAlmostEqual(float(metadata["health"].item()), 0.6)
        self.assertAlmostEqual(float(metadata["charge"].item()), 0.75)
        self.assertEqual(float(metadata["damage_event"].item()), 1.0)
        self.assertEqual(providers[0].sample(1.2).damage_event, 0.0)


class TelemetryWorkerTests(unittest.TestCase):
    def test_worker_failure_does_not_block_submission(self) -> None:
        class FailingProvider:
            def analyze_frame(self, timestamp, bgra):
                raise RuntimeError("synthetic OCR failure")

            def sample(self, timestamp=None):
                return TelemetrySnapshot(timestamp or 0.0, 0, 0, 0, 0)

            def acknowledge(self, snapshot):
                return None

            def diagnostics(self):
                return {"provider": "failing"}

        worker = TelemetryWorker(FailingProvider())
        worker.start()
        start = time.perf_counter()
        worker.submit(CapturedFrame(1.0, torch.empty(2, 1, 1, dtype=torch.uint8), _surface()))
        elapsed = time.perf_counter() - start
        deadline = time.perf_counter() + 1.0
        while worker.diagnostics()["worker"]["worker_errors"] == 0 and time.perf_counter() < deadline:
            time.sleep(0.001)
        self.assertFalse(worker.should_terminate(1.5, None))
        self.assertTrue(worker.should_terminate(1.5, 0.4))
        worker.stop()
        self.assertLess(elapsed, 0.05)
        self.assertEqual(worker.diagnostics()["worker"]["worker_errors"], 1)


if __name__ == "__main__":
    unittest.main()
