from __future__ import annotations

import json
import tempfile
import threading
import time
import unittest
from dataclasses import replace
from pathlib import Path
from unittest.mock import patch

import torch
from torchvision.io import write_jpeg

from overai.config import ModelConfig
from overai.recording import (
    AxisDenormalizer,
    AxisNormalization,
    ControlProfile,
    EpisodeRecorder,
    _validate_episode,
    finalize_dataset,
)


class _SyntheticBackend:
    def __init__(
        self,
        cfg: ModelConfig,
        *,
        fail_capture: bool = False,
        pause_after_first: bool = True,
        initial_capture_delay: float = 0.0,
    ) -> None:
        self.cfg = cfg
        self.fail_capture = fail_capture
        self.pause_after_first = pause_after_first
        self.initial_capture_delay = initial_capture_delay
        self.held_calls = 0
        self.stopped = False
        self.frame_timeouts: list[int] = []
        self.capture_calls = 0

    def start(self) -> None:
        self.stopped = False

    def stop(self) -> None:
        self.stopped = True

    def latest_frame(self, timeout_ms: int):
        self.frame_timeouts.append(timeout_ms)
        if self.fail_capture:
            return None
        if self.capture_calls == 0 and self.initial_capture_delay:
            time.sleep(self.initial_capture_delay)
        self.capture_calls += 1
        return (
            time.perf_counter(),
            torch.zeros(
                2,
                self.cfg.image_height,
                self.cfg.image_width,
                dtype=torch.uint8,
            ),
        )

    def drain_mouse_deltas(self, start: float, end: float) -> tuple[int, int]:
        return 3, -2

    def held_inputs(self) -> set[str]:
        self.held_calls += 1
        if not self.pause_after_first:
            return set()
        return set() if self.held_calls == 1 else {"F8"}

    def target_active(self) -> bool:
        return True

    def emergency_stop_requested(self) -> bool:
        return False


class _FrameAvailableOnlyWhenWaitedForBackend(_SyntheticBackend):
    def latest_frame(self, timeout_ms: int):
        if self.capture_calls > 0 and timeout_ms == 0:
            self.frame_timeouts.append(timeout_ms)
            return None
        return super().latest_frame(timeout_ms)


class RecordingTests(unittest.TestCase):
    def _profile_path(self, root: Path) -> Path:
        path = root / "profile.json"
        path.write_text(
            json.dumps(
                {
                    "window": {"process_name": "game.exe", "title_regex": "^Game"},
                    "movement": {
                        "left": "A",
                        "right": "D",
                        "forward": "W",
                        "reverse": "S",
                    },
                    "buttons": ["MOUSE1", "MOUSE2", "SPACE", "E", "Q", "SHIFT"],
                    "pause_key": "F8",
                    "emergency_stop_key": "F9",
                    "invert_axes": [False, True],
                }
            ),
            encoding="utf-8",
        )
        return path

    def test_control_profile_and_movement(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            profile = ControlProfile.from_json(self._profile_path(Path(temporary)))
            self.assertEqual(len(profile.buttons), 6)
            self.assertEqual(
                EpisodeRecorder._movement({"A", "W"}, profile.movement), (0, 2)
            )
            self.assertEqual(
                EpisodeRecorder._movement({"A", "D", "S"}, profile.movement),
                (1, 0),
            )

    def test_control_profile_accepts_any_positive_button_count(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            path = self._profile_path(Path(temporary))
            payload = json.loads(path.read_text(encoding="utf-8"))
            payload["buttons"] = [f"BUTTON_{index}" for index in range(11)]
            path.write_text(json.dumps(payload), encoding="utf-8")
            self.assertEqual(len(ControlProfile.from_json(path).buttons), 11)

    def test_control_profile_rejects_removed_fields(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            path = self._profile_path(Path(temporary))
            payload = json.loads(path.read_text(encoding="utf-8"))
            payload["telemetry_provider"] = "zero"
            path.write_text(json.dumps(payload), encoding="utf-8")
            with self.assertRaisesRegex(ValueError, "unsupported fields"):
                ControlProfile.from_json(path)

    def test_axis_calibration_and_inverse_residual(self) -> None:
        raw = torch.tensor([[1, 0], [2, -1], [-3, 4]], dtype=torch.int32)
        durations = torch.full((3,), 0.1)
        normalization = AxisNormalization.derive(raw, durations)
        axes = normalization.normalize(raw, durations)
        self.assertTrue(torch.all(axes.abs() <= 1))
        self.assertGreater(normalization.scale_counts_per_second[0], 20)
        inverse = AxisDenormalizer(normalization, invert_axes=(False, True))
        first = inverse.convert(torch.tensor([0.5, 0.5]), 0.01)
        second = inverse.convert(torch.tensor([0.5, 0.5]), 0.01)
        self.assertGreaterEqual(first[0] + second[0], 0)
        self.assertLessEqual(first[1] + second[1], 0)

    def test_recorder_closes_pause_segment_atomically(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            cfg = replace(ModelConfig.tiny(), num_buttons=6)
            profile = ControlProfile.from_json(self._profile_path(root))
            backend = _SyntheticBackend(cfg)
            episode = EpisodeRecorder(
                backend, profile, cfg, root, "train", "paused-1"
            ).record(duration_seconds=None)
            controls = torch.load(
                episode / "controls.pt", map_location="cpu", weights_only=True
            )
            self.assertEqual(tuple(controls["raw_mouse_deltas"].shape), (1, 2))
            self.assertTrue((episode / "frames_r" / "000000.jpg").is_file())
            self.assertFalse(episode.with_name("paused-1.recording").exists())
            self.assertTrue(backend.stopped)
            self.assertEqual(backend.frame_timeouts, [2000])
            metadata = json.loads((episode / "episode.json").read_text(encoding="utf-8"))
            self.assertNotIn("telemetry", metadata)
            self.assertFalse(
                {"health", "damage_events", "kill_events", "charge"} & controls.keys()
            )

    def test_capture_failure_removes_temporary_segment(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            cfg = replace(ModelConfig.tiny(), num_buttons=6)
            profile = ControlProfile.from_json(self._profile_path(root))
            backend = _SyntheticBackend(cfg, fail_capture=True)
            with self.assertRaisesRegex(RuntimeError, "no fresh"):
                EpisodeRecorder(
                    backend, profile, cfg, root, "train", "failed-1"
                ).record(duration_seconds=0.25)
            self.assertFalse((root / "episodes" / "failed-1.recording").exists())

    def test_initial_capture_delay_does_not_consume_recording_ticks(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            cfg = replace(
                ModelConfig.tiny(),
                num_buttons=6,
                fast_hz=60,
                video_hz=30,
                slow_hz=10,
            )
            profile = ControlProfile.from_json(self._profile_path(root))
            backend = _SyntheticBackend(
                cfg, pause_after_first=False, initial_capture_delay=0.1
            )
            episode = EpisodeRecorder(
                backend, profile, cfg, root, "train", "delayed-1"
            ).record(duration_seconds=0.2)
            controls = torch.load(episode / "controls.pt", weights_only=True)
            self.assertEqual(controls["fast_timestamps"].shape[0], 12)
            self.assertLess(float(controls["fast_timestamps"][0]), 0.03)
            self.assertAlmostEqual(float(controls["frame_timestamps"][0]), 0.0, places=4)
            self.assertEqual(backend.frame_timeouts[0], 2000)
            self.assertTrue(all(timeout == 34 for timeout in backend.frame_timeouts[1:]))

    def test_subsequent_capture_waits_for_wgc_callback_jitter(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            cfg = replace(
                ModelConfig.tiny(),
                num_buttons=6,
                fast_hz=60,
                video_hz=30,
                slow_hz=10,
            )
            profile = ControlProfile.from_json(self._profile_path(root))
            backend = _FrameAvailableOnlyWhenWaitedForBackend(
                cfg, pause_after_first=False
            )
            episode = EpisodeRecorder(
                backend, profile, cfg, root, "train", "wgc-jitter"
            ).record(duration_seconds=0.2)
            controls = torch.load(episode / "controls.pt", weights_only=True)
            self.assertEqual(controls["fast_timestamps"].shape[0], 12)
            self.assertEqual(controls["frame_timestamps"].shape[0], 6)
            self.assertEqual(backend.frame_timeouts, [2000, 34, 34, 34, 34, 34])

    def test_jpeg_worker_failure_cannot_deadlock_recorder_cleanup(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            cfg = replace(
                ModelConfig.tiny(),
                num_buttons=6,
                fast_hz=1000,
                video_hz=1000,
                slow_hz=1000,
            )
            profile = ControlProfile.from_json(self._profile_path(root))
            backend = _SyntheticBackend(cfg, pause_after_first=False)
            errors: list[BaseException] = []

            def delayed_failure(*_args, **_kwargs) -> None:
                time.sleep(0.05)
                raise OSError("synthetic JPEG failure")

            def run() -> None:
                try:
                    EpisodeRecorder(
                        backend, profile, cfg, root, "train", "jpeg-failed"
                    ).record(duration_seconds=0.1)
                except BaseException as error:
                    errors.append(error)

            with patch("overai.recording.write_jpeg", side_effect=delayed_failure):
                recorder_thread = threading.Thread(target=run)
                recorder_thread.start()
                recorder_thread.join(timeout=2.0)
            self.assertFalse(recorder_thread.is_alive(), "recorder cleanup deadlocked")
            self.assertEqual(len(errors), 1)
            self.assertIsInstance(errors[0], RuntimeError)
            self.assertFalse((root / "episodes" / "jpeg-failed.recording").exists())

    def test_finalization_rejects_legacy_hud_controls(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            episode = Path(temporary) / "episode"
            (episode / "frames_r").mkdir(parents=True)
            (episode / "frames_b").mkdir()
            write_jpeg(
                torch.zeros(1, 4, 4, dtype=torch.uint8),
                str(episode / "frames_r" / "000000.jpg"),
            )
            write_jpeg(
                torch.zeros(1, 4, 4, dtype=torch.uint8),
                str(episode / "frames_b" / "000000.jpg"),
            )
            controls = {
                "fast_timestamps": torch.tensor([0.0]),
                "frame_timestamps": torch.tensor([0.0]),
                "slow_timestamps": torch.tensor([0.0]),
                "raw_mouse_deltas": torch.zeros(1, 2, dtype=torch.int32),
                "fast_durations": torch.ones(1),
                "axes": torch.zeros(1, 2),
                "health": torch.zeros(1, 1),
                "movement": torch.ones(1, 2, dtype=torch.long),
                "buttons": torch.zeros(1, 6, dtype=torch.uint8),
            }
            torch.save(controls, episode / "controls.pt")
            with self.assertRaisesRegex(ValueError, "dataset format 3"):
                _validate_episode(episode)

    def test_finalize_uses_training_scale_for_both_splits(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            profile = ControlProfile.from_json(self._profile_path(root))
            for split, episode_id, multiplier in (
                ("train", "train-1", 1),
                ("validation", "validation-1", 10),
            ):
                episode = root / split / "episodes" / episode_id
                (episode / "frames_r").mkdir(parents=True)
                (episode / "frames_b").mkdir()
                write_jpeg(torch.zeros(1, 4, 4, dtype=torch.uint8), str(episode / "frames_r" / "000000.jpg"))
                write_jpeg(torch.zeros(1, 4, 4, dtype=torch.uint8), str(episode / "frames_b" / "000000.jpg"))
                controls = {
                    "fast_timestamps": torch.tensor([0.0, 0.1], dtype=torch.float64),
                    "frame_timestamps": torch.tensor([0.0], dtype=torch.float64),
                    "slow_timestamps": torch.tensor([0.0], dtype=torch.float64),
                    "raw_mouse_deltas": torch.tensor(
                        [[multiplier, 0], [2 * multiplier, multiplier]],
                        dtype=torch.int32,
                    ),
                    "fast_durations": torch.tensor([0.1, 0.1]),
                    "axes": torch.zeros(2, 2),
                    "movement": torch.ones(1, 2, dtype=torch.long),
                    "buttons": torch.zeros(1, 11, dtype=torch.uint8),
                }
                torch.save(controls, episode / "controls.pt")
                (episode / "episode.json").write_text(
                    json.dumps(
                        {
                            "id": episode_id,
                            "split": split,
                            "profile_sha256": profile.sha256(),
                            "finalized": False,
                        }
                    ),
                    encoding="utf-8",
                )
            normalization = finalize_dataset(root / "train", root / "validation")
            validation = torch.load(
                root / "validation" / "episodes" / "validation-1" / "controls.pt",
                weights_only=True,
            )
            self.assertTrue(torch.all(validation["axes"].abs() <= 1))
            manifest = json.loads(
                (root / "validation" / "validation.json").read_text(encoding="utf-8")
            )
            self.assertEqual(
                manifest["axis_normalization"]["scale_counts_per_second"],
                list(normalization.scale_counts_per_second),
            )
            self.assertNotIn("telemetry", manifest)
            self.assertEqual(manifest["version"], 3)
            self.assertEqual(manifest["num_buttons"], 11)


if __name__ == "__main__":
    unittest.main()
