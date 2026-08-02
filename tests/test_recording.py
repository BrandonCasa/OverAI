from __future__ import annotations

import json
import tempfile
import time
import unittest
from pathlib import Path

import torch
from torchvision.io import write_jpeg

from overai.config import ModelConfig
from overai.recording import (
    AxisDenormalizer,
    AxisNormalization,
    ControlProfile,
    EpisodeRecorder,
    finalize_dataset,
)


class _SyntheticBackend:
    def __init__(self, cfg: ModelConfig, *, fail_capture: bool = False) -> None:
        self.cfg = cfg
        self.fail_capture = fail_capture
        self.held_calls = 0
        self.stopped = False

    def start(self) -> None:
        self.stopped = False

    def stop(self) -> None:
        self.stopped = True

    def latest_frame(self, timeout_ms: int):
        if self.fail_capture:
            return None
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
        return set() if self.held_calls == 1 else {"F8"}

    def target_active(self) -> bool:
        return True

    def emergency_stop_requested(self) -> bool:
        return False


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
                    "telemetry_provider": "zero",
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
            cfg = ModelConfig.tiny()
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
            metadata = json.loads((episode / "episode.json").read_text(encoding="utf-8"))
            self.assertEqual(metadata["telemetry"]["provider"], "zero")
            self.assertEqual(metadata["telemetry"]["diagnostics"]["provider"], "zero")

    def test_capture_failure_removes_temporary_segment(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            cfg = ModelConfig.tiny()
            profile = ControlProfile.from_json(self._profile_path(root))
            backend = _SyntheticBackend(cfg, fail_capture=True)
            with self.assertRaisesRegex(RuntimeError, "no fresh"):
                EpisodeRecorder(
                    backend, profile, cfg, root, "train", "failed-1"
                ).record(duration_seconds=0.25)
            self.assertFalse((root / "episodes" / "failed-1.recording").exists())

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
                    "health": torch.zeros(1, 1),
                    "damage_events": torch.zeros(1, 1),
                    "kill_events": torch.zeros(1, 1),
                    "charge": torch.zeros(1, 1),
                    "movement": torch.ones(1, 2, dtype=torch.long),
                    "buttons": torch.zeros(1, 6, dtype=torch.uint8),
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
            self.assertEqual(manifest["telemetry"]["provider"], "zero")


if __name__ == "__main__":
    unittest.main()
