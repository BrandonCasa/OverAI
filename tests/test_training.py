from __future__ import annotations

import tempfile
import unittest
from pathlib import Path

import torch

from overai.config import ModelConfig
from overai.data import DemonstrationWindowDataset, dataset_summary
from overai.model import HierarchicalImitationController
from overai.synthetic import create_synthetic_dataset
from overai.training import (
    TrainingConfig,
    load_checkpoint,
    save_checkpoint,
    train_sequence_batch,
)


class TrainingPipelineTests(unittest.TestCase):
    def test_synthetic_dataset_and_cpu_train_step(self) -> None:
        cfg = ModelConfig.tiny()
        with tempfile.TemporaryDirectory() as temporary:
            manifest = create_synthetic_dataset(Path(temporary), cfg, seconds=4.0)
            self.assertEqual(dataset_summary(manifest, cfg)["episodes"], 1)
            dataset = DemonstrationWindowDataset(
                manifest,
                cfg,
                history_seconds=0.0,
                optimization_seconds=1.0,
                stride_seconds=1.0,
            )
            batch = dataset.collate([dataset[0]])
            self.assertEqual(batch.process_ticks, cfg.fast_hz)
            self.assertEqual(batch.movement.shape[-1], 2)
            self.assertEqual(
                batch.health.shape[1], cfg.slow_hz + cfg.slow_horizon - 1
            )
            first_context = batch.observation_context(0, torch.device("cpu"))
            initial_actions = batch.executed_actions(0, torch.device("cpu"))
            self.assertTrue(torch.equal(initial_actions.axes, torch.zeros_like(initial_actions.axes)))
            self.assertTrue(
                torch.equal(initial_actions.movement, torch.ones_like(initial_actions.movement))
            )
            self.assertTrue(
                torch.equal(initial_actions.buttons, torch.zeros_like(initial_actions.buttons))
            )
            held_context = batch.observation_context(
                cfg.fast_ticks_per_slow - 1, torch.device("cpu")
            )
            self.assertTrue(torch.equal(first_context.health, held_context.health))
            next_context = batch.observation_context(
                cfg.fast_ticks_per_slow, torch.device("cpu")
            )
            self.assertFalse(torch.equal(first_context.health, next_context.health))
            timing = batch.timing_context(0, torch.device("cpu"))
            self.assertEqual(timing.absolute_time.dtype, torch.float32)
            self.assertEqual(timing.since_video_frame.dtype, torch.float32)
            self.assertEqual(timing.since_slow_update.dtype, torch.float32)
            self.assertEqual(timing.fast_delta_time.dtype, torch.float32)
            later_timing = batch.timing_context(1, torch.device("cpu"))
            self.assertEqual(later_timing.fast_delta_time.dtype, torch.float32)
            self.assertEqual(
                tuple(batch.load_frame(0, torch.device("cpu")).shape),
                (1, 2, cfg.image_height, cfg.image_width),
            )
            later_batch = dataset.collate([dataset[1]])
            previous = later_batch.executed_actions(0, torch.device("cpu"))
            self.assertFalse(torch.equal(previous.axes, later_batch.axes[:, 0]))
            model = HierarchicalImitationController(cfg)
            optimizer = torch.optim.AdamW(model.parameters(), lr=1e-4)
            result = train_sequence_batch(
                model,
                batch,
                optimizer,
                torch.device("cpu"),
                tbptt_ticks=2,
                use_bf16=False,
            )
            self.assertTrue(torch.isfinite(torch.tensor(result.mean_loss)))
            self.assertEqual(result.optimizer_steps, 2)

    def test_manifest_validation_rejects_non_finite_telemetry(self) -> None:
        cfg = ModelConfig.tiny()
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            manifest = create_synthetic_dataset(root, cfg, seconds=4.0)
            controls_path = root / "episodes" / "synthetic-001" / "controls.pt"
            controls = torch.load(controls_path, weights_only=True)
            controls["health"][0, 0] = float("nan")
            torch.save(controls, controls_path)
            with self.assertRaisesRegex(ValueError, "health contains non-finite"):
                dataset_summary(manifest, cfg)

    def test_manifest_discards_one_unpaired_terminal_fast_tick(self) -> None:
        cfg = ModelConfig.tiny()
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            manifest = create_synthetic_dataset(root, cfg, seconds=4.0)
            controls_path = root / "episodes" / "synthetic-001" / "controls.pt"
            controls = torch.load(controls_path, weights_only=True)
            controls["fast_timestamps"] = torch.cat(
                (controls["fast_timestamps"], controls["fast_timestamps"][-1:] + 0.25)
            )
            controls["axes"] = torch.cat((controls["axes"], controls["axes"][-1:]))
            torch.save(controls, controls_path)

            summary = dataset_summary(manifest, cfg)
            self.assertEqual(summary["fast_ticks"], 16)
            self.assertEqual(summary["discarded_terminal_fast_ticks"], 1)

            controls = torch.load(controls_path, weights_only=True)
            controls["fast_timestamps"] = torch.cat(
                (controls["fast_timestamps"], controls["fast_timestamps"][-1:] + 0.25)
            )
            controls["axes"] = torch.cat((controls["axes"], controls["axes"][-1:]))
            torch.save(controls, controls_path)
            with self.assertRaisesRegex(ValueError, "complete video interval"):
                dataset_summary(manifest, cfg)

    def test_checkpoint_resume_repeats_partial_epoch_and_validates_calibration(self) -> None:
        cfg = ModelConfig.tiny()
        training_cfg = TrainingConfig(epochs=4, num_workers=0, bf16=False)
        axis_normalization = {
            "method": "percentile_counts_per_second",
            "percentile": 99.5,
            "scale_counts_per_second": [10.0, 20.0],
        }
        telemetry = {"provider": "zero", "sha256": None}
        with tempfile.TemporaryDirectory() as temporary:
            checkpoint = Path(temporary) / "checkpoint.pt"
            source = HierarchicalImitationController(cfg)
            source_optimizer = torch.optim.AdamW(source.parameters(), lr=1e-4)
            save_checkpoint(
                checkpoint,
                source,
                source_optimizer,
                epoch=2,
                global_step=17,
                training_cfg=training_cfg,
                axis_normalization=axis_normalization,
                control_profile_sha256="profile-a",
                telemetry=telemetry,
                epoch_complete=False,
            )
            target = HierarchicalImitationController(cfg)
            target_optimizer = torch.optim.AdamW(target.parameters(), lr=1e-4)
            self.assertEqual(
                load_checkpoint(
                    checkpoint,
                    target,
                    target_optimizer,
                    axis_normalization=axis_normalization,
                    control_profile_sha256="profile-a",
                    telemetry=telemetry,
                ),
                (2, 17),
            )

            mismatches = (
                ({**axis_normalization, "scale_counts_per_second": [1.0, 2.0]}, "profile-a", telemetry, "axis normalization"),
                (axis_normalization, "profile-b", telemetry, "control profile"),
                (axis_normalization, "profile-a", {"provider": "hud_telemetry", "sha256": "other"}, "telemetry"),
            )
            for axes, profile_hash, telemetry_manifest, message in mismatches:
                with self.subTest(message=message), self.assertRaisesRegex(ValueError, message):
                    load_checkpoint(
                        checkpoint,
                        target,
                        target_optimizer,
                        axis_normalization=axes,
                        control_profile_sha256=profile_hash,
                        telemetry=telemetry_manifest,
                    )

            payload = torch.load(checkpoint, weights_only=True)
            payload["epoch_complete"] = True
            torch.save(payload, checkpoint)
            self.assertEqual(
                load_checkpoint(
                    checkpoint,
                    target,
                    target_optimizer,
                    axis_normalization=axis_normalization,
                    control_profile_sha256="profile-a",
                    telemetry=telemetry,
                ),
                (3, 17),
            )


if __name__ == "__main__":
    unittest.main()
