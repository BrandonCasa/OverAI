from __future__ import annotations

import tempfile
import unittest
from pathlib import Path

import torch

from overai.config import ModelConfig
from overai.data import DemonstrationWindowDataset, dataset_summary
from overai.model import HierarchicalImitationController
from overai.synthetic import create_synthetic_dataset
from overai.training import train_sequence_batch


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
            held_context = batch.observation_context(
                cfg.fast_ticks_per_slow - 1, torch.device("cpu")
            )
            self.assertTrue(torch.equal(first_context.health, held_context.health))
            next_context = batch.observation_context(
                cfg.fast_ticks_per_slow, torch.device("cpu")
            )
            self.assertFalse(torch.equal(first_context.health, next_context.health))
            self.assertEqual(
                tuple(batch.load_frame(0, torch.device("cpu")).shape),
                (1, 3, cfg.image_height, cfg.image_width),
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


if __name__ == "__main__":
    unittest.main()
