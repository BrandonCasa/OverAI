from __future__ import annotations

import tempfile
import unittest
from dataclasses import replace
from pathlib import Path

import torch

from overai.config import ModelConfig
from overai.data import DemonstrationWindowDataset
from overai.distillation import (
    DistillationConfig,
    load_teacher_checkpoint,
    train_distillation_batch,
    validate_distillation_configs,
)
from overai.model import HierarchicalImitationController
from overai.synthetic import create_synthetic_dataset
from overai.training import TrainingConfig, save_checkpoint


class DistillationTests(unittest.TestCase):
    @staticmethod
    def _student_config() -> ModelConfig:
        return replace(
            ModelConfig.tiny(),
            vision_dim=16,
            model_dim=32,
            controller_dim=32,
            vision_layers=1,
            action_dim=8,
        )

    def test_configs_must_preserve_observable_contract(self) -> None:
        teacher = ModelConfig.tiny()
        validate_distillation_configs(teacher, self._student_config())
        with self.assertRaisesRegex(ValueError, "fast_horizon"):
            validate_distillation_configs(
                teacher, replace(self._student_config(), fast_horizon=8)
            )

    def test_cpu_distillation_step_updates_only_student(self) -> None:
        teacher_cfg = ModelConfig.tiny()
        student_cfg = self._student_config()
        with tempfile.TemporaryDirectory() as temporary:
            manifest = create_synthetic_dataset(
                Path(temporary), teacher_cfg, seconds=4.0
            )
            dataset = DemonstrationWindowDataset(
                manifest,
                student_cfg,
                history_seconds=0.0,
                optimization_seconds=1.0,
                stride_seconds=1.0,
            )
            batch = dataset.collate([dataset[0]])
            teacher = HierarchicalImitationController(teacher_cfg)
            student = HierarchicalImitationController(student_cfg)
            optimizer = torch.optim.AdamW(student.parameters(), lr=1e-4)
            teacher_before = {
                name: parameter.detach().clone()
                for name, parameter in teacher.named_parameters()
            }
            student_before = {
                name: parameter.detach().clone()
                for name, parameter in student.named_parameters()
            }

            result = train_distillation_batch(
                teacher,
                student,
                batch,
                optimizer,
                torch.device("cpu"),
                tbptt_ticks=2,
                use_bf16=False,
                distillation_cfg=DistillationConfig(),
            )

            self.assertTrue(torch.isfinite(torch.tensor(result.mean_loss)))
            self.assertGreater(result.metrics["teacher_loss"], 0.0)
            self.assertEqual(result.optimizer_steps, 2)
            for name, parameter in teacher.named_parameters():
                self.assertTrue(torch.equal(parameter, teacher_before[name]))
                self.assertIsNone(parameter.grad)
            self.assertTrue(
                any(
                    not torch.equal(parameter, student_before[name])
                    for name, parameter in student.named_parameters()
                )
            )

    def test_teacher_checkpoint_validates_dataset_identity(self) -> None:
        teacher_cfg = ModelConfig.tiny()
        student_cfg = self._student_config()
        axes = {
            "method": "percentile_counts_per_second",
            "percentile": 99.5,
            "scale_counts_per_second": [10.0, 20.0],
        }
        with tempfile.TemporaryDirectory() as temporary:
            checkpoint = Path(temporary) / "teacher.pt"
            teacher = HierarchicalImitationController(teacher_cfg)
            optimizer = torch.optim.AdamW(teacher.parameters(), lr=1e-4)
            save_checkpoint(
                checkpoint,
                teacher,
                optimizer,
                epoch=0,
                global_step=1,
                training_cfg=TrainingConfig(num_workers=0, bf16=False),
                axis_normalization=axes,
                control_profile_sha256="profile-a",
                epoch_complete=True,
                batches_completed_in_epoch=1,
                data_loader_generator_state=torch.Generator().get_state(),
            )
            loaded, _ = load_teacher_checkpoint(
                checkpoint,
                student_cfg,
                torch.device("cpu"),
                axis_normalization=axes,
                control_profile_sha256="profile-a",
            )
            self.assertFalse(loaded.training)
            self.assertTrue(all(not parameter.requires_grad for parameter in loaded.parameters()))
            with self.assertRaisesRegex(ValueError, "control profile"):
                load_teacher_checkpoint(
                    checkpoint,
                    student_cfg,
                    torch.device("cpu"),
                    axis_normalization=axes,
                    control_profile_sha256="profile-b",
                )


if __name__ == "__main__":
    unittest.main()
