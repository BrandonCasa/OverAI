from __future__ import annotations

import unittest
from dataclasses import replace
from pathlib import Path

from overai.config import ModelConfig


class ModelConfigTests(unittest.TestCase):
    def test_defaults_match_current_720p_io(self) -> None:
        cfg = ModelConfig()
        self.assertEqual((cfg.image_height, cfg.image_width), (720, 1280))
        self.assertEqual((cfg.grid_height, cfg.grid_width), (18, 32))
        self.assertEqual(cfg.grid_tokens, 576)
        self.assertEqual(cfg.input_channels, 2)
        self.assertEqual(cfg.num_buttons, 8)
        self.assertEqual(cfg.control_query_tokens, 2 + cfg.num_buttons + 2)
        self.assertEqual(cfg.memory_tokens, 360)
        self.assertEqual(cfg.slow_horizon / cfg.slow_hz, 2.0)
        self.assertEqual(cfg.fast_horizon / cfg.fast_hz, 2.0)
        root = Path(__file__).resolve().parents[1]
        self.assertEqual(
            cfg,
            ModelConfig.from_json(root / "configs" / "rtx4080_720p_new.json"),
        )

    def test_distillation_target_matches_original_hardware_profile(self) -> None:
        root = Path(__file__).resolve().parents[1]
        student = ModelConfig.from_json(root / "configs" / "rtx4080_720p_new.json")
        h100 = ModelConfig.from_json(root / "configs" / "h100_720p.json")
        self.assertEqual(
            replace(student, gradient_checkpointing=False),
            h100,
        )

    def test_teacher_profile_is_larger_but_io_compatible(self) -> None:
        root = Path(__file__).resolve().parents[1]
        teacher = ModelConfig.from_json(root / "configs" / "rtx4080_720p.json")
        student = ModelConfig.from_json(root / "configs" / "rtx4080_720p_new.json")
        self.assertGreater(teacher.model_dim, student.model_dim)
        self.assertGreater(teacher.vision_layers, student.vision_layers)
        for field in (
            "image_height",
            "image_width",
            "input_channels",
            "channel_order",
            "video_hz",
            "slow_hz",
            "fast_hz",
            "slow_horizon",
            "fast_horizon",
            "num_buttons",
        ):
            self.assertEqual(getattr(teacher, field), getattr(student, field))

    def test_rejects_inefficient_window_larger_than_grid(self) -> None:
        with self.assertRaisesRegex(ValueError, "attention windows"):
            replace(ModelConfig.tiny(), window_height=5)

    def test_rejects_invalid_position_encoding_width(self) -> None:
        with self.assertRaisesRegex(ValueError, "divisible by four"):
            replace(ModelConfig.tiny(), vision_dim=30, num_heads=2)


if __name__ == "__main__":
    unittest.main()
