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
            ModelConfig.from_json(root / "configs" / "rtx4080_720p.json"),
        )

    def test_hardware_profiles_share_learned_architecture(self) -> None:
        root = Path(__file__).resolve().parents[1]
        rtx = ModelConfig.from_json(root / "configs" / "rtx4080_720p.json")
        h100 = ModelConfig.from_json(root / "configs" / "h100_720p.json")
        self.assertEqual(
            replace(rtx, gradient_checkpointing=False),
            h100,
        )

    def test_rejects_inefficient_window_larger_than_grid(self) -> None:
        with self.assertRaisesRegex(ValueError, "attention windows"):
            replace(ModelConfig.tiny(), window_height=5)

    def test_rejects_invalid_position_encoding_width(self) -> None:
        with self.assertRaisesRegex(ValueError, "divisible by four"):
            replace(ModelConfig.tiny(), vision_dim=30, num_heads=2)


if __name__ == "__main__":
    unittest.main()
