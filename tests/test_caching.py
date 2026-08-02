from __future__ import annotations

import tempfile
import unittest
from pathlib import Path

import torch

from overai.data import CONTROL_KEYS, _load_controls, _load_controls_cached
from overai.losses import horizon_weights


class FunctoolsCachingTests(unittest.TestCase):
    def tearDown(self) -> None:
        _load_controls_cached.cache_clear()
        horizon_weights.cache_clear()

    def test_horizon_weights_reuse_identical_tensor(self) -> None:
        horizon_weights.cache_clear()
        device = torch.device("cpu")
        first = horizon_weights(120, device, torch.float32)
        second = horizon_weights(120, device, torch.float32)

        self.assertIs(first, second)
        self.assertEqual(horizon_weights.cache_info().hits, 1)
        self.assertAlmostEqual(float(first.mean()), 1.0, places=6)

    def test_control_cache_invalidates_when_file_changes(self) -> None:
        _load_controls_cached.cache_clear()
        with tempfile.TemporaryDirectory() as temporary:
            path = Path(temporary) / "controls.pt"

            def save(axis_value: float) -> None:
                controls = {
                    key: torch.tensor([axis_value]) for key in CONTROL_KEYS
                }
                torch.save(controls, path)

            save(1.0)
            first = _load_controls(path)
            second = _load_controls(path)
            self.assertIs(first, second)
            self.assertEqual(_load_controls_cached.cache_info().hits, 1)

            save(2.0)
            refreshed = _load_controls(path)
            self.assertIsNot(first, refreshed)
            self.assertEqual(float(refreshed["axes"].item()), 2.0)


if __name__ == "__main__":
    unittest.main()
