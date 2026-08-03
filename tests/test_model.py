from __future__ import annotations

import unittest

import torch

from overai.blocks import FourierTimeEmbedding
from overai.config import ModelConfig
from overai.losses import fast_axis_loss, slow_control_loss
from overai.model import (
    HierarchicalImitationController,
    RuntimeController,
    derive_button_events,
)
from overai.types import (
    ExecutedActions,
    FastTargets,
    SlowTargets,
    TimingContext,
)


class ControllerTests(unittest.TestCase):
    def setUp(self) -> None:
        torch.manual_seed(7)
        self.cfg = ModelConfig.tiny()
        self.model = HierarchicalImitationController(self.cfg)
        self.actions = ExecutedActions(
            movement=torch.ones(1, 2, dtype=torch.long),
            buttons=torch.zeros(1, self.cfg.num_buttons),
            axes=torch.zeros(1, 2),
        )
        self.timing = TimingContext(
            absolute_time=torch.zeros(1, 1),
            since_video_frame=torch.zeros(1, 1),
            since_slow_update=torch.zeros(1, 1),
            fast_delta_time=torch.full((1, 1), 1 / self.cfg.fast_hz),
        )

    def test_fp16_fourier_time_embedding_remains_finite_after_two_seconds(
        self,
    ) -> None:
        embedding = FourierTimeEmbedding(4, 32, frequencies=16).half().eval()
        timestamps = torch.tensor(
            [[2.0, 60.0, 600.0, 3600.0]], dtype=torch.float16
        )
        with torch.inference_mode():
            output = embedding(timestamps)
        self.assertEqual(output.dtype, torch.float16)
        self.assertTrue(torch.isfinite(output).all())

    def test_streaming_shapes_and_gradients(self) -> None:
        state = self.model.initial_state(1, "cpu")
        frame = torch.randint(
            0,
            256,
            (1, self.cfg.input_channels, self.cfg.image_height, self.cfg.image_width),
            dtype=torch.uint8,
        )
        output = self.model.on_video_frame(
            frame,
            self.actions,
            self.timing,
            state,
            run_slow_decoder=True,
        )
        self.assertEqual(tuple(output.fast.immediate_axes.shape), (1, 2))
        self.assertEqual(
            tuple(output.fast.axis_trajectory.shape), (1, self.cfg.fast_horizon, 2)
        )
        self.assertIsNotNone(output.slow)
        assert output.slow is not None
        self.assertEqual(tuple(output.slow.immediate_movement_logits.shape), (1, 2, 3))
        self.assertEqual(
            tuple(output.slow.trajectory_movement_logits.shape),
            (1, self.cfg.slow_horizon, 2, 3),
        )
        self.assertEqual(
            tuple(output.slow.trajectory_button_logits.shape),
            (1, self.cfg.slow_horizon, self.cfg.num_buttons),
        )
        fast_losses = fast_axis_loss(
            output.fast,
            FastTargets(torch.zeros(1, self.cfg.fast_horizon, 2)),
            self.cfg,
        )
        slow_losses = slow_control_loss(
            output.slow,
            SlowTargets(
                movement=torch.ones(
                    1, self.cfg.slow_horizon, 2, dtype=torch.long
                ),
                buttons=torch.zeros(1, self.cfg.slow_horizon, self.cfg.num_buttons),
            ),
            self.cfg,
        )
        loss = torch.stack(tuple(fast_losses.values())).sum()
        loss = loss + torch.stack(tuple(slow_losses.values())).sum()
        self.assertTrue(torch.isfinite(loss))
        loss.backward()
        self.assertIsNotNone(self.model.vision.patch_projection.weight.grad)

    def test_memory_tier_schedule(self) -> None:
        state = self.model.initial_state(1, "cpu")
        frame = torch.zeros(
            1,
            self.cfg.input_channels,
            self.cfg.image_height,
            self.cfg.image_width,
            dtype=torch.uint8,
        )
        for _ in range(4):
            output = self.model.on_video_frame(
                frame,
                self.actions,
                self.timing,
                state,
                run_slow_decoder=False,
            )
            state = output.state
        self.assertEqual(int(state.memory.recent_valid.sum()), 4)
        self.assertEqual(int(state.memory.intermediate_valid.sum()), 2)
        self.assertEqual(int(state.memory.long_valid.sum()), 1)

    def test_rate_derived_runtime_scheduler(self) -> None:
        scheduler = RuntimeController(self.model)
        state = self.model.initial_state(1, "cpu")
        frame = torch.zeros(
            1,
            self.cfg.input_channels,
            self.cfg.image_height,
            self.cfg.image_width,
            dtype=torch.uint8,
        )
        first = scheduler.step(frame, self.actions, self.timing, state)
        self.assertIsNotNone(first.discrete)
        second = scheduler.step(
            None, self.actions, self.timing, first.state
        )
        self.assertIsNone(second.discrete)

    def test_button_events(self) -> None:
        pressed, released = derive_button_events(
            torch.tensor([[0, 1, 1]]), torch.tensor([[1, 1, 0]])
        )
        self.assertTrue(torch.equal(pressed, torch.tensor([[True, False, False]])))
        self.assertTrue(torch.equal(released, torch.tensor([[False, False, True]])))

    def test_invalid_grid_is_rejected(self) -> None:
        with self.assertRaises(ValueError):
            ModelConfig(image_height=100, image_width=100, patch_size=40)


if __name__ == "__main__":
    unittest.main()
