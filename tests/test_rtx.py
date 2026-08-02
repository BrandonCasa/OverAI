from __future__ import annotations

import unittest
from dataclasses import replace
from typing import Any, cast

import torch

from overai.benchmark import _scheduled_paths
from overai.config import ModelConfig
from overai.rtx import METADATA_INPUT_NAMES, Rtx4080Controller, _update_timing


_STATE_KEYS = (
    "recent",
    "intermediate",
    "long",
    "recent_valid",
    "intermediate_valid",
    "long_valid",
    "fast_hidden",
    "fast_previous_trajectory",
    "previous_axis_trajectory",
    "previous_slow_trajectory",
)


class _FakeEngine:
    def __init__(self, name: str, calls: list[str], num_buttons: int) -> None:
        self.name = name
        self.calls = calls
        self.num_buttons = num_buttons

    def execute(self, inputs: dict[str, torch.Tensor]) -> dict[str, torch.Tensor]:
        self.calls.append(self.name)
        if self.name.startswith("video_"):
            outputs = {f"next_{key}": inputs[key].clone() for key in _STATE_KEYS}
            outputs.update(
                shared_tokens=torch.zeros(1),
                immediate_axes=torch.zeros(1, 2),
                axis_trajectory=torch.zeros(1),
            )
            return outputs
        if self.name == "fast_tick":
            return {
                "next_fast_hidden": inputs["fast_hidden"].clone(),
                "next_fast_previous_trajectory": inputs[
                    "fast_previous_trajectory"
                ].clone(),
                "axis_trajectory": inputs["previous_axis_trajectory"].clone(),
                "immediate_axes": torch.zeros(1, 2),
            }
        return {
            "immediate_movement_logits": torch.zeros(1, 2, 3),
            "immediate_button_logits": torch.zeros(1, self.num_buttons),
            "next_previous_slow_trajectory": torch.ones(1),
        }


def _fake_controller(cfg: ModelConfig) -> tuple[Rtx4080Controller, list[str]]:
    controller = object.__new__(Rtx4080Controller)
    controller.cfg = cfg
    controller.device = torch.device("cpu")
    controller.state = {key: torch.zeros(1) for key in _STATE_KEYS}
    controller.state["shared_tokens"] = torch.zeros(1)
    controller.video_frame_index = 0
    calls: list[str] = []
    controller.engines = cast(
        Any,
        {
            name: _FakeEngine(name, calls, cfg.num_buttons)
            for name in (
                "video_ordinary",
                "video_intermediate",
                "video_long",
                "fast_tick",
                "slow_tick",
            )
        },
    )
    return controller, calls


def _metadata() -> dict[str, torch.Tensor]:
    return {name: torch.zeros(1) for name in METADATA_INPUT_NAMES}


class RtxSchedulingTests(unittest.TestCase):
    def test_graph_metadata_has_no_hud_inputs(self) -> None:
        self.assertEqual(
            METADATA_INPUT_NAMES,
            [
                "movement",
                "buttons",
                "executed_axes",
                "absolute_time",
                "since_video_frame",
                "since_slow_update",
                "fast_delta_time",
            ],
        )

    def test_memory_and_discrete_schedules_use_independent_configured_phases(self) -> None:
        cfg = ModelConfig.tiny()
        controller, calls = _fake_controller(cfg)
        frame = torch.zeros(1)
        metadata = _metadata()
        for tick in range(8):
            controller.step(tick, metadata, frame if tick % 2 == 0 else None)
        self.assertEqual(
            calls,
            [
                "video_ordinary",
                "slow_tick",
                "fast_tick",
                "video_intermediate",
                "fast_tick",
                "video_ordinary",
                "slow_tick",
                "fast_tick",
                "video_long",
                "fast_tick",
            ],
        )

    def test_slow_decoder_runs_on_frame_free_slow_boundary(self) -> None:
        cfg = replace(ModelConfig.tiny(), video_hz=1, slow_hz=2, fast_hz=4)
        controller, calls = _fake_controller(cfg)
        metadata = _metadata()
        controller.step(0, metadata, torch.zeros(1))
        controller.step(1, metadata, None)
        controller.step(2, metadata, None)
        self.assertEqual(
            calls,
            ["video_ordinary", "slow_tick", "fast_tick", "fast_tick", "slow_tick"],
        )

    def test_runtime_timing_uses_configured_fast_rate(self) -> None:
        metadata = _metadata()
        _update_timing(metadata, 1.0, 0.2, 0.4, fast_hz=4)
        self.assertEqual(float(metadata["fast_delta_time"].item()), 0.25)

    def test_benchmark_identifies_every_synchronous_scheduled_path(self) -> None:
        self.assertEqual(
            _scheduled_paths(ModelConfig.tiny()),
            {"video+slow", "video", "fast"},
        )
        frame_free_slow = replace(
            ModelConfig.tiny(), video_hz=1, slow_hz=2, fast_hz=4
        )
        self.assertEqual(
            _scheduled_paths(frame_free_slow),
            {"video+slow", "fast", "fast+slow"},
        )


if __name__ == "__main__":
    unittest.main()
