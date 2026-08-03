from __future__ import annotations

import threading
import unittest
from dataclasses import replace

import torch

from overai.config import ModelConfig
from overai.windows_backend import WindowsCaptureBackend


class RawInputStartupTests(unittest.TestCase):
    def test_same_aspect_source_resize_is_normalized_without_emergency(self) -> None:
        backend = object.__new__(WindowsCaptureBackend)
        backend.cfg = replace(
            ModelConfig.tiny(),
            image_height=72,
            image_width=128,
            grid_height=9,
            grid_width=16,
        )
        backend._source_size = None
        backend._capture_interruption = None
        backend._last_preprocess_ms = 0.0

        first = backend._preprocess_frame_buffer(
            torch.zeros(180, 320, 4, dtype=torch.uint8)
        )
        second = backend._preprocess_frame_buffer(
            torch.zeros(360, 640, 4, dtype=torch.uint8)
        )

        self.assertIsNotNone(first)
        self.assertIsNotNone(second)
        assert first is not None
        assert second is not None
        self.assertEqual(tuple(first.shape), (2, 72, 128))
        self.assertEqual(tuple(second.shape), (2, 72, 128))
        self.assertEqual(backend._source_size, (360, 640))
        self.assertIsNone(backend.capture_interruption_reason())

    def test_aspect_mismatch_is_skipped_for_bounded_recorder_recovery(self) -> None:
        backend = object.__new__(WindowsCaptureBackend)
        backend.cfg = replace(
            ModelConfig.tiny(),
            image_height=72,
            image_width=128,
            grid_height=9,
            grid_width=16,
        )
        backend._source_size = None
        backend._capture_interruption = None
        backend._last_preprocess_ms = 0.0

        result = backend._preprocess_frame_buffer(
            torch.zeros(100, 100, 4, dtype=torch.uint8)
        )

        self.assertIsNone(result)
        self.assertEqual(
            backend.capture_interruption_reason(), "capture_aspect_mismatch"
        )

    def test_backend_preserves_first_emergency_reason(self) -> None:
        backend = object.__new__(WindowsCaptureBackend)
        backend._stop_lock = threading.Lock()
        backend._emergency = False
        backend._stop_reason = None

        backend._request_stop("capture_closed")
        backend._request_stop("emergency_stop_key")

        self.assertTrue(backend.emergency_stop_requested())
        self.assertEqual(backend.emergency_stop_reason(), "capture_closed")

    def test_input_thread_initialization_error_is_propagated(self) -> None:
        backend = object.__new__(WindowsCaptureBackend)
        backend._input_ready = threading.Event()
        backend._input_error = None
        backend._input_thread = None

        def fail_initialization() -> None:
            backend._input_error = OSError("RegisterRawInputDevices failed")
            backend._input_ready.set()

        backend._input_loop = fail_initialization
        with self.assertRaisesRegex(RuntimeError, "Raw Input initialization failed") as raised:
            backend._start_raw_input()
        self.assertIsInstance(raised.exception.__cause__, OSError)


if __name__ == "__main__":
    unittest.main()
