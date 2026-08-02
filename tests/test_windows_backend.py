from __future__ import annotations

import threading
import unittest

from overai.windows_backend import WindowsCaptureBackend


class RawInputStartupTests(unittest.TestCase):
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
