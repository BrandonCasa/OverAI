"""Windows Graphics Capture plus Raw Input backend for demonstration recording."""

from __future__ import annotations

import ctypes
import os
import re
import threading
import time
from collections import deque
from ctypes import wintypes
from pathlib import Path

import torch
from torch.nn import functional as F

from .config import ModelConfig
from .recording import ControlProfile
from .telemetry import CapturedFrame

if os.name != "nt":
    raise ImportError("the Windows capture backend is only available on Windows")


user32 = ctypes.WinDLL("user32", use_last_error=True)
kernel32 = ctypes.WinDLL("kernel32", use_last_error=True)

WM_INPUT = 0x00FF
WM_DESTROY = 0x0002
WM_CLOSE = 0x0010
WM_QUIT = 0x0012
RID_INPUT = 0x10000003
RIM_TYPEMOUSE = 0
RIM_TYPEKEYBOARD = 1
RIDEV_INPUTSINK = 0x00000100
RI_KEY_BREAK = 0x0001
HWND_MESSAGE = -3
PROCESS_QUERY_LIMITED_INFORMATION = 0x1000
INPUT_MOUSE = 0
INPUT_KEYBOARD = 1
MOUSEEVENTF_MOVE = 0x0001
KEYEVENTF_KEYUP = 0x0002


class RAWINPUTDEVICE(ctypes.Structure):
    _fields_ = (
        ("usUsagePage", wintypes.USHORT),
        ("usUsage", wintypes.USHORT),
        ("dwFlags", wintypes.DWORD),
        ("hwndTarget", wintypes.HWND),
    )


class RAWINPUTHEADER(ctypes.Structure):
    _fields_ = (
        ("dwType", wintypes.DWORD),
        ("dwSize", wintypes.DWORD),
        ("hDevice", wintypes.HANDLE),
        ("wParam", wintypes.WPARAM),
    )


class MOUSEINPUT(ctypes.Structure):
    _fields_ = (
        ("dx", wintypes.LONG),
        ("dy", wintypes.LONG),
        ("mouseData", wintypes.DWORD),
        ("dwFlags", wintypes.DWORD),
        ("time", wintypes.DWORD),
        ("dwExtraInfo", wintypes.WPARAM),
    )


class KEYBDINPUT(ctypes.Structure):
    _fields_ = (
        ("wVk", wintypes.WORD),
        ("wScan", wintypes.WORD),
        ("dwFlags", wintypes.DWORD),
        ("time", wintypes.DWORD),
        ("dwExtraInfo", wintypes.WPARAM),
    )


class INPUTUNION(ctypes.Union):
    _fields_ = (("mi", MOUSEINPUT), ("ki", KEYBDINPUT))


class INPUT(ctypes.Structure):
    _anonymous_ = ("union",)
    _fields_ = (("type", wintypes.DWORD), ("union", INPUTUNION))


WNDPROC = ctypes.WINFUNCTYPE(
    ctypes.c_ssize_t,
    wintypes.HWND,
    wintypes.UINT,
    wintypes.WPARAM,
    wintypes.LPARAM,
)


class WNDCLASSW(ctypes.Structure):
    _fields_ = (
        ("style", wintypes.UINT),
        ("lpfnWndProc", WNDPROC),
        ("cbClsExtra", ctypes.c_int),
        ("cbWndExtra", ctypes.c_int),
        ("hInstance", wintypes.HINSTANCE),
        ("hIcon", wintypes.HICON),
        ("hCursor", wintypes.HANDLE),
        ("hbrBackground", wintypes.HBRUSH),
        ("lpszMenuName", wintypes.LPCWSTR),
        ("lpszClassName", wintypes.LPCWSTR),
    )


def _raise_last_error(message: str) -> None:
    raise OSError(ctypes.get_last_error(), message)


def _window_process_name(hwnd: int) -> str:
    pid = wintypes.DWORD()
    user32.GetWindowThreadProcessId(hwnd, ctypes.byref(pid))
    process = kernel32.OpenProcess(PROCESS_QUERY_LIMITED_INFORMATION, False, pid.value)
    if not process:
        return ""
    try:
        length = wintypes.DWORD(32768)
        buffer = ctypes.create_unicode_buffer(length.value)
        if not kernel32.QueryFullProcessImageNameW(
            process, 0, buffer, ctypes.byref(length)
        ):
            return ""
        return Path(buffer.value).name
    finally:
        kernel32.CloseHandle(process)


def _find_window(profile: ControlProfile) -> int:
    matches: list[int] = []
    title_pattern = re.compile(profile.window.title_regex)

    @ctypes.WINFUNCTYPE(wintypes.BOOL, wintypes.HWND, wintypes.LPARAM)
    def callback(hwnd: int, _: int) -> bool:
        if not user32.IsWindowVisible(hwnd):
            return True
        length = user32.GetWindowTextLengthW(hwnd)
        if length <= 0:
            return True
        title = ctypes.create_unicode_buffer(length + 1)
        user32.GetWindowTextW(hwnd, title, length + 1)
        if title_pattern.search(title.value) and _window_process_name(
            hwnd
        ).casefold() == (profile.window.process_name.casefold()):
            matches.append(hwnd)
        return True

    user32.EnumWindows(callback, 0)
    if not matches:
        raise RuntimeError(
            f"no visible {profile.window.process_name} window matched "
            f"{profile.window.title_regex!r}"
        )
    if len(matches) > 1:
        raise RuntimeError("control profile matched more than one window")
    return matches[0]


def _vk_code(name: str) -> int | None:
    normalized = name.upper()
    if len(normalized) == 1 and normalized.isalnum():
        return ord(normalized)
    fixed = {
        "SPACE": 0x20,
        "SHIFT": 0x10,
        "CTRL": 0x11,
        "ALT": 0x12,
        "TAB": 0x09,
        "ESC": 0x1B,
        "UP": 0x26,
        "DOWN": 0x28,
        "LEFT": 0x25,
        "RIGHT": 0x27,
    }
    if normalized in fixed:
        return fixed[normalized]
    if normalized.startswith("F") and normalized[1:].isdigit():
        number = int(normalized[1:])
        if 1 <= number <= 24:
            return 0x6F + number
    return None


class WindowsCaptureBackend:
    """WGC recorder backend; mouse motion is sourced only from WM_INPUT."""

    def __init__(self, profile_path: Path, cfg: ModelConfig) -> None:
        self.profile = ControlProfile.from_json(profile_path)
        self.cfg = cfg
        if len(self.profile.buttons) != cfg.num_buttons:
            raise ValueError(
                "control profile button count does not match model config: "
                f"{len(self.profile.buttons)} != {cfg.num_buttons}"
            )
        self.target_hwnd = _find_window(self.profile)
        self._frame_lock = threading.Lock()
        self._latest_frame: CapturedFrame | None = None
        self._mouse_lock = threading.Lock()
        self._mouse_events: deque[tuple[float, int, int]] = deque()
        self._held_vks: set[int] = set()
        self._held_mouse: set[str] = set()
        self._emergency = False
        self._input_thread: threading.Thread | None = None
        self._input_hwnd: int | None = None
        self._input_thread_id: int | None = None
        self._capture_control = None
        self._wndproc: object | None = None
        self._injected_held: set[str] = set()
        self._source_size: tuple[int, int] | None = None
        self._last_preprocess_ms = 0.0

    def _binding_held(self, binding: str) -> bool:
        normalized = binding.upper()
        if normalized.startswith("MOUSE"):
            return normalized in self._held_mouse
        code = _vk_code(normalized)
        if code is None:
            raise ValueError(f"unsupported input binding: {binding}")
        return code in self._held_vks

    def held_inputs(self) -> set[str]:
        bindings = {
            *self.profile.movement.values(),
            *self.profile.buttons,
            self.profile.pause_key,
            self.profile.emergency_stop_key,
        }
        return {binding for binding in bindings if self._binding_held(binding)}

    def _handle_mouse_buttons(self, flags: int) -> None:
        pairs = (
            (0x0001, 0x0002, "MOUSE1"),
            (0x0004, 0x0008, "MOUSE2"),
            (0x0010, 0x0020, "MOUSE3"),
            (0x0040, 0x0080, "MOUSE4"),
            (0x0100, 0x0200, "MOUSE5"),
        )
        for down, up, name in pairs:
            if flags & down:
                self._held_mouse.add(name)
            if flags & up:
                self._held_mouse.discard(name)

    def _handle_raw_input(self, raw_handle: int) -> None:
        size = wintypes.UINT()
        header_size = ctypes.sizeof(RAWINPUTHEADER)
        if (
            user32.GetRawInputData(
                raw_handle, RID_INPUT, None, ctypes.byref(size), header_size
            )
            == 0xFFFFFFFF
        ):
            return
        buffer = ctypes.create_string_buffer(size.value)
        if (
            user32.GetRawInputData(
                raw_handle, RID_INPUT, buffer, ctypes.byref(size), header_size
            )
            == 0xFFFFFFFF
        ):
            return
        header = RAWINPUTHEADER.from_buffer_copy(buffer.raw[:header_size])
        payload = buffer.raw[header_size:]
        if header.dwType == RIM_TYPEMOUSE and len(payload) >= 20:
            button_flags = int.from_bytes(payload[4:6], "little")
            delta_x = int.from_bytes(payload[12:16], "little", signed=True)
            delta_y = int.from_bytes(payload[16:20], "little", signed=True)
            self._handle_mouse_buttons(button_flags)
            if delta_x or delta_y:
                with self._mouse_lock:
                    self._mouse_events.append((time.perf_counter(), delta_x, delta_y))
        elif header.dwType == RIM_TYPEKEYBOARD and len(payload) >= 8:
            flags = int.from_bytes(payload[2:4], "little")
            virtual_key = int.from_bytes(payload[6:8], "little")
            if flags & RI_KEY_BREAK:
                self._held_vks.discard(virtual_key)
            else:
                self._held_vks.add(virtual_key)
                if virtual_key == _vk_code(self.profile.emergency_stop_key):
                    self._emergency = True

    def _input_loop(self) -> None:
        instance = kernel32.GetModuleHandleW(None)
        class_name = f"OverAIRawInput_{os.getpid()}_{id(self)}"

        @WNDPROC
        def wndproc(hwnd: int, message: int, wparam: int, lparam: int) -> int:
            if message == WM_INPUT:
                self._handle_raw_input(lparam)
                return 0
            if message in (WM_CLOSE, WM_DESTROY):
                user32.PostQuitMessage(0)
                return 0
            return user32.DefWindowProcW(hwnd, message, wparam, lparam)

        self._wndproc = wndproc
        window_class = WNDCLASSW(
            0,
            wndproc,
            0,
            0,
            instance,
            None,
            None,
            None,
            None,
            class_name,
        )
        if not user32.RegisterClassW(ctypes.byref(window_class)):
            _raise_last_error("RegisterClassW failed")
        hwnd = user32.CreateWindowExW(
            0,
            class_name,
            class_name,
            0,
            0,
            0,
            0,
            0,
            HWND_MESSAGE,
            None,
            instance,
            None,
        )
        if not hwnd:
            _raise_last_error("CreateWindowExW failed")
        self._input_hwnd = hwnd
        self._input_thread_id = kernel32.GetCurrentThreadId()
        devices = (RAWINPUTDEVICE * 2)(
            RAWINPUTDEVICE(0x01, 0x02, RIDEV_INPUTSINK, hwnd),
            RAWINPUTDEVICE(0x01, 0x06, RIDEV_INPUTSINK, hwnd),
        )
        if not user32.RegisterRawInputDevices(
            devices, len(devices), ctypes.sizeof(RAWINPUTDEVICE)
        ):
            _raise_last_error("RegisterRawInputDevices failed")
        message = wintypes.MSG()
        while user32.GetMessageW(ctypes.byref(message), None, 0, 0) > 0:
            user32.TranslateMessage(ctypes.byref(message))
            user32.DispatchMessageW(ctypes.byref(message))
        user32.DestroyWindow(hwnd)
        user32.UnregisterClassW(class_name, instance)

    def start(self) -> None:
        try:
            import windows_capture  # pyright: ignore[reportMissingImports]
        except ImportError as error:
            raise RuntimeError("install the recording dependency group") from error
        self._input_thread = threading.Thread(
            target=self._input_loop, name="overai-raw-input", daemon=True
        )
        self._input_thread.start()
        capture = windows_capture.WindowsCapture(
            cursor_capture=False,
            draw_border=False,
            minimum_update_interval=1,
            dirty_region=False,
            window_hwnd=self.target_hwnd,
        )

        @capture.event
        def on_frame_arrived(frame, _capture_control) -> None:
            captured_at = time.perf_counter()
            preprocess_start = time.perf_counter()
            buffer = torch.from_numpy(frame.frame_buffer)
            height, width = buffer.shape[:2]
            if abs(width / height - 16 / 9) > 1e-3:
                self._emergency = True
                return
            source_size = (height, width)
            if self._source_size is None:
                self._source_size = source_size
            elif source_size != self._source_size:
                self._emergency = True
                return
            bgra_surface: torch.Tensor | None = None
            if self.profile.hud_telemetry is not None:
                reference_width, reference_height = (
                    self.profile.hud_telemetry.reference_resolution
                )
                color_surface = buffer.permute(2, 0, 1).contiguous()
                if (height, width) != (reference_height, reference_width):
                    color_surface = (
                        F.interpolate(
                            color_surface.unsqueeze(0).float(),
                            size=(reference_height, reference_width),
                            mode="bilinear",
                            align_corners=False,
                        )
                        .round()
                        .clamp(0, 255)
                        .to(torch.uint8)[0]
                    )
                bgra_surface = color_surface
                channels = color_surface[[2, 0]]
            else:
                channels = buffer[..., (2, 0)].permute(2, 0, 1).contiguous()
            if tuple(channels.shape[1:]) != (
                self.cfg.image_height,
                self.cfg.image_width,
            ):
                channels = (
                    F.interpolate(
                        channels.unsqueeze(0).float(),
                        size=(self.cfg.image_height, self.cfg.image_width),
                        mode="bilinear",
                        align_corners=False,
                    )
                    .round()
                    .clamp(0, 255)
                    .to(torch.uint8)[0]
                )
            else:
                channels = channels.clone()
            self._last_preprocess_ms = (
                time.perf_counter() - preprocess_start
            ) * 1000.0
            with self._frame_lock:
                self._latest_frame = CapturedFrame(
                    captured_at, channels, bgra_surface
                )

        @capture.event
        def on_closed() -> None:
            self._emergency = True

        self._capture_control = capture.start_free_threaded()

    def stop(self) -> None:
        if self._capture_control is not None:
            self._capture_control.stop()
            self._capture_control = None
        if self._input_thread_id is not None:
            user32.PostThreadMessageW(self._input_thread_id, WM_QUIT, 0, 0)
        if self._input_thread is not None:
            self._input_thread.join(timeout=2)
            self._input_thread = None

    def latest_frame(self, timeout_ms: int) -> CapturedFrame | None:
        deadline = time.perf_counter() + timeout_ms / 1000
        while True:
            with self._frame_lock:
                frame = self._latest_frame
                self._latest_frame = None
            if frame is not None:
                return frame
            if time.perf_counter() >= deadline:
                return None
            time.sleep(0.0005)

    def drain_mouse_deltas(self, start: float, end: float) -> tuple[int, int]:
        delta_x = delta_y = 0
        with self._mouse_lock:
            while self._mouse_events and self._mouse_events[0][0] <= end:
                timestamp, x, y = self._mouse_events.popleft()
                if timestamp >= start:
                    delta_x += x
                    delta_y += y
        return delta_x, delta_y

    def target_active(self) -> bool:
        return bool(user32.IsWindow(self.target_hwnd)) and (
            user32.GetForegroundWindow() == self.target_hwnd
        )

    def emergency_stop_requested(self) -> bool:
        return self._emergency

    def capture_diagnostics(self) -> dict[str, float]:
        return {"preprocessing_ms": self._last_preprocess_ms}

    @staticmethod
    def _send(inputs: list[INPUT]) -> None:
        if not inputs:
            return
        values = (INPUT * len(inputs))(*inputs)
        sent = user32.SendInput(len(values), values, ctypes.sizeof(INPUT))
        if sent != len(values):
            _raise_last_error("SendInput failed")

    def apply_relative_mouse(self, delta_x: int, delta_y: int) -> None:
        if delta_x == 0 and delta_y == 0:
            return
        self._send(
            [
                INPUT(
                    INPUT_MOUSE,
                    INPUTUNION(
                        mi=MOUSEINPUT(delta_x, delta_y, 0, MOUSEEVENTF_MOVE, 0, 0)
                    ),
                )
            ]
        )

    @staticmethod
    def _mouse_transition(name: str, pressed: bool) -> INPUT:
        flags = {
            "MOUSE1": (0x0002, 0x0004),
            "MOUSE2": (0x0008, 0x0010),
            "MOUSE3": (0x0020, 0x0040),
            "MOUSE4": (0x0080, 0x0100),
            "MOUSE5": (0x0080, 0x0100),
        }
        down, up = flags[name]
        data = 1 if name == "MOUSE4" else 2 if name == "MOUSE5" else 0
        return INPUT(
            INPUT_MOUSE,
            INPUTUNION(mi=MOUSEINPUT(0, 0, data, down if pressed else up, 0, 0)),
        )

    @staticmethod
    def _key_transition(name: str, pressed: bool) -> INPUT:
        virtual_key = _vk_code(name)
        if virtual_key is None:
            raise ValueError(f"unsupported output binding: {name}")
        return INPUT(
            INPUT_KEYBOARD,
            INPUTUNION(
                ki=KEYBDINPUT(virtual_key, 0, 0 if pressed else KEYEVENTF_KEYUP, 0, 0)
            ),
        )

    def apply_discrete(
        self, movement: tuple[int, int], buttons: tuple[bool, ...]
    ) -> None:
        if len(buttons) != len(self.profile.buttons):
            raise ValueError(
                "inference button count does not match control profile: "
                f"{len(buttons)} != {len(self.profile.buttons)}"
            )
        desired: set[str] = set()
        if movement[0] == 0:
            desired.add(self.profile.movement["left"])
        elif movement[0] == 2:
            desired.add(self.profile.movement["right"])
        if movement[1] == 0:
            desired.add(self.profile.movement["reverse"])
        elif movement[1] == 2:
            desired.add(self.profile.movement["forward"])
        desired.update(
            binding
            for binding, pressed in zip(self.profile.buttons, buttons, strict=True)
            if pressed
        )
        transitions: list[INPUT] = []
        for binding in sorted(self._injected_held - desired):
            transitions.append(
                self._mouse_transition(binding.upper(), False)
                if binding.upper().startswith("MOUSE")
                else self._key_transition(binding, False)
            )
        for binding in sorted(desired - self._injected_held):
            transitions.append(
                self._mouse_transition(binding.upper(), True)
                if binding.upper().startswith("MOUSE")
                else self._key_transition(binding, True)
            )
        self._send(transitions)
        self._injected_held = desired

    def release_all(self) -> None:
        transitions = [
            (
                self._mouse_transition(binding.upper(), False)
                if binding.upper().startswith("MOUSE")
                else self._key_transition(binding, False)
            )
            for binding in sorted(self._injected_held)
        ]
        self._send(transitions)
        self._injected_held.clear()
