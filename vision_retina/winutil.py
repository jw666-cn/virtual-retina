"""Windows 轻量系统工具：光标位置、屏幕尺寸、DPI。

只依赖 ctypes，不引入第三方库；非 Windows 平台返回保守默认值。
"""
from __future__ import annotations

import sys
from typing import Any


def _win():
    import ctypes
    from ctypes import wintypes as w

    user32 = ctypes.WinDLL("user32", use_last_error=True)
    return ctypes, w, user32


def cursor_position() -> tuple[int, int] | None:
    """返回物理像素坐标下的鼠标位置；失败返回 None。"""
    if sys.platform != "win32":
        return None
    try:
        ctypes, w, user32 = _win()
        pt = w.POINT()
        if user32.GetCursorPos(ctypes.byref(pt)):
            return int(pt.x), int(pt.y)
    except Exception:
        pass
    return None


def virtual_screen_rect() -> tuple[int, int, int, int]:
    """虚拟桌面（所有显示器）的 (x, y, width, height)。"""
    if sys.platform != "win32":
        return (0, 0, 0, 0)
    try:
        ctypes, w, user32 = _win()
        user32.GetSystemMetrics.restype = ctypes.c_int
        return (
            int(user32.GetSystemMetrics(76)),  # SM_XVIRTUALSCREEN
            int(user32.GetSystemMetrics(77)),  # SM_YVIRTUALSCREEN
            int(user32.GetSystemMetrics(78)),  # SM_CXVIRTUALSCREEN
            int(user32.GetSystemMetrics(79)),  # SM_CYVIRTUALSCREEN
        )
    except Exception:
        return (0, 0, 0, 0)


def foreground_window() -> dict[str, Any] | None:
    """当前前台窗口的标题与矩形（用于安全判断/审计）。"""
    if sys.platform != "win32":
        return None
    try:
        ctypes, w, user32 = _win()
        hwnd = user32.GetForegroundWindow()
        if not hwnd:
            return None
        length = user32.GetWindowTextLengthW(hwnd)
        buf = ctypes.create_unicode_buffer(length + 2)
        user32.GetWindowTextW(hwnd, buf, length + 2)
        rect = w.RECT()
        user32.GetWindowRect(hwnd, ctypes.byref(rect))
        return {
            "hwnd": int(hwnd),
            "title": buf.value,
            "rect": [rect.left, rect.top, rect.right, rect.bottom],
        }
    except Exception:
        return None


__all__ = ["cursor_position", "virtual_screen_rect", "foreground_window"]
