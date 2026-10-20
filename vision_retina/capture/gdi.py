"""Windows GDI 桌面采集（DXGI 不可用时的兜底实现）。

特点与限制：
- 兼容性最好（RDP、部分虚拟机、老显卡都能用），但每次都是整帧 BitBlt，没有 dirty-rects；
- 因此本实现每帧都标记 `is_key_frame=True`，由上层总线的帧间差分补齐增量信息；
- 额外把鼠标指针画进画面（DXGI 路径默认不含指针），方便视觉模型理解"手"的位置。
"""
from __future__ import annotations

import ctypes
from ctypes import wintypes as w
from typing import Any

import numpy as np

from ..frames import Rect, RetinaFrame, now_ms
from .base import CaptureUnavailable, RetinaCapture, enable_dpi_awareness

SRCCOPY = 0x00CC0020
DIB_RGB_COLORS = 0
BI_RGB = 0
SM_XVIRTUALSCREEN = 76
SM_YVIRTUALSCREEN = 77
SM_CXVIRTUALSCREEN = 78
SM_CYVIRTUALSCREEN = 79
CURSOR_SHOWING = 0x00000001
DI_NORMAL = 0x0003


class BITMAPINFOHEADER(ctypes.Structure):
    _fields_ = [
        ("biSize", w.DWORD),
        ("biWidth", w.LONG),
        ("biHeight", w.LONG),
        ("biPlanes", w.WORD),
        ("biBitCount", w.WORD),
        ("biCompression", w.DWORD),
        ("biSizeImage", w.DWORD),
        ("biXPelsPerMeter", w.LONG),
        ("biYPelsPerMeter", w.LONG),
        ("biClrUsed", w.DWORD),
        ("biClrImportant", w.DWORD),
    ]


class BITMAPINFO(ctypes.Structure):
    _fields_ = [("bmiHeader", BITMAPINFOHEADER), ("bmiColors", w.DWORD * 3)]


class CURSORINFO(ctypes.Structure):
    _fields_ = [
        ("cbSize", w.DWORD),
        ("flags", w.DWORD),
        ("hCursor", w.HANDLE),
        ("ptScreenPos", w.POINT),
    ]


class GdiCapture(RetinaCapture):
    """整帧 GDI 抓屏。"""

    source = "desktop"

    def __init__(self, fps: float = 2.0, ring_size: int = 8,
                 region: tuple[int, int, int, int] | None = None, draw_cursor: bool = True):
        super().__init__(fps=fps, ring_size=ring_size, region=region)
        self.draw_cursor = bool(draw_cursor)
        self._user32 = None
        self._gdi32 = None
        self._hdc_screen = None
        self._hdc_mem = None
        self._hbmp = None
        self._old_obj = None
        self._width = 0
        self._height = 0
        self._origin = (0, 0)
        self._buffer = None
        self._bmi = None
        self._dpi_mode = "unaware"

    # ---- 生命周期 ----

    def _open_backend(self) -> None:
        self._dpi_mode = enable_dpi_awareness()
        try:
            self._user32 = ctypes.WinDLL("user32", use_last_error=True)
            self._gdi32 = ctypes.WinDLL("gdi32", use_last_error=True)
        except OSError as exc:  # pragma: no cover - 仅非 Windows 会走到
            raise CaptureUnavailable(f"无法加载 user32/gdi32：{exc}") from exc

        u, g = self._user32, self._gdi32
        u.GetDC.restype = w.HDC
        u.GetDC.argtypes = [w.HWND]
        u.ReleaseDC.argtypes = [w.HWND, w.HDC]
        u.GetSystemMetrics.restype = ctypes.c_int
        u.GetSystemMetrics.argtypes = [ctypes.c_int]
        u.GetCursorInfo.argtypes = [ctypes.POINTER(CURSORINFO)]
        u.GetCursorInfo.restype = w.BOOL
        u.DrawIconEx.restype = w.BOOL
        u.DrawIconEx.argtypes = [w.HDC, ctypes.c_int, ctypes.c_int, w.HANDLE,
                                 ctypes.c_int, ctypes.c_int, w.UINT, w.HANDLE, w.UINT]
        g.CreateCompatibleDC.restype = w.HDC
        g.CreateCompatibleDC.argtypes = [w.HDC]
        g.CreateCompatibleBitmap.restype = w.HBITMAP
        g.CreateCompatibleBitmap.argtypes = [w.HDC, ctypes.c_int, ctypes.c_int]
        g.SelectObject.restype = w.HGDIOBJ
        g.SelectObject.argtypes = [w.HDC, w.HGDIOBJ]
        g.BitBlt.restype = w.BOOL
        g.BitBlt.argtypes = [w.HDC, ctypes.c_int, ctypes.c_int, ctypes.c_int, ctypes.c_int,
                             w.HDC, ctypes.c_int, ctypes.c_int, w.DWORD]
        g.GetDIBits.restype = ctypes.c_int
        g.GetDIBits.argtypes = [w.HDC, w.HBITMAP, w.UINT, w.UINT, ctypes.c_void_p,
                                ctypes.POINTER(BITMAPINFO), w.UINT]
        g.DeleteObject.argtypes = [w.HGDIOBJ]
        g.DeleteDC.argtypes = [w.HDC]

        self._width = int(u.GetSystemMetrics(SM_CXVIRTUALSCREEN))
        self._height = int(u.GetSystemMetrics(SM_CYVIRTUALSCREEN))
        self._origin = (int(u.GetSystemMetrics(SM_XVIRTUALSCREEN)),
                        int(u.GetSystemMetrics(SM_YVIRTUALSCREEN)))
        if self._width <= 0 or self._height <= 0:
            raise CaptureUnavailable("无法获取桌面尺寸（当前可能没有交互式桌面会话）")

        self._hdc_screen = u.GetDC(None)
        if not self._hdc_screen:
            raise CaptureUnavailable("GetDC 失败：没有可访问的桌面")
        self._hdc_mem = g.CreateCompatibleDC(self._hdc_screen)
        self._hbmp = g.CreateCompatibleBitmap(self._hdc_screen, self._width, self._height)
        if not self._hdc_mem or not self._hbmp:
            self._close_backend()
            raise CaptureUnavailable("创建兼容位图失败")
        self._old_obj = g.SelectObject(self._hdc_mem, self._hbmp)

        self._bmi = BITMAPINFO()
        self._bmi.bmiHeader.biSize = ctypes.sizeof(BITMAPINFOHEADER)
        self._bmi.bmiHeader.biWidth = self._width
        self._bmi.bmiHeader.biHeight = -self._height   # 负值 = 自上而下
        self._bmi.bmiHeader.biPlanes = 1
        self._bmi.bmiHeader.biBitCount = 32
        self._bmi.bmiHeader.biCompression = BI_RGB
        self._buffer = (ctypes.c_ubyte * (self._width * self._height * 4))()

    def _close_backend(self) -> None:
        g, u = self._gdi32, self._user32
        try:
            if g and self._hdc_mem and self._old_obj:
                g.SelectObject(self._hdc_mem, self._old_obj)
            if g and self._hbmp:
                g.DeleteObject(self._hbmp)
            if g and self._hdc_mem:
                g.DeleteDC(self._hdc_mem)
            if u and self._hdc_screen:
                u.ReleaseDC(None, self._hdc_screen)
        finally:
            self._hdc_mem = self._hdc_screen = self._hbmp = self._old_obj = None
            self._buffer = None
            self._bmi = None

    # ---- 读帧 ----

    def _read(self, timeout_ms: int) -> RetinaFrame | None:
        u, g = self._user32, self._gdi32
        if not g.BitBlt(self._hdc_mem, 0, 0, self._width, self._height,
                        self._hdc_screen, self._origin[0], self._origin[1], SRCCOPY):
            raise CaptureUnavailable(f"BitBlt 失败（Win32 错误 {ctypes.get_last_error()}）")
        if self.draw_cursor:
            self._paint_cursor()
        if not g.GetDIBits(self._hdc_mem, self._hbmp, 0, self._height, self._buffer,
                           ctypes.byref(self._bmi), DIB_RGB_COLORS):
            raise CaptureUnavailable(f"GetDIBits 失败（Win32 错误 {ctypes.get_last_error()}）")

        bgra = np.frombuffer(self._buffer, dtype=np.uint8).reshape(self._height, self._width, 4)
        rgb = bgra[:, :, :3][:, :, ::-1].copy()
        rgb = self._apply_region(rgb)
        return RetinaFrame(
            frame_id=0,
            timestamp_ms=now_ms(),
            source=self.source,
            pixels=rgb,
            dirty_rects=[],          # GDI 不提供脏区，空表示整帧
            is_key_frame=True,
            meta={"backend": "gdi", "virtual_origin": list(self._origin)},
        )

    def _paint_cursor(self) -> None:
        info = CURSORINFO()
        info.cbSize = ctypes.sizeof(CURSORINFO)
        if not self._user32.GetCursorInfo(ctypes.byref(info)):
            return
        if not (info.flags & CURSOR_SHOWING) or not info.hCursor:
            return
        x = info.ptScreenPos.x - self._origin[0]
        y = info.ptScreenPos.y - self._origin[1]
        # DrawIconEx(hdc, x, y, hIcon, w, h, 0, NULL, DI_NORMAL)
        self._user32.DrawIconEx(
            ctypes.c_void_p(self._hdc_mem), int(x), int(y), ctypes.c_void_p(info.hCursor),
            0, 0, 0, None, DI_NORMAL,
        )

    # ---- 描述 ----

    def describe(self) -> dict[str, Any]:
        data = super().describe()
        data.update({
            "backend": "gdi",
            "width": self._width,
            "height": self._height,
            "dpi_mode": self._dpi_mode,
            "draw_cursor": self.draw_cursor,
            "dirty_rects": "unsupported (full frame every time)",
        })
        return data

    @classmethod
    def list_sources(cls) -> list[dict[str, Any]]:
        return [{"name": "desktop", "kind": "desktop", "backend": "gdi", "available": True}]


__all__ = ["GdiCapture", "Rect"]
