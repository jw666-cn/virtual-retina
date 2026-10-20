"""macOS 采集接口占位（Phase 3 的三平台接口之一）。

本机为 Windows，无法实测；接口先固定下来，实现留待 macOS 机器接入时补齐：

- 桌面：CGDisplayStream（macOS 10.13–12）或 ScreenCaptureKit（macOS 12.3+，
  需要 Screen Recording 权限）；dirty-rects 用 CGDisplayStreamUpdateGetRects 获取。
- 摄像头：AVFoundation + OpenCV；或直接复用 CameraCapture（OpenCV 跨平台）。

任何调用在本平台都会明确报 `CaptureUnavailable`，不会静默返回假帧。
"""
from __future__ import annotations

from typing import Any

from .base import CaptureUnavailable, RetinaCapture


class MacDisplayCapture(RetinaCapture):
    source = "desktop"
    backend = "macos"

    def _open_backend(self) -> None:
        raise CaptureUnavailable(
            "macOS 桌面采集尚未实现（预留接口）。"
            "计划后端：ScreenCaptureKit（macOS 12.3+）/ CGDisplayStream；需要屏幕录制权限。"
        )

    def _close_backend(self) -> None:
        return None

    def _read(self, timeout_ms: int):
        return None

    @classmethod
    def list_sources(cls) -> list[dict[str, Any]]:
        return [{
            "name": "desktop", "kind": "desktop", "backend": "macos",
            "available": False, "reason": "macOS 后端未实现（本机为 Windows）",
        }]


__all__ = ["MacDisplayCapture"]
