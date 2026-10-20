"""Linux 采集接口占位（Phase 3 的三平台接口之一）。

计划后端：
- Wayland：PipeWire + xdg-desktop-portal（org.freedesktop.portal.ScreenCast），
  由门户弹出授权框，拿到 stream 后经 pipewire 读取 DMA-BUF；
- X11：XShm / XComposite 截图（无 dirty-rects，整帧）；
- 摄像头：V4L2（OpenCV VideoCapture 默认后端即可）。

任何调用在本平台都会明确报 `CaptureUnavailable`，不会静默返回假帧。
"""
from __future__ import annotations

from typing import Any

from .base import CaptureUnavailable, RetinaCapture


class LinuxDisplayCapture(RetinaCapture):
    source = "desktop"
    backend = "linux"

    def _open_backend(self) -> None:
        raise CaptureUnavailable(
            "Linux 桌面采集尚未实现（预留接口）。"
            "计划后端：Wayland=PipeWire + xdg-desktop-portal ScreenCast；X11=XShm。"
        )

    def _close_backend(self) -> None:
        return None

    def _read(self, timeout_ms: int):
        return None

    @classmethod
    def list_sources(cls) -> list[dict[str, Any]]:
        return [{
            "name": "desktop", "kind": "desktop", "backend": "linux",
            "available": False, "reason": "Linux 后端未实现（本机为 Windows）",
        }]


__all__ = ["LinuxDisplayCapture"]
