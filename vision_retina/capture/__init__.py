"""采集层工厂：按平台/来源选择后端，并统一暴露给总线。

用法：
    cap = create_capture("desktop")          # 自动：Windows DXGI → GDI 兜底
    cap = create_capture("desktop", backend="gdi")
    cam = create_capture("camera", camera_index=0)
    with cap:
        frame = cap.get_frame()
"""
from __future__ import annotations

import sys
from typing import Any

from .base import AccessLost, CaptureError, CaptureUnavailable, RetinaCapture, enable_dpi_awareness
from .camera import CameraCapture
from .dxgi import DxgiCapture
from .gdi import GdiCapture
from .linux import LinuxDisplayCapture
from .macos import MacDisplayCapture

WINDOWS_BACKENDS = {"dxgi": DxgiCapture, "gdi": GdiCapture}
PLATFORM_BACKENDS = {
    "darwin": {"macos": MacDisplayCapture},
    "linux": {"linux": LinuxDisplayCapture},
}


def available_backends(source: str = "desktop") -> list[str]:
    source = (source or "desktop").lower()
    if source == "camera":
        return ["opencv"]
    if sys.platform == "win32":
        return ["dxgi", "gdi"]
    return list(PLATFORM_BACKENDS.get(sys.platform, {}).keys())


def create_capture(
    source: str = "desktop",
    backend: str = "auto",
    config: Any = None,
    *,
    open_capture: bool = True,
    **kwargs: Any,
) -> RetinaCapture:
    """创建（并默认打开）采集器。

    config 为 `vision_retina.config.Config` 时，从它的 capture/camera/bus 段取默认值；
    显式 kwargs 优先级最高。
    """
    source = (source or "desktop").lower()
    cfg_capture = config.section("capture") if config is not None else {}
    cfg_camera = config.section("camera") if config is not None else {}
    cfg_bus = config.section("bus") if config is not None else {}

    fps = kwargs.pop("fps", None) or cfg_capture.get("fps") or 2.0
    ring_size = kwargs.pop("ring_size", None) or cfg_capture.get("ring_size") or 8
    region = kwargs.pop("region", None) or cfg_capture.get("region") or None

    if source == "camera":
        camera_ref = kwargs.pop("camera_path", None)
        if camera_ref is None:
            camera_ref = kwargs.pop("camera_index", None)
        if camera_ref is None:
            camera_ref = cfg_camera.get("path") or cfg_camera.get("index", 0)
        if isinstance(camera_ref, str) and camera_ref.strip().isdigit():
            camera_ref = int(camera_ref.strip())
        return _open_instance(
            CameraCapture(
                index=camera_ref,
                width=int(kwargs.pop("width", None) or cfg_camera.get("width", 1280)),
                height=int(kwargs.pop("height", None) or cfg_camera.get("height", 720)),
                fps=float(kwargs.pop("fps_camera", None) or cfg_camera.get("fps", fps)),
                ring_size=int(ring_size),
                region=region,
            ),
            open_capture,
        )

    if source != "desktop":
        raise CaptureUnavailable(f"未知采集源：{source}（可选 desktop / camera）")

    if backend and backend != "auto":
        table = {**WINDOWS_BACKENDS, **PLATFORM_BACKENDS.get(sys.platform, {})}
        cls = table.get(backend)
        if cls is None:
            raise CaptureUnavailable(
                f"当前平台（{sys.platform}）不支持后端 {backend}，可用：{available_backends('desktop')}"
            )
        order = [cls]
    else:
        if sys.platform == "win32":
            order = [DxgiCapture, GdiCapture]
        elif sys.platform == "darwin":
            order = [MacDisplayCapture]
        else:
            order = [LinuxDisplayCapture]

    errors: list[str] = []
    for cls in order:
        init_kwargs: dict[str, Any] = {}
        if cls is DxgiCapture:
            init_kwargs["output_index"] = int(
                kwargs.pop("output_index", None) or cfg_capture.get("output_index", 0)
            )
        instance = cls(fps=float(fps), ring_size=int(ring_size), region=region, **init_kwargs)
        try:
            return _open_instance(instance, open_capture)
        except CaptureError as exc:
            errors.append(f"{cls.__name__}: {exc}")
            try:
                instance.close()
            except Exception:
                pass
    raise CaptureUnavailable("所有桌面采集后端都不可用：\n  " + "\n  ".join(errors))


def _open_instance(instance: RetinaCapture, open_capture: bool) -> RetinaCapture:
    if open_capture:
        instance.open()
    return instance


def list_sources() -> list[dict[str, Any]]:
    """列出可用采集源（桌面各后端 + 摄像头）。"""
    items: list[dict[str, Any]] = []
    for name in available_backends("desktop"):
        cls = {"dxgi": DxgiCapture, "gdi": GdiCapture,
               "macos": MacDisplayCapture, "linux": LinuxDisplayCapture}[name]
        for entry in cls.list_sources():
            entry.setdefault("name", "desktop")
            entry["backend"] = name
            items.append(entry)
    items.extend(CameraCapture.list_sources())
    return items


__all__ = [
    "RetinaCapture", "CaptureError", "CaptureUnavailable", "AccessLost",
    "DxgiCapture", "GdiCapture", "CameraCapture", "MacDisplayCapture", "LinuxDisplayCapture",
    "create_capture", "list_sources", "available_backends", "enable_dpi_awareness",
]
