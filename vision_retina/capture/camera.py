"""摄像头采集（OpenCV）。

与桌面采集共用 `RetinaCapture` 接口：上层总线只看到 `source="camera"` 的帧，
同样支持环形缓冲、区域白名单、静默与增量编码。
"""
from __future__ import annotations

import sys
from typing import Any

import numpy as np

from ..frames import RetinaFrame, now_ms
from .base import CaptureUnavailable, RetinaCapture


class CameraCapture(RetinaCapture):
    source = "camera"

    def __init__(self, index: int | str = 0, width: int = 1280, height: int = 720,
                 fps: float = 2.0, ring_size: int = 8,
                 region: tuple[int, int, int, int] | None = None):
        super().__init__(fps=fps, ring_size=ring_size, region=region)
        # index 可以是摄像头序号（UVC）或视频文件路径（无硬件时的联调/回归源）
        self._is_file = isinstance(index, str)
        self.index = index if self._is_file else int(index)
        self.requested_width = int(width)
        self.requested_height = int(height)
        self._cap = None
        self._backend = "opencv"
        self._actual = (0, 0)
        self._loops = 0

    def _open_backend(self) -> None:
        try:
            import cv2
        except ImportError as exc:
            raise CaptureUnavailable("缺少 opencv-python，无法采集摄像头") from exc

        api = cv2.CAP_DSHOW if sys.platform == "win32" else cv2.CAP_ANY
        if self._is_file:
            cap = cv2.VideoCapture(str(self.index))
        else:
            cap = cv2.VideoCapture(self.index, api)
        if not cap.isOpened():
            cap.release()
            if self._is_file:
                raise CaptureUnavailable(f"无法打开视频文件：{self.index}")
            raise CaptureUnavailable(
                f"无法打开摄像头 index={self.index}。"
                "请检查设备是否被占用、Windows 相机隐私设置是否允许桌面应用访问。"
            )
        if not self._is_file:
            if self.requested_width > 0:
                cap.set(cv2.CAP_PROP_FRAME_WIDTH, self.requested_width)
            if self.requested_height > 0:
                cap.set(cv2.CAP_PROP_FRAME_HEIGHT, self.requested_height)
            cap.set(cv2.CAP_PROP_BUFFERSIZE, 1)  # 尽量取最新帧，避免延迟
        self._cap = cap
        self._cv2 = cv2
        self._actual = (int(cap.get(cv2.CAP_PROP_FRAME_WIDTH)),
                        int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT)))

    def _close_backend(self) -> None:
        if self._cap is not None:
            try:
                self._cap.release()
            finally:
                self._cap = None

    def _read(self, timeout_ms: int) -> RetinaFrame | None:
        if self._cap is None:
            raise CaptureUnavailable("摄像头尚未打开")
        ok, bgr = self._cap.read()
        if not ok and self._is_file:
            # 视频文件播完从头循环，方便无硬件时做长跑/回归
            self._cap.set(self._cv2.CAP_PROP_POS_FRAMES, 0)
            ok, bgr = self._cap.read()
            self._loops += 1
        if not ok or bgr is None:
            # 偶发丢帧：返回 None，让上层按空帧处理；连续失败由上层重试
            return None
        rgb = self._cv2.cvtColor(bgr, self._cv2.COLOR_BGR2RGB)
        rgb = self._apply_region(rgb)
        meta: dict[str, Any] = {"backend": "opencv"}
        if self._is_file:
            meta.update({"video_path": str(self.index), "loop": self._loops})
        else:
            meta["camera_index"] = self.index
        return RetinaFrame(
            frame_id=0,
            timestamp_ms=now_ms(),
            source=self.source,
            pixels=rgb,
            dirty_rects=[],
            is_key_frame=True,
            meta=meta,
        )

    def describe(self) -> dict[str, Any]:
        data = super().describe()
        data.update({
            "backend": "opencv",
            "index": self.index,
            "is_file": self._is_file,
            "requested": [self.requested_width, self.requested_height],
            "actual": list(self._actual),
            "loops": self._loops,
        })
        return data

    @classmethod
    def list_sources(cls) -> list[dict[str, Any]]:
        try:
            import cv2
        except ImportError:
            return [{"name": "camera", "kind": "camera", "available": False, "reason": "opencv-python 未安装"}]
        try:
            try:
                # OpenCV 5 把日志控制移到 cv2.utils.logging（cv2.setLogLevel 已不存在）
                cv2.utils.logging.setLogLevel(cv2.utils.logging.LOG_LEVEL_SILENT)
            except Exception:
                pass
        except Exception:
            pass
        found: list[dict[str, Any]] = []
        for index in range(8):
            cap = cv2.VideoCapture(index, cv2.CAP_DSHOW if sys.platform == "win32" else cv2.CAP_ANY)
            try:
                if cap.isOpened():
                    found.append({
                        "name": f"camera:{index}",
                        "kind": "camera",
                        "index": index,
                        "width": int(cap.get(cv2.CAP_PROP_FRAME_WIDTH)),
                        "height": int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT)),
                        "available": True,
                    })
            finally:
                cap.release()
        return found or [{"name": "camera", "kind": "camera", "available": False, "reason": "未发现摄像头"}]


__all__ = ["CameraCapture", "np"]
