"""采集层统一抽象（任务书 §3.1 / §8）。

所有平台实现同一个 `RetinaCapture` 接口；上层总线不感知平台差异。

线程模型：
- 每个采集器自带一条采集线程与一个固定长度的环形缓冲；
- 生产者只写环形缓冲，消费者 `get_frame()` 取帧；
- 缓冲满时丢弃最旧帧（保证低延迟，不阻塞采集）。
"""
from __future__ import annotations

import queue
import threading
import time
from abc import ABC, abstractmethod
from typing import Any

from ..frames import Rect, RetinaFrame, now_ms


class CaptureError(RuntimeError):
    """采集层通用错误。"""


class CaptureUnavailable(CaptureError):
    """当前环境不支持该采集方式（如非 Windows、无桌面会话、无摄像头）。"""


class AccessLost(CaptureError):
    """桌面复制被系统收回（锁屏/切换显示器/全屏独显切换），需要重建。"""


def enable_dpi_awareness() -> str:
    """让进程按物理像素工作，避免 DPI 缩放下坐标错位。"""
    import ctypes
    import sys

    if sys.platform != "win32":
        return "not-windows"
    try:
        # PER_MONITOR_AWARE_V2 = -4
        if ctypes.windll.user32.SetProcessDpiAwarenessContext(ctypes.c_void_p(-4)):
            return "per-monitor-v2"
    except Exception:
        pass
    try:
        ctypes.windll.shcore.SetProcessDpiAwareness(2)  # PROCESS_PER_MONITOR_DPI_AWARE
        return "per-monitor"
    except Exception:
        pass
    try:
        ctypes.windll.user32.SetProcessDPIAware()
        return "system-aware"
    except Exception:
        return "unaware"


class RetinaCapture(ABC):
    """采集器接口。"""

    source: str = "desktop"

    def __init__(self, fps: float = 2.0, ring_size: int = 8, region: tuple[int, int, int, int] | None = None):
        self.fps = max(0.1, float(fps))
        self.ring_size = max(2, int(ring_size))
        self._region: tuple[int, int, int, int] | None = tuple(region) if region else None
        self._queue: queue.Queue[RetinaFrame] = queue.Queue(maxsize=self.ring_size)
        self._thread: threading.Thread | None = None
        self._stop = threading.Event()
        self._latest: RetinaFrame | None = None
        self._latest_lock = threading.Lock()
        self._counter = 0
        self._counter_lock = threading.Lock()
        self._error: str | None = None
        self._stats = {"captured": 0, "dropped": 0, "errors": 0, "recoveries": 0}
        self._opened = False

    # ---- 生命周期 ----

    def open(self) -> "RetinaCapture":
        if self._opened:
            return self
        self._open_backend()
        self._opened = True
        return self

    def start(self) -> "RetinaCapture":
        if not self._opened:
            self.open()
        if self._thread and self._thread.is_alive():
            return self
        self._stop.clear()
        self._thread = threading.Thread(target=self._run, name=f"{type(self).__name__}-capture", daemon=True)
        self._thread.start()
        return self

    def stop(self) -> None:
        self._stop.set()
        if self._thread:
            self._thread.join(timeout=3.0)
            self._thread = None

    def close(self) -> None:
        self.stop()
        if self._opened:
            try:
                self._close_backend()
            finally:
                self._opened = False

    def __enter__(self) -> "RetinaCapture":
        return self.start()

    def __exit__(self, *exc: Any) -> None:
        self.close()

    # ---- 取帧 ----

    def get_frame(self, timeout: float = 1.0) -> RetinaFrame | None:
        """从环形缓冲取一帧；超时返回 None。"""
        try:
            return self._queue.get(timeout=timeout)
        except queue.Empty:
            return None

    def drain(self) -> list[RetinaFrame]:
        frames: list[RetinaFrame] = []
        while True:
            try:
                frames.append(self._queue.get_nowait())
            except queue.Empty:
                return frames

    def get_latest(self, timeout: float = 1.0) -> RetinaFrame | None:
        """等第一帧，然后丢弃积压帧，返回最新一帧。

        消费端（总线/大脑调用）偶尔会被网络阻塞；若按 FIFO 逐帧消费，
        队列会长期积压旧画面，既增加内存又让“看到”的画面滞后。
        """
        frame = self.get_frame(timeout=timeout)
        if frame is None:
            return None
        while True:
            try:
                frame = self._queue.get_nowait()
            except queue.Empty:
                return frame

    def latest(self) -> RetinaFrame | None:
        with self._latest_lock:
            return self._latest

    def set_region(self, region: tuple[int, int, int, int] | None) -> None:
        """设置采集区域（隐私白名单）。下一帧生效。"""
        self._region = tuple(region) if region else None

    @property
    def region(self) -> tuple[int, int, int, int] | None:
        return self._region

    # ---- 状态 ----

    def stats(self) -> dict[str, Any]:
        with self._latest_lock:
            latest = self._latest
        return {
            **self._stats,
            "fps": self.fps,
            "ring_size": self.ring_size,
            "queue_depth": self._queue.qsize(),
            "running": bool(self._thread and self._thread.is_alive()),
            "last_frame_id": latest.frame_id if latest else None,
            "last_error": self._error,
        }

    def describe(self) -> dict[str, Any]:
        return {
            "class": type(self).__name__,
            "source": self.source,
            "region": list(self._region) if self._region else None,
            "stats": self.stats(),
        }

    @classmethod
    def list_sources(cls) -> list[dict[str, Any]]:
        return [{"name": "desktop", "available": True}]

    # ---- 采集线程 ----

    def _run(self) -> None:
        interval = 1.0 / self.fps
        next_tick = time.perf_counter()
        while not self._stop.is_set():
            timeout_ms = max(1, int(interval * 1000))
            try:
                frame = self._read(timeout_ms)
            except AccessLost as exc:
                self._error = str(exc)
                self._stats["recoveries"] += 1
                self._recover()
                continue
            except CaptureError as exc:
                self._error = str(exc)
                self._stats["errors"] += 1
                time.sleep(min(1.0, interval))
                continue
            except Exception as exc:  # 防御：任何未预期异常都不应杀死采集线程
                self._error = f"{type(exc).__name__}: {exc}"
                self._stats["errors"] += 1
                time.sleep(min(1.0, interval))
                continue
            if frame is not None:
                with self._counter_lock:
                    self._counter += 1
                    frame.frame_id = self._counter
                if frame.timestamp_ms <= 0:
                    frame.timestamp_ms = now_ms()
                self._publish(frame)
            next_tick += interval
            delay = next_tick - time.perf_counter()
            if delay > 0:
                # 用 stop 事件等待，保证 stop() 能立刻打断
                self._stop.wait(delay)
            else:
                next_tick = time.perf_counter()

    def _publish(self, frame: RetinaFrame) -> None:
        try:
            self._queue.put_nowait(frame)
        except queue.Full:
            try:
                self._queue.get_nowait()
                self._stats["dropped"] += 1
            except queue.Empty:
                pass
            try:
                self._queue.put_nowait(frame)
            except queue.Full:
                self._stats["dropped"] += 1
        with self._latest_lock:
            self._latest = frame
        self._stats["captured"] += 1

    def _recover(self) -> None:
        try:
            self._close_backend()
        except Exception:
            pass
        time.sleep(0.5)
        try:
            self._open_backend()
        except Exception as exc:  # 保持线程存活，持续重试
            self._error = f"重建采集失败：{exc}"

    # ---- 子类实现 ----

    @abstractmethod
    def _open_backend(self) -> None: ...

    @abstractmethod
    def _close_backend(self) -> None: ...

    @abstractmethod
    def _read(self, timeout_ms: int) -> RetinaFrame | None:
        """读取一帧；无新帧返回 None。"""

    # ---- 工具 ----

    def _region_bounds(self, width: int, height: int) -> tuple[int, int, int, int]:
        """区域白名单裁剪后的 (x0, y0, x1, y1)，无区域时返回整幅。"""
        if not self._region:
            return 0, 0, width, height
        x, y, w, h = self._region
        x0, y0 = max(0, x), max(0, y)
        x1, y1 = min(width, x + w), min(height, y + h)
        if x1 <= x0 or y1 <= y0:
            return 0, 0, width, height
        return x0, y0, x1, y1

    def _apply_region(self, pixels):
        height, width = pixels.shape[:2]
        x0, y0, x1, y1 = self._region_bounds(width, height)
        if (x0, y0, x1, y1) == (0, 0, width, height):
            return pixels
        return pixels[y0:y1, x0:x1].copy()


__all__ = ["RetinaCapture", "CaptureError", "CaptureUnavailable", "AccessLost", "enable_dpi_awareness", "Rect"]
