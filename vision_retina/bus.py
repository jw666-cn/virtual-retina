"""统一输入总线（任务书 §4）：把采集层的原始帧变成"有语义的观察结果"。

四个机制：
1. 帧间差分：优先使用采集层给的 dirty-rects（DXGI），否则按网格做像素差分；
2. 中央凹压缩：围绕焦点（默认鼠标）裁剪高分辨率区域，降低单帧 token；
3. 静默机制：画面无实质变化时返回 `silence`/`unchanged`，不产生图像数据；
4. 双源统一抽象：desktop 与 camera 产出同一种 `Observation`。

总线不持久化任何画面：默认只在内存里保留上一帧用于差分。
"""
from __future__ import annotations

import time
from dataclasses import dataclass, field
from typing import Any, Callable

import numpy as np

from .adaptive import AdaptiveController, SceneProfile
from .capture import CaptureError, RetinaCapture, create_capture
from .config import Config, load_config
from .frames import Rect, RetinaFrame, merge_rects, now_ms
from .winutil import cursor_position


@dataclass
class Observation:
    """一次观察的结果。"""

    kind: str                      # frame | unchanged | silence
    seq: int
    source: str
    frame: RetinaFrame | None = None          # 处理后的帧（可能是中央凹裁剪）
    raw_size: tuple[int, int] = (0, 0)       # 处理前尺寸
    dirty_rects: list[Rect] = field(default_factory=list)
    changed_ratio: float = 0.0
    is_key_frame: bool = False
    reason: str = ""
    fovea: dict[str, Any] | None = None
    silence_seconds: float = 0.0
    capture_dirty_rects: list[Rect] = field(default_factory=list)

    @property
    def has_image(self) -> bool:
        return self.frame is not None

    def to_dict(self, include_pixels: bool = False) -> dict[str, Any]:
        data: dict[str, Any] = {
            "kind": self.kind,
            "seq": self.seq,
            "source": self.source,
            "is_key_frame": self.is_key_frame,
            "changed_ratio": round(self.changed_ratio, 6),
            "dirty_rects": [r.to_dict() for r in self.dirty_rects],
            "capture_dirty_rects": [r.to_dict() for r in self.capture_dirty_rects],
            "raw_size": list(self.raw_size),
            "reason": self.reason,
            "fovea": self.fovea,
            "silence_seconds": round(self.silence_seconds, 3),
        }
        if self.frame is not None:
            data.update({
                "frame_id": self.frame.frame_id,
                "timestamp_ms": self.frame.timestamp_ms,
                "width": self.frame.width,
                "height": self.frame.height,
            })
            if include_pixels:
                data["pixels"] = self.frame.pixels
        return data


def _to_gray(pixels: np.ndarray) -> np.ndarray:
    # 用整数权重近似 BT.601，避免 float 大数组开销
    return (pixels[..., 0].astype(np.uint16) * 77
            + pixels[..., 1].astype(np.uint16) * 150
            + pixels[..., 2].astype(np.uint16) * 29) >> 8


def grid_diff(prev: np.ndarray, cur: np.ndarray, grid: int = 32,
              threshold: int = 12) -> tuple[list[Rect], float]:
    """按 grid×grid 分块比较两帧，返回变化矩形与变化比例。

    用块均值而不是逐像素，既抗噪又便宜；连通块合并成尽量少的矩形。
    """
    if prev is None or prev.shape != cur.shape:
        h, w = cur.shape[:2]
        return [Rect(0, 0, w, h)], 1.0
    h, w = cur.shape[:2]
    gh, gw = h // grid, w // grid
    if gh < 2 or gw < 2:
        diff = np.abs(_to_gray(prev).astype(np.int16) - _to_gray(cur).astype(np.int16))
        changed = bool((diff > threshold).any())
        rects = [Rect(0, 0, w, h)] if changed else []
        return rects, 1.0 if changed else 0.0

    a = _to_gray(prev[:gh * grid, :gw * grid]).reshape(gh, grid, gw, grid).mean(axis=(1, 3))
    b = _to_gray(cur[:gh * grid, :gw * grid]).reshape(gh, grid, gw, grid).mean(axis=(1, 3))
    mask = np.abs(a - b) > threshold
    ratio = float(mask.mean())
    if not mask.any():
        return [], 0.0
    rects = _mask_to_rects(mask, grid)
    return rects, ratio


def _mask_to_rects(mask: np.ndarray, grid: int) -> list[Rect]:
    """把布尔块掩码连通用 BFS 合并成外接矩形。"""
    gh, gw = mask.shape
    seen = np.zeros_like(mask, dtype=bool)
    rects: list[Rect] = []
    for sy in range(gh):
        for sx in range(gw):
            if not mask[sy, sx] or seen[sy, sx]:
                continue
            stack = [(sy, sx)]
            seen[sy, sx] = True
            min_y = max_y = sy
            min_x = max_x = sx
            while stack:
                y, x = stack.pop()
                min_y, max_y = min(min_y, y), max(max_y, y)
                min_x, max_x = min(min_x, x), max(max_x, x)
                for dy in (-1, 0, 1):
                    for dx in (-1, 0, 1):
                        ny, nx = y + dy, x + dx
                        if 0 <= ny < gh and 0 <= nx < gw and mask[ny, nx] and not seen[ny, nx]:
                            seen[ny, nx] = True
                            stack.append((ny, nx))
            rects.append(Rect(min_x * grid, min_y * grid,
                              (max_x - min_x + 1) * grid, (max_y - min_y + 1) * grid))
    return merge_rects(rects, gap=grid)


class RetinaBus:
    """采集 → 差分 → 中央凹 → 观察结果。"""

    def __init__(self, capture: RetinaCapture, config: Config | None = None,
                 clock: Callable[[], float] = time.monotonic):
        self.capture = capture
        self.config = config or load_config()
        bus_cfg = self.config.section("bus")
        self.mode = str(bus_cfg.get("mode", "full"))          # full | delta
        self.keyframe_interval_s = float(bus_cfg.get("keyframe_interval_s", 30))
        self.silence_timeout_s = float(bus_cfg.get("silence_timeout_s", 3.0))
        self.delta_threshold = int(bus_cfg.get("delta_threshold", 12))
        self.change_min_ratio = float(bus_cfg.get("change_min_ratio", 0.002))
        self.diff_grid = int(bus_cfg.get("diff_grid", 32))
        fovea = bus_cfg.get("fovea") or {}
        self.fovea_enabled = bool(fovea.get("enabled", False))
        self.fovea_focus = str(fovea.get("focus", "mouse"))
        self.fovea_radius = int(fovea.get("radius", 240))
        self.fovea_scale = float(fovea.get("scale", 0.66))
        self.fovea_mode = str(fovea.get("mode", "crop"))

        self.adaptive = AdaptiveController(self.config)
        self.scene_profile: SceneProfile | None = self.adaptive.current
        self._last_eval_at = 0.0
        self._last_emit_at = 0.0

        self._clock = clock
        self._prev: RetinaFrame | None = None
        self._last_key_at = 0.0
        self._last_change_at = 0.0
        self._seq = 0
        self._started_at = 0.0
        self._status: dict[str, Any] = {
            "observations": 0, "frames": 0, "unchanged": 0, "silences": 0,
            "key_frames": 0, "errors": 0,
        }

    # ---- 生命周期 ----

    @classmethod
    def create(cls, config: Config | None = None, source: str | None = None,
               backend: str | None = None) -> "RetinaBus":
        cfg = config or load_config()
        cap_cfg = cfg.section("capture")
        capture = create_capture(
            source or cap_cfg.get("source", "desktop"),
            backend or cap_cfg.get("backend", "auto"),
            config=cfg,
        )
        return cls(capture, cfg)

    def start(self) -> "RetinaBus":
        self.capture.start()
        self._started_at = self._clock()
        return self

    def close(self) -> None:
        self.capture.close()

    def __enter__(self) -> "RetinaBus":
        return self.start()

    def __exit__(self, *exc: Any) -> None:
        self.close()

    # ---- 观察 ----

    def observe(self, force_keyframe: bool = False, timeout: float = 1.5) -> Observation:
        frame = self.capture.get_latest(timeout=timeout)
        return self.evaluate(frame, force_keyframe=force_keyframe)

    def observe_latest(self, force_keyframe: bool = False) -> Observation | None:
        frames = self.capture.drain()
        if not frames:
            return None
        return self.evaluate(frames[-1], force_keyframe=force_keyframe)

    def evaluate(self, frame: RetinaFrame | None, force_keyframe: bool = False) -> Observation:
        """把一帧原始画面变成 Observation（无帧时按静默处理）。"""
        self._seq += 1
        seq = self._seq
        now = self._clock()
        self._status["observations"] += 1

        if frame is None:
            self._status["silences"] += 1
            return Observation(
                kind="silence", seq=seq, source=self.capture.source,
                reason="采集超时，没有新帧",
                silence_seconds=now - (self._last_change_at or now),
            )

        raw_h, raw_w = frame.pixels.shape[:2]
        capture_dirty = list(frame.dirty_rects)

        # ---- 1) 差分 ----
        if self._prev is None or self._prev.pixels.shape != frame.pixels.shape:
            dirty, ratio = [Rect(0, 0, raw_w, raw_h)], 1.0
        else:
            dirty, ratio = grid_diff(self._prev.pixels, frame.pixels,
                                     grid=self.diff_grid, threshold=self.delta_threshold)

        changed = ratio >= self.change_min_ratio

        # ---- 2) 自适应场景（仿暗适应）----
        dt = (now - self._last_eval_at) if self._last_eval_at else 0.0
        self._last_eval_at = now
        self.scene_profile = self.adaptive.update(dt, ratio, keyframe=force_keyframe)

        # ---- 3) 关键帧判定 ----
        is_key = bool(
            force_keyframe
            or self._prev is None
            or (now - self._last_key_at) >= self.keyframe_interval_s
        )

        # ---- 4) 静默 ----
        if changed:
            self._last_change_at = now

        if not changed and not is_key:
            idle = now - (self._last_change_at or now)
            if idle >= self.silence_timeout_s or self.mode == "delta":
                self._status["silences" if idle >= self.silence_timeout_s else "unchanged"] += 1
                self._prev = frame
                kind = "silence" if idle >= self.silence_timeout_s else "unchanged"
                return Observation(
                    kind=kind, seq=seq, source=frame.source,
                    raw_size=(raw_w, raw_h),
                    dirty_rects=dirty, changed_ratio=ratio,
                    is_key_frame=False,
                    reason=("画面静默：无实质变化" if kind == "silence"
                            else "无实质变化，增量模式不发送图像"),
                    silence_seconds=idle,
                    capture_dirty_rects=capture_dirty,
                )

        if is_key:
            self._last_key_at = now
            self._status["key_frames"] += 1

        # 静态场景降采样：变化很小但未到静默阈值时，按场景 fps 限流。
        if (self.scene_profile and self.scene_profile.name == "static"
                and not is_key and changed):
            min_interval = 1.0 / max(0.1, self.scene_profile.fps)
            if (now - self._last_emit_at) < min_interval:
                self._status["unchanged"] += 1
                self._prev = frame
                return Observation(
                    kind="unchanged", seq=seq, source=frame.source,
                    raw_size=(raw_w, raw_h), dirty_rects=dirty, changed_ratio=ratio,
                    is_key_frame=False, reason="静态场景：按自适应帧率降采样",
                    capture_dirty_rects=capture_dirty,
                )

        # ---- 5) 中央凹 ----
        fovea_meta = None
        pixels = frame.pixels
        out_dirty = dirty
        if self.fovea_enabled:
            pixels, fovea_meta = self._foveate(frame)
            if fovea_meta is not None:
                f = fovea_meta
                # dirty 裁剪到中央凹窗口，并平移到窗口坐标
                out_dirty = []
                for r in dirty:
                    x0 = max(r.x, f["x"]); y0 = max(r.y, f["y"])
                    x1 = min(r.x + r.w, f["x"] + f["w"]); y1 = min(r.y + r.h, f["y"] + f["h"])
                    if x1 > x0 and y1 > y0:
                        out_dirty.append(Rect(x0 - f["x"], y0 - f["y"], x1 - x0, y1 - y0))

        processed = RetinaFrame(
            frame_id=frame.frame_id,
            timestamp_ms=frame.timestamp_ms,
            source=frame.source,
            pixels=pixels,
            dirty_rects=out_dirty,
            is_key_frame=is_key,
            meta={**frame.meta, "bus_seq": seq, "fovea": fovea_meta,
                  "scene": self.scene_profile.to_dict() if self.scene_profile else None},
        )

        self._prev = frame
        self._last_emit_at = now
        self._status["frames"] += 1
        return Observation(
            kind="frame", seq=seq, source=frame.source, frame=processed,
            raw_size=(raw_w, raw_h), dirty_rects=out_dirty,
            changed_ratio=ratio, is_key_frame=is_key,
            fovea=fovea_meta, capture_dirty_rects=capture_dirty,
            reason="关键帧" if is_key else "增量帧",
        )

    # ---- 中央凹 ----

    def _foveate(self, frame: RetinaFrame) -> tuple[np.ndarray, dict[str, Any] | None]:
        h, w = frame.pixels.shape[:2]
        radius = max(16, self.fovea_radius)
        side = radius * 2
        if side >= min(h, w):
            return frame.pixels, None
        focus = self._focus_point(w, h)
        if focus is None:
            return frame.pixels, None
        cx, cy = focus
        x0 = min(max(0, cx - side // 2), w - side)
        y0 = min(max(0, cy - side // 2), h - side)
        crop = frame.pixels[y0:y0 + side, x0:x0 + side]
        scale = min(1.0, max(0.1, self.fovea_scale))
        out_w = max(1, int(round(side * scale)))
        out_h = max(1, int(round(side * scale)))
        if out_w != side or out_h != side:
            import cv2
            interp = cv2.INTER_AREA if scale < 1 else cv2.INTER_CUBIC
            crop = cv2.resize(crop, (out_w, out_h), interpolation=interp)
        # 局部坐标 → 采集图像坐标（采集区域偏移由调用方换算到屏幕坐标）
        meta = {
            "x": x0, "y": y0, "w": side, "h": side,
            "out_w": out_w, "out_h": out_h, "scale": scale,
            "focus": "mouse" if self.fovea_focus == "mouse" else self.fovea_focus,
            "focus_point": [cx, cy],
            "screen_offset": list(self.capture.region[:2]) if self.capture.region else [0, 0],
        }
        return crop, meta

    def _focus_point(self, w: int, h: int) -> tuple[int, int] | None:
        if self.fovea_focus != "mouse":
            return w // 2, h // 2
        pos = cursor_position()
        if pos is None:
            return w // 2, h // 2
        off_x, off_y = (self.capture.region[:2] if self.capture.region else (0, 0))
        x, y = pos[0] - off_x, pos[1] - off_y
        if 0 <= x < w and 0 <= y < h:
            return x, y
        return w // 2, h // 2

    # ---- 控制/状态 ----

    def set_region(self, region: tuple[int, int, int, int] | None) -> None:
        self.capture.set_region(region)
        # 区域改变后，队列里残留的旧尺寸帧必须丢弃，否则下一次 observe 会拿到旧区域画面。
        self.capture.drain()
        self._prev = None  # 区域变化后重新建立基线

    @property
    def region(self) -> tuple[int, int, int, int] | None:
        return self.capture.region

    def set_fovea(self, enabled: bool | None = None, radius: int | None = None,
                  scale: float | None = None) -> dict[str, Any]:
        if enabled is not None:
            self.fovea_enabled = bool(enabled)
        if radius is not None:
            self.fovea_radius = max(16, int(radius))
        if scale is not None:
            self.fovea_scale = min(1.0, max(0.1, float(scale)))
        return {"enabled": self.fovea_enabled, "radius": self.fovea_radius, "scale": self.fovea_scale}

    @property
    def counters(self) -> dict[str, int]:
        """总线计数器快照（observations/frames/unchanged/silences/key_frames/errors）。"""
        return dict(self._status)

    def status(self) -> dict[str, Any]:
        return {
            "mode": self.mode,
            "source": self.capture.source,
            "region": list(self.capture.region) if self.capture.region else None,
            "fovea": {"enabled": self.fovea_enabled, "focus": self.fovea_focus,
                      "radius": self.fovea_radius, "scale": self.fovea_scale},
            "adaptive": self.adaptive.status(),
            "silence_timeout_s": self.silence_timeout_s,
            "keyframe_interval_s": self.keyframe_interval_s,
            "uptime_s": round(self._clock() - self._started_at, 1) if self._started_at else 0,
            "counters": dict(self._status),
            "capture": self.capture.stats(),
        }

    def list_sources(self) -> list[dict[str, Any]]:
        from .capture import list_sources
        return list_sources()


__all__ = ["RetinaBus", "Observation", "grid_diff"]
