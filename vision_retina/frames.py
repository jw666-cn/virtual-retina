"""RetinaFrame：视网膜对外的统一帧结构（任务书 §4.3）。

约定：
- pixels 为 HxWx3 uint8 RGB，行优先、连续内存；
- timestamp 为 epoch 毫秒；
- dirtyRects 空数组表示整帧；
- 帧对象一经创建不再修改（pixels 只读使用），避免采集线程与消费线程竞争。
"""
from __future__ import annotations

import io
import time
from dataclasses import dataclass, field
from typing import Any, Iterable

import numpy as np


@dataclass(frozen=True)
class Rect:
    """像素矩形（左上角 + 宽高）。"""

    x: int
    y: int
    w: int
    h: int

    @property
    def area(self) -> int:
        return max(0, self.w) * max(0, self.h)

    @property
    def right(self) -> int:
        return self.x + self.w

    @property
    def bottom(self) -> int:
        return self.y + self.h

    def to_dict(self) -> dict[str, int]:
        return {"x": self.x, "y": self.y, "w": self.w, "h": self.h}

    def __iter__(self):
        """允许 tuple(rect) / list(rect) 等用法，兼容 (x, y, w, h) 接口。"""
        yield self.x
        yield self.y
        yield self.w
        yield self.h

    def __len__(self) -> int:
        return 4

    def __getitem__(self, index: int) -> int:
        return (self.x, self.y, self.w, self.h)[index]

    @classmethod
    def from_ltrb(cls, left: int, top: int, right: int, bottom: int) -> "Rect":
        return cls(int(left), int(top), max(0, int(right) - int(left)), max(0, int(bottom) - int(top)))

    def clip(self, width: int, height: int) -> "Rect":
        x0 = max(0, min(self.x, width))
        y0 = max(0, min(self.y, height))
        x1 = max(x0, min(self.right, width))
        y1 = max(y0, min(self.bottom, height))
        return Rect(x0, y0, x1 - x0, y1 - y0)

    def union(self, other: "Rect") -> "Rect":
        x0, y0 = min(self.x, other.x), min(self.y, other.y)
        x1, y1 = max(self.right, other.right), max(self.bottom, other.bottom)
        return Rect.from_ltrb(x0, y0, x1, y1)

    def intersects(self, other: "Rect") -> bool:
        return not (self.right <= other.x or other.right <= self.x or self.bottom <= other.y or other.bottom <= self.y)

    def expand(self, margin: int) -> "Rect":
        return Rect(self.x - margin, self.y - margin, self.w + 2 * margin, self.h + 2 * margin)


def merge_rects(rects: Iterable[Rect], gap: int = 0) -> list[Rect]:
    """合并重叠/相邻矩形，减少增量区域碎片（O(n^2)，帧内矩形通常 < 百级）。"""
    pending = [r for r in rects if r.area > 0]
    merged: list[Rect] = []
    while pending:
        current = pending.pop()
        changed = True
        while changed:
            changed = False
            grown = current.expand(gap)
            for other in list(pending):
                if grown.intersects(other):
                    current = current.union(other)
                    pending.remove(other)
                    changed = True
                    break
        merged.append(current)
    merged.sort(key=lambda r: (r.y, r.x))
    return merged


@dataclass
class RetinaFrame:
    frame_id: int
    timestamp_ms: int
    source: str
    pixels: np.ndarray
    dirty_rects: list[Rect] = field(default_factory=list)
    is_key_frame: bool = True
    meta: dict[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        if not isinstance(self.pixels, np.ndarray):
            raise TypeError("pixels 必须是 numpy 数组")
        if self.pixels.ndim != 3 or self.pixels.shape[2] != 3:
            raise ValueError(f"pixels 期望 HxWx3，实际 {self.pixels.shape}")
        if self.pixels.dtype != np.uint8:
            raise ValueError(f"pixels 期望 uint8，实际 {self.pixels.dtype}")
        if self.source not in ("desktop", "camera"):
            raise ValueError(f"source 只支持 desktop/camera，实际 {self.source}")
        if not self.pixels.flags["C_CONTIGUOUS"]:
            self.pixels = np.ascontiguousarray(self.pixels)
        # 用只读视图，避免把调用方传入的数组本身改成只读
        readonly = self.pixels.view()
        readonly.setflags(write=False)
        self.pixels = readonly

    @property
    def width(self) -> int:
        return int(self.pixels.shape[1])

    @property
    def height(self) -> int:
        return int(self.pixels.shape[0])

    @property
    def dirty_area(self) -> int:
        if not self.dirty_rects:
            return self.width * self.height
        return sum(r.clip(self.width, self.height).area for r in self.dirty_rects)

    def crop(self, region: tuple[int, int, int, int]) -> "RetinaFrame":
        x, y, w, h = region
        x0, y0 = max(0, x), max(0, y)
        x1, y1 = min(self.width, x + w), min(self.height, y + h)
        pixels = np.ascontiguousarray(self.pixels[y0:y1, x0:x1])
        return RetinaFrame(
            frame_id=self.frame_id,
            timestamp_ms=self.timestamp_ms,
            source=self.source,
            pixels=pixels,
            dirty_rects=[r for r in (rect.clip(x1 - x0, y1 - y0) for rect in self.dirty_rects) if r.area > 0],
            is_key_frame=self.is_key_frame,
            meta={**self.meta, "cropped_from": [x0, y0, x1 - x0, y1 - y0]},
        )

    def to_dict(self) -> dict[str, Any]:
        return {
            "frameId": self.frame_id,
            "timestamp": self.timestamp_ms,
            "source": self.source,
            "width": self.width,
            "height": self.height,
            "dirtyRects": [r.to_dict() for r in self.dirty_rects],
            "isKeyFrame": self.is_key_frame,
            "meta": {k: v for k, v in self.meta.items() if isinstance(v, (int, float, str, bool, list, dict)) or v is None},
        }

    # ---- 编码（送给大脑的图像） ----

    def encode(self, fmt: str = "jpeg", quality: int = 80, max_width: int = 0) -> bytes:
        from PIL import Image

        image = Image.fromarray(self.pixels, mode="RGB")
        if max_width and image.width > max_width:
            ratio = max_width / image.width
            size = (max_width, max(1, round(image.height * ratio)))
            image = image.resize(size, Image.LANCZOS)
        buffer = io.BytesIO()
        if fmt.lower() == "png":
            image.save(buffer, format="PNG", optimize=True)
        else:
            image.save(buffer, format="JPEG", quality=int(quality), subsampling=1, optimize=True)
        return buffer.getvalue()

    def to_data_uri(self, fmt: str = "jpeg", quality: int = 80, max_width: int = 0) -> str:
        import base64

        payload = self.encode(fmt=fmt, quality=quality, max_width=max_width)
        return f"data:image/{'png' if fmt.lower() == 'png' else 'jpeg'};base64,{base64.b64encode(payload).decode('ascii')}"

    def save(self, path: str, fmt: str | None = None) -> str:
        from PIL import Image

        Image.fromarray(self.pixels, mode="RGB").save(path, format=(fmt or None))
        return path


def now_ms() -> int:
    return int(time.time() * 1000)


def frames_differ(a: np.ndarray, b: np.ndarray, threshold: int = 12) -> bool:
    """快速判断两帧是否存在超过阈值的像素差异（抽样加速）。"""
    if a.shape != b.shape:
        return True
    step = max(1, a.shape[0] // 128)
    sample_a = a[::step, ::step].astype(np.int16)
    sample_b = b[::step, ::step].astype(np.int16)
    return bool(np.max(np.abs(sample_a - sample_b)) > threshold)
