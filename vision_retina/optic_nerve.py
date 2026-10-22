"""视神经转化引擎（OpticNerve，V2.0）。

定位
----
视网膜（采集）与大脑（OpenAI 兼容视觉模型）之间的**实时转化层**：

    任何输入  →  对当前大脑模型最优的表示  →  更省 token、更容易看清

它不改变大脑接口，也不产生新的模型能力；只决定"把什么形态的输入交给大脑"。
继承 MarkItDown 0.1.6 的插件化框架（``DocumentConverter`` / ``register_converter`` /
优先级排序 / StreamInfo），并把结果类型从单一的 markdown 扩展为四类：

==================  ==========================================================
``markdown``        文档与文本（PDF/Word/PPT/Excel/邮件/EPUB/网页/纯文本）
``image``           视觉画面帧：增强像素 + 稀疏编码 + 元信息（**不转文本**）
``structured``      表格 / CSV / JSON / 代码大纲：行列与结构不丢
``frame``           视频关键帧：抽帧 + 时间戳 + 转写挂点
==================  ==========================================================

关键约束
--------
画面 **只** 输出增强像素 / 稀疏编码（VLM 吃像素）；markdown 只服务于文档与文本类。
``ResultKind.IMAGE`` / ``ResultKind.FRAME`` 的结果 ``markdown`` 恒为空串。

典型用法
--------
>>> from vision_retina.optic_nerve import OpticNerve, VisionOptions
>>> nerve = OpticNerve()                       # 内置转换器已注册
>>> nerve.convert("report.docx").markdown       # 文档 → markdown
>>> r = nerve.convert(frame)                    # RetinaFrame/np.ndarray → 增强帧
>>> r.image.tokens, r.tokens.ratio              # 帧 token 与压缩比
>>> brain.ask_images(r.image_bytes(), "画面里是什么？")

坐标：稀疏编码会把多个内容块拼成一张"马赛克"，模型看到的坐标与原始帧不同。
``r.meta["mapper"]``（``CoordinateMapper``）负责双向换算，指令式变焦据此还原。

开放扩展
--------
新内容类型 = 新增一个 ``DocumentConverter`` 子类并 ``register_converter``，
引擎本体（``OpticNerve.convert``）不动；帧类扩展实现 ``accepts_object`` /
``convert_object`` 即可。
"""

from __future__ import annotations

import base64
import binascii
import hashlib
import io
import json
import math
import mimetypes
import os
import re
import tempfile
import time
import zipfile
from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from enum import Enum
from typing import Any, BinaryIO, Callable, Iterable, Iterator, Sequence
from xml.etree import ElementTree as ET

import numpy as np

from .frames import Rect

try:  # 可选：ROI 复用（存在则用于前景掩码，不存在也能跑）
    from . import roi as _roi  # type: ignore
except Exception:  # pragma: no cover - 防御式导入
    _roi = None  # type: ignore[assignment]


__all__ = [
    # 结果类型
    "ResultKind",
    "DocumentConverterResult",
    "EncodedFrame",
    "StructuredPayload",
    "TokenReport",
    # 框架
    "StreamInfo",
    "DocumentConverter",
    "ConverterRegistration",
    "PRIORITY_SPECIFIC",
    "PRIORITY_STRUCTURED",
    "PRIORITY_DOCUMENT",
    "PRIORITY_PLAIN_TEXT",
    "PRIORITY_FALLBACK",
    # 视觉
    "VisionConverter",
    "VisionOptions",
    "CoordinateMapper",
    "BlockPlacement",
    "ContentBlock",
    "detect_blocks",
    "enhance_pixels",
    "readability",
    "estimate_image_tokens",
    "estimate_text_tokens",
    # 其他转换器
    "StructuredConverter",
    "CodeConverter",
    "VideoConverter",
    "PlainTextConverter",
    "MarkdownConverter",
    "HtmlConverter",
    "DocxConverter",
    "XlsxConverter",
    "PptxConverter",
    "PdfConverter",
    "EpubConverter",
    "ImageDocumentConverter",
    "FallbackConverter",
    # 引擎
    "OpticNerve",
    "OpticNerveStream",
    "default_engine",
    # 异常
    "OpticNerveError",
    "UnsupportedContentError",
    "MissingDependencyException",
]


# ===========================================================================
# 0. 通用工具：token 估算 / 数组归一 / 编解码 / 缩放
# ===========================================================================

#: 图像 token 估算口径（与 service.estimate_image_tokens 保持一致）
#: 570 * sqrt(百万像素)，下限 48；这是主流 VLM 的 patch 计费近似。
_IMAGE_TOKEN_COEF = 570.0
_IMAGE_TOKEN_MIN = 48
_TEXT_CHARS_PER_TOKEN = 3.6  # 中英混排经验值（保守）


def estimate_image_tokens(width: int, height: int) -> int:
    """估算一张图送给 VLM 的 token 数（与既有编码链路口径一致）。"""
    global _service_estimate_tokens
    if _service_estimate_tokens is None:
        try:  # 延迟导入，避免包内循环依赖
            from .service import estimate_image_tokens as _impl  # type: ignore

            _service_estimate_tokens = _impl
        except Exception:
            _service_estimate_tokens = False
    if callable(_service_estimate_tokens):
        try:
            return int(_service_estimate_tokens(width, height))
        except Exception:
            pass
    if width <= 0 or height <= 0:
        return 0
    px = float(width) * float(height)
    return max(_IMAGE_TOKEN_MIN, int(_IMAGE_TOKEN_COEF * math.sqrt(px / 1_000_000.0)))


_service_estimate_tokens: Any = None


def estimate_text_tokens(text: str) -> int:
    """估算文本 token 数（中英混排，偏保守）。"""
    if not text:
        return 0
    return max(1, int(math.ceil(len(text) / _TEXT_CHARS_PER_TOKEN)))


def _rect(x: float, y: float, w: float, h: float) -> Rect:
    """构造 ``Rect``（统一走关键字，避免上游字段顺序差异）。"""
    return Rect(x=int(round(x)), y=int(round(y)), w=int(round(w)), h=int(round(h)))


def _clip_rect(r: Rect, width: int, height: int) -> Rect:
    x1 = max(0, min(int(r.x), width))
    y1 = max(0, min(int(r.y), height))
    x2 = max(x1, min(int(r.x) + int(r.w), width))
    y2 = max(y1, min(int(r.y) + int(r.h), height))
    return _rect(x1, y1, x2 - x1, y2 - y1)


def _intersect(a: Rect, b: Rect) -> Rect:
    x1, y1 = max(a.x, b.x), max(a.y, b.y)
    x2 = min(a.x + a.w, b.x + b.w)
    y2 = min(a.y + a.h, b.y + b.h)
    return _rect(x1, y1, max(0, x2 - x1), max(0, y2 - y1))


def _area(r: Rect) -> int:
    return max(0, int(r.w)) * max(0, int(r.h))


def _as_bgr(pixels: Any) -> np.ndarray:
    """把各种像素输入归一为 ``HxWx3 uint8`` BGR。

    约定：``np.ndarray`` 按 OpenCV 习惯视为 BGR；PIL.Image 视为 RGB。
    """
    if pixels is None:
        raise UnsupportedContentError("像素为空")
    if hasattr(pixels, "convert") and hasattr(pixels, "size"):  # PIL.Image
        if pixels.mode != "RGB":
            pixels = pixels.convert("RGB")
        arr = np.asarray(pixels)
        return np.ascontiguousarray(arr[:, :, ::-1])
    arr = np.asarray(pixels)
    if arr.dtype != np.uint8:
        if arr.dtype.kind == "f":
            arr = np.clip(arr * (255.0 if float(arr.max(initial=0.0)) <= 1.0 else 1.0), 0, 255)
        arr = arr.astype(np.uint8)
    if arr.ndim == 2:
        arr = np.stack([arr, arr, arr], axis=-1)
    elif arr.ndim == 3 and arr.shape[2] == 4:
        arr = arr[:, :, :3]
    elif arr.ndim == 3 and arr.shape[2] == 1:
        arr = np.repeat(arr, 3, axis=2)
    if arr.ndim != 3 or arr.shape[2] < 3:
        raise UnsupportedContentError(f"不支持的像素形状：{arr.shape}")
    return np.ascontiguousarray(arr[:, :, :3])


def _luma(bgr: np.ndarray) -> np.ndarray:
    b = bgr[:, :, 0].astype(np.float32)
    g = bgr[:, :, 1].astype(np.float32)
    r = bgr[:, :, 2].astype(np.float32)
    return 0.114 * b + 0.587 * g + 0.299 * r


def _box_blur(gray: np.ndarray, k: int = 3) -> np.ndarray:
    """积分图实现的快速均值模糊（无 cv2 也可用）。"""
    k = max(1, int(k) | 1)
    pad = k // 2
    g = np.asarray(gray, dtype=np.float32)
    if k <= 1:
        return g.copy()
    p = np.pad(g, ((pad, pad), (pad, pad)), mode="edge")
    integral = np.zeros((p.shape[0] + 1, p.shape[1] + 1), dtype=np.float64)
    np.cumsum(np.cumsum(p, axis=0, dtype=np.float64), axis=1, out=integral[1:, 1:])
    h, w = g.shape
    y1 = np.arange(h)[:, None]
    x1 = np.arange(w)[None, :]
    y2, x2 = y1 + k, x1 + k
    total = integral[y2, x2] - integral[y1, x2] - integral[y2, x1] + integral[y1, x1]
    return (total / (k * k)).astype(np.float32)


def _resize(bgr: np.ndarray, width: int, height: int) -> np.ndarray:
    width = max(1, int(width))
    height = max(1, int(height))
    if bgr.shape[1] == width and bgr.shape[0] == height:
        return bgr
    try:
        import cv2  # type: ignore

        interp = cv2.INTER_AREA if (width * height) < (bgr.shape[0] * bgr.shape[1]) else cv2.INTER_CUBIC
        return cv2.resize(bgr, (width, height), interpolation=interp)
    except Exception:
        pass
    try:
        from PIL import Image  # type: ignore

        img = Image.fromarray(bgr[:, :, ::-1])
        img = img.resize((width, height), Image.LANCZOS)
        return np.ascontiguousarray(np.asarray(img)[:, :, ::-1])
    except Exception:
        pass
    # 最近邻兜底
    ys = np.clip(np.linspace(0, bgr.shape[0] - 1, height).round().astype(int), 0, bgr.shape[0] - 1)
    xs = np.clip(np.linspace(0, bgr.shape[1] - 1, width).round().astype(int), 0, bgr.shape[1] - 1)
    return np.ascontiguousarray(bgr[ys][:, xs])


def encode_image(bgr: np.ndarray, *, mimetype: str = "image/jpeg", quality: int = 82) -> bytes:
    """把 BGR 数组编码为字节（jpeg/png），cv2 → PIL 依次降级。"""
    bgr = _as_bgr(bgr)
    fmt = "png" if "png" in mimetype else "jpeg"
    try:
        import cv2  # type: ignore

        if fmt == "jpeg":
            ok, buf = cv2.imencode(".jpg", bgr, [int(cv2.IMWRITE_JPEG_QUALITY), int(quality)])
        else:
            ok, buf = cv2.imencode(".png", bgr, [int(cv2.IMWRITE_PNG_COMPRESSION), 3])
        if ok:
            return buf.tobytes()
    except Exception:
        pass
    try:
        from PIL import Image  # type: ignore

        img = Image.fromarray(bgr[:, :, ::-1])
        out = io.BytesIO()
        if fmt == "jpeg":
            img.save(out, format="JPEG", quality=int(quality), optimize=True)
        else:
            img.save(out, format="PNG", optimize=True)
        return out.getvalue()
    except Exception as exc:  # pragma: no cover - 无图像库的极端环境
        raise MissingDependencyException("需要 opencv-python 或 pillow 才能编码图像") from exc


def decode_image(data: bytes) -> np.ndarray:
    """把图片字节解码为 BGR 数组（cv2 → PIL 依次降级）。"""
    try:
        import cv2  # type: ignore

        arr = cv2.imdecode(np.frombuffer(data, dtype=np.uint8), cv2.IMREAD_COLOR)
        if arr is not None:
            return arr
    except Exception:
        pass
    try:
        from PIL import Image  # type: ignore

        img = Image.open(io.BytesIO(data))
        if img.mode != "RGB":
            img = img.convert("RGB")
        return np.ascontiguousarray(np.asarray(img)[:, :, ::-1])
    except Exception as exc:
        raise UnsupportedContentError("无法解码图片（需要 opencv-python 或 pillow）") from exc


def _sha8(data: bytes) -> str:
    return hashlib.blake2b(data, digest_size=8).hexdigest()


# ===========================================================================
# 1. 保留：MarkItDown 插件化骨架
# ===========================================================================


@dataclass(frozen=True, kw_only=True)
class StreamInfo:
    """输入流的类型与来源描述（保留 MarkItDown 语义，供 accepts 判定）。

    字段与 MarkItDown 0.1.6 一致：mimetype / extension / charset / filename /
    local_path / url。子类或引擎可通过 :meth:`copy_and_update` 派生变体。
    """

    mimetype: str = ""
    extension: str | None = None
    charset: str | None = None
    filename: str | None = None
    local_path: str | None = None
    url: str | None = None

    def copy_and_update(self, *args: Any, **kwargs: Any) -> "StreamInfo":
        """返回替换若干字段后的新 StreamInfo（不可变语义）。"""
        if args:
            raise TypeError("copy_and_update 只接受关键字参数")
        allowed = {f for f in self.__dataclass_fields__}  # type: ignore[attr-defined]
        unknown = set(kwargs) - allowed
        if unknown:
            raise TypeError(f"未知字段: {sorted(unknown)}")
        return dataclass_replace(self, **kwargs)


def dataclass_replace(obj: Any, **kwargs: Any) -> Any:
    """轻量版 ``dataclasses.replace``（对 frozen dataclass 同样适用）。"""
    import dataclasses

    return dataclasses.replace(obj, **kwargs)


def _coerce_stream_info(value: Any) -> StreamInfo | None:
    """把 dict 之类的宽松类型收成 StreamInfo（公开 API 容错）。"""
    if value is None or isinstance(value, StreamInfo):
        return value
    if isinstance(value, dict):
        allowed = set(StreamInfo.__dataclass_fields__)  # type: ignore[attr-defined]
        return StreamInfo(**{k: v for k, v in value.items() if k in allowed})
    return value  # 交给调用方，保持原有报错行为


def _looks_like_observation(value: Any) -> bool:
    """鸭子类型识别 ``bus.Observation``（避免引擎 import 采集层造成循环依赖）。"""
    return (
        hasattr(value, "kind")
        and hasattr(value, "seq")
        and (hasattr(value, "frame") or hasattr(value, "source"))
        and not isinstance(value, np.ndarray)
    )


@dataclass(frozen=True)
class ConverterRegistration:
    """一条注册记录：转换器 + 优先级（数值越小越先尝试）。"""

    converter: "DocumentConverter"
    priority: float

    @property
    def name(self) -> str:
        return getattr(self.converter, "name", type(self.converter).__name__)


class DocumentConverter(ABC):
    """转换器抽象基类（MarkItDown 兼容的 accepts/convert 契约）。

    扩展点
    ------
    * 流式输入：实现 :meth:`accepts` / :meth:`convert`（文档、图片、视频文件）。
    * 对象输入：实现 :meth:`accepts_object` / :meth:`convert_object`
      （视网膜帧、numpy 数组、dict/list 等，无需落盘）。
    """

    #: 人类可读名字，注册表与结果里会带出
    name: str = "converter"

    #: 兜底型转换器（"什么都能接"）：只在所有专用转换器都失败后才尝试，
    #: 避免它在嗅探出的高可信候选之前抢走内容。
    generic: bool = False

    def accepts(self, file_stream: BinaryIO, stream_info: StreamInfo, **kwargs: Any) -> bool:
        """该转换器是否接受这个流（默认拒绝）。"""
        return False

    @abstractmethod
    def convert(
        self, file_stream: BinaryIO, stream_info: StreamInfo, **kwargs: Any
    ) -> "DocumentConverterResult":
        """把流转换为结果（必须实现）。"""
        raise NotImplementedError

    # --- 对象扩展点（引擎本体不为此改动）------------------------------------
    def accepts_object(self, obj: Any, stream_info: StreamInfo, **kwargs: Any) -> bool:
        """是否接受内存对象（帧/数组/结构化数据）。默认不接受。"""
        return False

    def convert_object(
        self, obj: Any, stream_info: StreamInfo, **kwargs: Any
    ) -> "DocumentConverterResult":
        """转换内存对象；未实现则报错。"""
        raise UnsupportedContentError(
            f"{getattr(self, 'name', type(self).__name__)} 不支持对象输入"
        )


#: 具体的单类型转换（视觉帧、表格、代码……）
PRIORITY_SPECIFIC = 0.0
#: 结构化数据（CSV/JSON/XLSX）
PRIORITY_STRUCTURED = 10.0
#: 富文档（docx/pptx/pdf/epub/html）
PRIORITY_DOCUMENT = 50.0
#: 纯文本 / markdown
PRIORITY_PLAIN_TEXT = 300.0
#: 兜底
PRIORITY_FALLBACK = 1000.0


# ===========================================================================
# 2. 结果类型：markdown / image / structured / frame 四类
# ===========================================================================


class ResultKind(str, Enum):
    """结果的四种形态。"""

    MARKDOWN = "markdown"
    IMAGE = "image"
    STRUCTURED = "structured"
    FRAME = "frame"

    @property
    def is_visual(self) -> bool:
        return self in (ResultKind.IMAGE, ResultKind.FRAME)


@dataclass
class TokenReport:
    """一次转化的 token 账本（原口径 vs 输出口径）。"""

    original: int = 0
    output: int = 0
    strategy: str = ""
    detail: dict[str, Any] = field(default_factory=dict)

    @property
    def ratio(self) -> float:
        if self.original <= 0:
            return 1.0 if self.output <= 0 else 0.0
        return round(self.output / float(self.original), 4)

    @property
    def saved(self) -> int:
        return max(0, int(self.original) - int(self.output))

    @property
    def saved_ratio(self) -> float:
        return round(1.0 - self.ratio, 4) if self.original > 0 else 0.0

    def to_dict(self) -> dict[str, Any]:
        return {
            "original": self.original,
            "output": self.output,
            "ratio": self.ratio,
            "saved": self.saved,
            "saved_ratio": self.saved_ratio,
            "strategy": self.strategy,
            "detail": dict(self.detail),
        }


@dataclass
class EncodedFrame:
    """一帧已编码的视觉载荷（增强像素 + 裁剪/稀疏信息）。

    ``data`` 是可直接进 OpenAI ``image_url`` 的 jpeg/png 字节；
    ``rect`` 是它在**原始帧**中对应的区域（稀疏编码后会小于整帧）。
    """

    data: bytes
    width: int
    height: int
    mimetype: str = "image/jpeg"
    rect: Rect | None = None
    scale: float = 1.0
    tokens: int = 0
    strategy: str = "full"
    meta: dict[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        if not self.tokens:
            self.tokens = estimate_image_tokens(self.width, self.height)

    @property
    def pixel_count(self) -> int:
        return int(self.width) * int(self.height)

    def to_base64(self) -> str:
        return base64.b64encode(self.data).decode("ascii")

    def to_data_uri(self) -> str:
        return f"data:{self.mimetype};base64,{self.to_base64()}"

    def to_openai_part(self) -> dict[str, Any]:
        """OpenAI 兼容的多模态内容块。"""
        return {"type": "image_url", "image_url": {"url": self.to_data_uri()}}

    def save(self, path: str | os.PathLike[str]) -> str:
        path = os.fspath(path)
        parent = os.path.dirname(os.path.abspath(path))
        if parent:
            os.makedirs(parent, exist_ok=True)
        with open(path, "wb") as fh:
            fh.write(self.data)
        return path

    def to_dict(self) -> dict[str, Any]:
        return {
            "width": self.width,
            "height": self.height,
            "mimetype": self.mimetype,
            "rect": [self.rect.x, self.rect.y, self.rect.w, self.rect.h] if self.rect else None,
            "scale": round(float(self.scale), 4),
            "tokens": self.tokens,
            "strategy": self.strategy,
            "bytes": len(self.data),
            "meta": dict(self.meta),
        }


@dataclass
class StructuredPayload:
    """结构化表示：行列/记录关系不丢，可再渲染成 markdown 或紧凑文本。

    ``format``：

    * ``table``      —— 列名 + 行（CSV/TSV/XLSX）
    * ``records``    —— 记录列表（JSON/JSONL）
    * ``key_value``  —— 扁平键值（配置/单对象）
    * ``code``       —— 代码大纲（符号表）
    * ``sheets``     —— 多工作表（XLSX）
    * ``envelope``   —— 容器（ZIP 等）
    """

    format: str = "table"
    columns: list[str] = field(default_factory=list)
    rows: list[list[Any]] = field(default_factory=list)
    records: list[dict[str, Any]] = field(default_factory=list)
    schema: dict[str, Any] = field(default_factory=dict)
    truncated: bool = False
    meta: dict[str, Any] = field(default_factory=dict)

    @property
    def shape(self) -> tuple[int, int]:
        if self.rows:
            return len(self.rows), max(len(r) for r in self.rows) if self.rows else 0
        return len(self.records), len(self.columns)

    def to_records(self) -> list[dict[str, Any]]:
        if self.records:
            return [dict(r) for r in self.records]
        out: list[dict[str, Any]] = []
        for row in self.rows:
            out.append({c: (row[i] if i < len(row) else None) for i, c in enumerate(self.columns)})
        return out

    def to_compact(self, sep: str = "\t") -> str:
        """紧凑文本（比 markdown 管道表省 30%~50% token）。"""
        if self.format in ("records", "key_value") or (not self.rows and self.records):
            cols = self.columns or sorted({k for r in self.records for k in r})
            lines = [sep.join(str(c) for c in cols)]
            for rec in self.records:
                lines.append(sep.join("" if rec.get(c) is None else str(rec.get(c)) for c in cols))
            return "\n".join(lines)
        cols = [str(c) for c in self.columns]
        lines = [sep.join(cols)] if cols else []
        lines += [sep.join("" if v is None else str(v) for v in row) for row in self.rows]
        return "\n".join(lines)

    def to_markdown(self) -> str:
        """渲染为 markdown（管道表；记录型退化为键值列表）。"""
        if self.format in ("records", "key_value") and self.records:
            cols = self.columns or sorted({k for rec in self.records for k in rec})
            if not cols:
                return ""
            head = "| " + " | ".join(str(c) for c in cols) + " |"
            sep = "| " + " | ".join("---" for _ in cols) + " |"
            body = [
                "| " + " | ".join(_md_cell(rec.get(c)) for c in cols) + " |" for rec in self.records
            ]
            return "\n".join([head, sep] + body)
        if not self.columns:
            return ""
        head = "| " + " | ".join(str(c) for c in self.columns) + " |"
        sep = "| " + " | ".join("---" for _ in self.columns) + " |"
        body = ["| " + " | ".join(_md_cell(v) for v in row) + " |" for row in self.rows]
        return "\n".join([head, sep] + body)

    def to_dict(self) -> dict[str, Any]:
        return {
            "format": self.format,
            "columns": list(self.columns),
            "rows": [list(r) for r in self.rows],
            "records": [dict(r) for r in self.records],
            "schema": dict(self.schema),
            "truncated": self.truncated,
            "shape": list(self.shape),
            "meta": dict(self.meta),
        }


def _md_cell(value: Any) -> str:
    if value is None:
        return ""
    text = str(value).replace("\r", " ").replace("\n", "<br>").replace("|", "\\|")
    return text.strip()


@dataclass
class DocumentConverterResult:
    """转化结果（四类形态共用一个容器）。

    与 MarkItDown 兼容：``.markdown`` / ``.title`` / ``.text_content`` / ``str(res)``。
    视觉结果额外带 ``.image`` / ``.images`` / ``.frames``，``markdown`` 为空。
    """

    kind: ResultKind = ResultKind.MARKDOWN
    markdown: str = ""
    title: str | None = None
    image: EncodedFrame | None = None
    images: list[EncodedFrame] = field(default_factory=list)
    structured: StructuredPayload | None = None
    frames: list[EncodedFrame] = field(default_factory=list)
    meta: dict[str, Any] = field(default_factory=dict)
    source: str | None = None
    converter: str | None = None
    tokens: TokenReport | None = None

    # --- MarkItDown 兼容面 --------------------------------------------------
    @property
    def text_content(self) -> str:
        return self.markdown

    def __str__(self) -> str:
        return self.markdown

    def __bool__(self) -> bool:
        return bool(self.markdown or self.image or self.images or self.structured or self.frames)

    # --- 与大脑的接口 -------------------------------------------------------
    @property
    def is_visual(self) -> bool:
        return self.kind.is_visual

    def all_frames(self) -> list[EncodedFrame]:
        """按"低清→高清"顺序返回全部视觉载荷。"""
        out: list[EncodedFrame] = []
        if self.image is not None:
            out.append(self.image)
        out.extend(self.images)
        out.extend(self.frames)
        return out

    def image_bytes(self) -> list[bytes]:
        """直接喂 ``brain.ask_images()`` 的字节列表。"""
        return [f.data for f in self.all_frames()]

    def to_openai_content(self, prompt: str = "") -> list[dict[str, Any]]:
        """拼成 OpenAI 兼容的 user content 数组（文本 + 图）。"""
        parts: list[dict[str, Any]] = []
        if prompt:
            parts.append({"type": "text", "text": prompt})
        for frame in self.all_frames():
            parts.append(frame.to_openai_part())
        if self.markdown and self.kind == ResultKind.MARKDOWN:
            parts.append({"type": "text", "text": self.markdown})
        return parts

    @property
    def mapper(self) -> "CoordinateMapper | None":
        mapper = self.meta.get("mapper")
        return mapper if isinstance(mapper, CoordinateMapper) else None

    def to_dict(self, include_payload: bool = False) -> dict[str, Any]:
        data: dict[str, Any] = {
            "kind": self.kind.value,
            "markdown": self.markdown if self.kind == ResultKind.MARKDOWN else "",
            "title": self.title,
            "source": self.source,
            "converter": self.converter,
            "tokens": self.tokens.to_dict() if self.tokens else None,
            "meta": {k: v for k, v in self.meta.items() if k != "mapper"},
        }
        if self.image is not None:
            data["image"] = self.image.to_dict()
        if self.images:
            data["images"] = [f.to_dict() for f in self.images]
        if self.frames:
            data["frames"] = [f.to_dict() for f in self.frames]
        if self.structured is not None:
            data["structured"] = self.structured.to_dict()
        data["meta"]["has_mapper"] = self.mapper is not None
        if include_payload:
            data["image_bytes"] = [base64.b64encode(f.data).decode("ascii") for f in self.all_frames()]
        return data


# ===========================================================================
# 3. 坐标映射（稀疏编码后的"马赛克坐标" ↔ 原始帧坐标）
# ===========================================================================


@dataclass
class BlockPlacement:
    """一个内容块在马赛克画布上的位置与缩放。"""

    index: int
    native: Rect
    mosaic: Rect
    scale: float
    salience: float = 0.0

    def mosaic_to_native_point(self, x: float, y: float) -> tuple[float, float]:
        sx = (x - self.mosaic.x) / max(1e-6, self.scale)
        sy = (y - self.mosaic.y) / max(1e-6, self.scale)
        return self.native.x + sx, self.native.y + sy

    def native_to_mosaic_point(self, x: float, y: float) -> tuple[float, float]:
        return (
            self.mosaic.x + (x - self.native.x) * self.scale,
            self.mosaic.y + (y - self.native.y) * self.scale,
        )

    def to_dict(self) -> dict[str, Any]:
        return {
            "index": self.index,
            "native": [self.native.x, self.native.y, self.native.w, self.native.h],
            "mosaic": [self.mosaic.x, self.mosaic.y, self.mosaic.w, self.mosaic.h],
            "scale": round(float(self.scale), 4),
            "salience": round(float(self.salience), 4),
        }


@dataclass
class CoordinateMapper:
    """稀疏编码画布与原始帧之间的双向坐标换算。

    模型看到的是"马赛克坐标"；指令式变焦（VLM 返回 ``[x, y, w, h]``）必须经
    :meth:`to_native` 还原后才能去原始帧上裁剪，否则会裁错位置。
    """

    mosaic_width: int
    mosaic_height: int
    native_width: int
    native_height: int
    placements: list[BlockPlacement] = field(default_factory=list)
    strategy: str = "sparse"

    def block_at(self, x: float, y: float) -> BlockPlacement | None:
        for p in self.placements:
            if p.mosaic.x <= x < p.mosaic.x + p.mosaic.w and p.mosaic.y <= y < p.mosaic.y + p.mosaic.h:
                return p
        return None

    def _nearest(self, cx: float, cy: float) -> BlockPlacement | None:
        best: BlockPlacement | None = None
        best_d = float("inf")
        for p in self.placements:
            dx = (p.mosaic.x + p.mosaic.w / 2.0) - cx
            dy = (p.mosaic.y + p.mosaic.h / 2.0) - cy
            d = dx * dx + dy * dy
            if d < best_d:
                best, best_d = p, d
        return best

    def to_native_point(self, x: float, y: float) -> tuple[int, int] | None:
        """马赛克坐标 → 原始帧坐标（落在块内则精确换算，否则就近吸附）。"""
        p = self.block_at(x, y) or self._nearest(x, y)
        if p is None:
            return None
        nx, ny = p.mosaic_to_native_point(x, y)
        nx = max(0.0, min(float(self.native_width), nx))
        ny = max(0.0, min(float(self.native_height), ny))
        return int(round(nx)), int(round(ny))

    def to_native(self, rect: Sequence[float] | Rect) -> Rect | None:
        """马赛克矩形 → 原始帧矩形。

        跨越多个内容块时取各块映射结果的并集（因此可能包含块间空白），
        单个块内则精确换算。
        """
        r = _coerce_rect(rect)
        if r is None or not self.placements:
            return None
        x1, y1, x2, y2 = r.x, r.y, r.x + r.w, r.y + r.h
        mapped: list[Rect] = []
        for p in self.placements:
            ix = max(p.mosaic.x, x1)
            iy = max(p.mosaic.y, y1)
            ix2 = min(p.mosaic.x + p.mosaic.w, x2)
            iy2 = min(p.mosaic.y + p.mosaic.h, y2)
            if ix2 <= ix or iy2 <= iy:
                continue
            nx1, ny1 = p.mosaic_to_native_point(ix, iy)
            nx2, ny2 = p.mosaic_to_native_point(ix2, iy2)
            mapped.append(_rect(nx1, ny1, nx2 - nx1, ny2 - ny1))
        if not mapped:
            center = self.to_native_point((x1 + x2) / 2.0, (y1 + y2) / 2.0)
            if center is None:
                return None
            return _clip_rect(_rect(center[0] - r.w / 2, center[1] - r.h / 2, r.w, r.h),
                              self.native_width, self.native_height)
        ux1 = min(m.x for m in mapped)
        uy1 = min(m.y for m in mapped)
        ux2 = max(m.x + m.w for m in mapped)
        uy2 = max(m.y + m.h for m in mapped)
        return _clip_rect(_rect(ux1, uy1, ux2 - ux1, uy2 - uy1), self.native_width, self.native_height)

    def to_mosaic(self, rect: Sequence[float] | Rect) -> Rect | None:
        """原始帧矩形 → 马赛克矩形（用于把"看这里"回填到模型视图）。"""
        r = _coerce_rect(rect)
        if r is None or not self.placements:
            return None
        x1, y1, x2, y2 = r.x, r.y, r.x + r.w, r.y + r.h
        mapped: list[Rect] = []
        for p in self.placements:
            ix = max(p.native.x, x1)
            iy = max(p.native.y, y1)
            ix2 = min(p.native.x + p.native.w, x2)
            iy2 = min(p.native.y + p.native.h, y2)
            if ix2 <= ix or iy2 <= iy:
                continue
            mx1, my1 = p.native_to_mosaic_point(ix, iy)
            mx2, my2 = p.native_to_mosaic_point(ix2, iy2)
            mapped.append(_rect(mx1, my1, mx2 - mx1, my2 - my1))
        if not mapped:
            return None
        ux1 = min(m.x for m in mapped)
        uy1 = min(m.y for m in mapped)
        ux2 = max(m.x + m.w for m in mapped)
        uy2 = max(m.y + m.h for m in mapped)
        return _clip_rect(_rect(ux1, uy1, ux2 - ux1, uy2 - uy1), self.mosaic_width, self.mosaic_height)

    def coverage(self) -> float:
        """马赛克覆盖的原始面积比例（动态场景保真度指标）。"""
        if not self.placements or self.native_width <= 0 or self.native_height <= 0:
            return 0.0
        covered = sum(_area(_intersect(p.native, _rect(0, 0, self.native_width, self.native_height)))
                      for p in self.placements)
        return round(min(1.0, covered / float(self.native_width * self.native_height)), 4)

    def to_dict(self) -> dict[str, Any]:
        return {
            "mosaic": [self.mosaic_width, self.mosaic_height],
            "native": [self.native_width, self.native_height],
            "strategy": self.strategy,
            "coverage": self.coverage(),
            "blocks": [p.to_dict() for p in self.placements],
        }


def _coerce_rect(value: Sequence[float] | Rect | None) -> Rect | None:
    if value is None:
        return None
    if isinstance(value, Rect):
        return value if value.w > 0 and value.h > 0 else None
    try:
        x, y, w, h = (float(v) for v in value)  # type: ignore[misc]
    except Exception:
        return None
    if w <= 0 or h <= 0:
        return None
    return _rect(x, y, w, h)


# ===========================================================================
# 4. 画面增强（对比度 / 锐化 / 光照均衡 / 小字超分）
# ===========================================================================


@dataclass
class EnhanceOptions:
    """增强参数（默认值针对桌面截图与文档翻拍调过）。"""

    enabled: bool = True
    equalize: bool = True
    sharpen: float = 0.62
    target_std: float = 46.0
    clip_gain: float = 2.1
    stretch_percentile: float = 1.0
    min_text_px: int = 16
    max_scale_up: float = 2.0


def enhance_pixels(pixels: Any, opts: EnhanceOptions | None = None) -> np.ndarray:
    """画面增强：光照均衡 → 对比度拉伸 → 局部均衡 → 锐化。

    目标不是"好看"，而是让 VLM 更容易看清：提高局部对比度与笔画边缘能量，
    同时不改变内容（不裁剪、不改文字）。
    """
    opts = opts or EnhanceOptions()
    bgr = _as_bgr(pixels)
    if not opts.enabled:
        return bgr

    arr = bgr.astype(np.float32)
    gray = _luma(arr)

    # 1) 全局亮度拉伸：按分位数裁掉极端值，提整体对比
    p = float(np.clip(opts.stretch_percentile, 0.0, 10.0))
    lo, hi = np.percentile(gray, [p, 100.0 - p]) if p > 0 else (float(gray.min()), float(gray.max()))
    if hi - lo < 1e-3:
        lo, hi = float(gray.min()), float(gray.max())
    if hi - lo > 1e-3:
        arr = (arr - lo) * (255.0 / (hi - lo))

    # 2) 局部对比度均衡（CLAHE 的 numpy 等价实现）：按局部标准差放大细节
    if opts.equalize:
        k = 31
        local_mean = _box_blur(_luma(arr), k)[:, :, None]
        local_sq = _box_blur(_luma(arr) ** 2, k)[:, :, None]
        local_std = np.sqrt(np.maximum(local_sq - local_mean ** 2, 1e-6))
        # 只放大弱对比、绝不压低强对比：下界固定 1.0。
        # （旧实现允许 0.55，会把本来清晰的黑字白底"均衡"成灰字灰底）
        ceiling = max(1.0, float(opts.clip_gain))
        gain = np.clip(float(opts.target_std) / (local_std + 1e-3), 1.0, ceiling)
        arr = local_mean + (arr - local_mean) * gain

    # 3) 锐化（unsharp mask）：增强笔画边缘
    amount = float(max(0.0, opts.sharpen))
    if amount > 0:
        blur = np.stack([_box_blur(arr[:, :, c], 3) for c in range(3)], axis=2)
        arr = arr + amount * (arr - blur)

    return np.clip(arr, 0, 255).astype(np.uint8)


def readability(pixels: Any, rect: Rect | Sequence[float] | None = None) -> dict[str, float]:
    """可读性度量：局部对比度（RMS）+ 梯度能量。

    用于验证"小字增强后可读性提升"，也便于线上排查某块为什么看不清。
    """
    bgr = _as_bgr(pixels)
    if rect is not None:
        r = _coerce_rect(rect)
        if r is not None:
            r = _clip_rect(r, bgr.shape[1], bgr.shape[0])
            crop = bgr[r.y:r.y + r.h, r.x:r.x + r.w]
            if crop.size:
                bgr = crop
    gray = _luma(bgr)
    rms = float(np.std(gray))
    gy, gx = np.gradient(gray)
    grad = float(np.mean(np.abs(gx) + np.abs(gy)))
    return {
        "rms_contrast": round(rms, 3),
        "gradient_energy": round(grad, 3),
        "score": round(0.55 * min(1.0, grad / 40.0) + 0.45 * min(1.0, rms / 64.0), 4),
    }


# ===========================================================================
# 5. 语义分块（xy-cut + 前景掩码，复用 roi 思路）
# ===========================================================================


@dataclass
class ContentBlock:
    """一个语义内容块（一段文字、一个控件、一块图表）。"""

    rect: Rect
    kind: str = "content"
    salience: float = 0.0
    text_px: float = 0.0  # 估计字高（像素）

    def to_dict(self) -> dict[str, Any]:
        return {
            "rect": [self.rect.x, self.rect.y, self.rect.w, self.rect.h],
            "kind": self.kind,
            "salience": round(float(self.salience), 4),
            "text_px": round(float(self.text_px), 2),
        }


def _foreground_mask(bgr: np.ndarray, *, tolerance: float = 14.0) -> np.ndarray:
    """背景估计 + 差分得到前景掩码（截图/文档/白底界面效果好）。

    背景取灰度直方图主峰（出现最多的亮度）；前景 = 与背景差异超过容差的像素。
    """
    gray = _luma(bgr)
    hist, edges = np.histogram(gray, bins=64, range=(0, 255))
    bg = float(edges[int(np.argmax(hist))]) + 2.0
    mask = np.abs(gray - bg) > tolerance
    return mask


def _ink_rect(bgr: np.ndarray, rect: Rect, *, margin: int = 3) -> Rect:
    """把块收紧到实际"墨迹"（前景）范围：不为块内留白付 token。

    前景 = 与块内灰度主峰（背景）差异超过容差的像素；找不到有效前景时
    退回原块，保证不会把内容裁没。
    """
    h, w = bgr.shape[:2]
    clipped = _clip_rect(rect, w, h)
    if clipped.w < 8 or clipped.h < 6:
        return rect
    crop = bgr[clipped.y:clipped.y + clipped.h, clipped.x:clipped.x + clipped.w]
    mask = _foreground_mask(crop)
    if mask.ndim == 3:
        mask = mask[:, :, 0]
    ys, xs = np.nonzero(mask)
    if xs.size < 8:
        return rect
    x0, x1 = int(xs.min()), int(xs.max())
    y0, y1 = int(ys.min()), int(ys.max())
    if (x1 - x0 + 1) < 8 or (y1 - y0 + 1) < 6:
        return rect
    ink = _rect(
        clipped.x + x0 - margin,
        clipped.y + y0 - margin,
        (x1 - x0 + 1) + 2 * margin,
        (y1 - y0 + 1) + 2 * margin,
    )
    return _intersect(ink, clipped)


def _runs(flags: np.ndarray, min_gap: int) -> list[tuple[int, int]]:
    """把一维布尔序列切成"有内容"的区间，间隔小于 min_gap 的合并。"""
    idx = np.flatnonzero(flags)
    if idx.size == 0:
        return []
    spans: list[tuple[int, int]] = []
    start = prev = int(idx[0])
    for value in idx[1:]:
        value = int(value)
        if value - prev <= max(1, min_gap):
            prev = value
            continue
        spans.append((start, prev + 1))
        start = prev = value
    spans.append((start, prev + 1))
    return spans


def _coalesce(spans: list[tuple[int, int]], min_size: int) -> list[tuple[int, int]]:
    """把过短的相邻区间并成一段（文本行 → 段落），避免内容被切成细条。"""
    if not spans:
        return spans
    merged: list[tuple[int, int]] = []
    start, end = spans[0]
    for next_start, next_end in spans[1:]:
        if end - start < min_size:  # 当前段太短：继续并入下一段
            end = next_end
            continue
        merged.append((start, end))
        start, end = next_start, next_end
    merged.append((start, end))
    return merged


def _xy_cut(
    mask: np.ndarray,
    rect: Rect,
    out: list[Rect],
    *,
    depth: int = 0,
    max_depth: int = 10,
    min_gap: int = 6,
    min_size: int = 14,
) -> None:
    """递归二分（水平/垂直交替），把画面切成互不重叠的内容块。

    单区间也会用来**收紧边界**（裁掉整条空白），只有两个方向都无处可裁
    时才把当前矩形作为叶子块收下。
    """
    if rect.w < min_size or rect.h < min_size or depth >= max_depth:
        out.append(rect)
        return
    sub = mask[rect.y:rect.y + rect.h, rect.x:rect.x + rect.w]
    if sub.size == 0 or not sub.any():
        return

    horizontal = depth % 2 == 0
    spans: list[tuple[int, int]] = []
    chosen = False
    for attempt in (0, 1):
        flags = sub.any(axis=1) if horizontal else sub.any(axis=0)
        spans = _coalesce(_runs(flags, min_gap), min_size)
        extent = rect.h if horizontal else rect.w
        if spans and (
            len(spans) > 1 or spans[0][0] > 0 or spans[-1][1] < extent
        ):
            chosen = True
            break
        horizontal = not horizontal
        if attempt == 1:
            break
    if not chosen:
        out.append(rect)
        return

    if len(spans) == 1:
        # 只有一个内容带：裁到带上再继续切（去掉四周空白）
        start, end = spans[0]
        if horizontal:
            child = _rect(rect.x, rect.y + start, rect.w, end - start)
        else:
            child = _rect(rect.x + start, rect.y, end - start, rect.h)
        if child == rect:
            out.append(rect)
            return
        _xy_cut(mask, child, out, depth=depth + 1, max_depth=max_depth,
                min_gap=min_gap, min_size=min_size)
        return

    for start, end in spans:
        if horizontal:
            child = _rect(rect.x, rect.y + start, rect.w, end - start)
        else:
            child = _rect(rect.x + start, rect.y, end - start, rect.h)
        _xy_cut(mask, child, out, depth=depth + 1, max_depth=max_depth,
                min_gap=min_gap, min_size=min_size)


def _estimate_text_height(crop_gray: np.ndarray) -> float:
    """估计字高：找"每行都有笔画边缘"的连续行段，取中位段长。

    旧实现用"按整行平均 |dx|"，长文本段会被几百个几乎不变的列摊薄，
    结果恒为 0；改成先数"强边缘个数"，再按行聚段。
    """
    if crop_gray.size == 0:
        return 0.0
    gray = crop_gray.astype(np.float32)
    if gray.shape[1] < 2:
        return 0.0
    edges = np.abs(np.diff(gray, axis=1))
    strong = edges > 24.0
    # 去掉竖直长边（对话框/表格/分栏边框）：它们在每一行都出现，会把所有行粘成一段
    keep = strong.mean(axis=0) < 0.5
    if keep.any():
        strong = strong & keep[None, :]
    row_has_edge = strong.sum(axis=1) >= 2
    spans = _runs(row_has_edge, 1)
    heights = [e - s for s, e in spans
               if 3 <= (e - s) <= max(3, int(crop_gray.shape[0] * 0.8))]
    if not heights:
        return 0.0
    return float(np.median(heights))


def _same_row(a: Rect, b: Rect, gap: int) -> bool:
    """两块是否属于同一行：纵向大面积重叠且横向间隙不大。"""
    overlap_y = min(a.bottom, b.bottom) - max(a.y, b.y)
    if overlap_y / float(max(1, min(a.h, b.h))) < 0.6:
        return False
    gap_x = max(a.x, b.x) - min(a.right, b.right)
    return gap_x <= max(2, int(gap))


def detect_blocks(
    pixels: Any,
    *,
    min_area: int = 96,
    pad: int = 6,
    max_blocks: int = 48,
    merge_gap: int = 18,
    split_gap: int = 6,
) -> list[ContentBlock]:
    """把画面切成内容块（按阅读顺序返回）。

    ``split_gap`` 控制"切"（小间隙也切开，保住文本行级细节），
    ``merge_gap`` 控制"并"（只把同一行被切碎的碎片合回去）。
    两者分开，才能在稀疏编码里既保住行级细节、又不把整页糊成一块。

    前景占比过高（照片/复杂画面）时返回空列表，调用方据此回退到"整帧"策略。
    """
    bgr = _as_bgr(pixels)
    h, w = bgr.shape[:2]
    if h < 8 or w < 8:
        return []
    mask = _foreground_mask(bgr)
    ratio = float(mask.mean())
    if ratio > 0.72 or ratio < 1e-4:
        return []  # 满画面 / 纯色：不裁

    rects: list[Rect] = []
    _xy_cut(mask, _rect(0, 0, w, h), rects, min_gap=max(2, int(split_gap)))
    gray = _luma(bgr)

    blocks: list[ContentBlock] = []
    for r in rects:
        r = _clip_rect(_rect(r.x - pad, r.y - pad, r.w + 2 * pad, r.h + 2 * pad), w, h)
        if _area(r) < min_area:
            continue
        if _area(_intersect(r, _rect(0, 0, w, h))) <= 0:
            continue
        crop = gray[r.y:r.y + r.h, r.x:r.x + r.w]
        text_px = _estimate_text_height(crop)
        kind = "text" if text_px >= 4 else "graphic"
        blocks.append(ContentBlock(rect=r, kind=kind, text_px=text_px))

    if not blocks:
        return []
    # 合并同一行被切碎的碎片：两个方向都要有实质重叠，纵向不能"叠着并"，
    # 否则整页文本会顺着 merge_gap 一级级并成一块。
    blocks.sort(key=lambda b: (b.rect.y, b.rect.x))
    merged: list[ContentBlock] = []
    for b in blocks:
        target = None
        for candidate in reversed(merged[-4:]):
            if _same_row(candidate.rect, b.rect, merge_gap):
                target = candidate
                break
        if target is not None:
            x1 = min(target.rect.x, b.rect.x)
            y1 = min(target.rect.y, b.rect.y)
            x2 = max(target.rect.x + target.rect.w, b.rect.x + b.rect.w)
            y2 = max(target.rect.y + target.rect.h, b.rect.y + b.rect.h)
            block = ContentBlock(
                rect=_rect(x1, y1, x2 - x1, y2 - y1),
                kind="text" if target.kind == "text" and b.kind == "text" else "content",
                text_px=max(target.text_px, b.text_px),
            )
            merged[merged.index(target)] = block
            continue
        merged.append(b)

    for b in merged:
        b.salience = _salience(bgr, b.rect)
    merged.sort(key=lambda blk: (-blk.salience, blk.rect.y, blk.rect.x))
    return merged[:max_blocks]


def _salience(bgr: np.ndarray, rect: Rect, dirty_rects: Sequence[Rect] = ()) -> float:
    """关注度：边缘能量 + 局部对比 + 文字密度；落在脏矩形内加权。"""
    r = _clip_rect(rect, bgr.shape[1], bgr.shape[0])
    crop = bgr[r.y:r.y + r.h, r.x:r.x + r.w]
    if crop.size == 0:
        return 0.0
    gray = _luma(crop)
    gy, gx = np.gradient(gray)
    edge = float(np.mean(np.abs(gx) + np.abs(gy)))
    contrast = float(np.std(gray))
    score = 0.6 * min(1.0, edge / 26.0) + 0.4 * min(1.0, contrast / 70.0)
    if dirty_rects:
        dirty_area = sum(_area(_intersect(r, _clip_rect(d, bgr.shape[1], bgr.shape[0])))
                         for d in dirty_rects)
        if dirty_area > 0:
            score = min(1.0, score + 0.35 * min(1.0, dirty_area / max(1, _area(r))))
    return round(score, 4)


# ===========================================================================
# 6. 视觉转换器：增强像素 + 稀疏编码 + token 预算
# ===========================================================================


@dataclass
class VisionOptions:
    """视觉转化参数（可由 ``config/retina.yaml`` 的 ``optic_nerve`` 段覆盖）。"""

    # 预算
    max_tokens: int = 900
    target_ratio: float = 0.62  # 目标 token 比（面积按 ratio² 折算）
    mode: str = "auto"  # auto | sparse | full | dual
    # 增强
    enhance: bool = True
    sharpen: float = 0.62
    min_text_px: int = 16
    max_scale_up: float = 2.0
    # 稀疏编码
    sparse: bool = True
    gap: int = 6
    max_blocks: int = 48
    min_block_area: int = 96
    keep_floor: float = 0.2
    dynamic_keep: float = 0.85
    dirty_bonus: float = 0.35
    # 输出
    jpeg_quality: int = 82
    png: bool = False
    min_output_width: int = 360
    max_output_width: int = 1280
    # 双路
    low_width: int = 640
    low_quality: int = 72
    focus_max_width: int = 960
    focus_quality: int = 88
    # 基线（"不改造时"大脑会收到的编码宽度）
    baseline_width: int = 1280

    @classmethod
    def from_mapping(cls, data: Any) -> "VisionOptions":
        """从 dict/Config 段构造；未知键忽略，非法值回退默认。"""
        opts = cls()
        if not data:
            return opts
        if not isinstance(data, dict):
            try:
                data = dict(data)  # type: ignore[arg-type]
            except Exception:
                return opts
        for key, value in data.items():
            if not hasattr(opts, key) or value is None:
                continue
            default = getattr(opts, key)
            try:
                if isinstance(default, bool):
                    setattr(opts, key, bool(value))
                elif isinstance(default, int):
                    setattr(opts, key, int(value))
                elif isinstance(default, float):
                    setattr(opts, key, float(value))
                elif isinstance(default, str):
                    setattr(opts, key, str(value))
            except (TypeError, ValueError):
                continue
        return opts


#: 场景档位对视觉转化的影响（与 adaptive 的场景名对齐）
_SCENE_POLICIES: dict[str, dict[str, Any]] = {
    "static": {"target_ratio": 0.48, "max_tokens": 680, "mode": "auto"},
    "normal": {},
    "dynamic": {"target_ratio": 0.95, "max_tokens": 1100, "mode": "full"},
}


def _scene_name(scene: Any) -> str:
    """从 str/dict/SceneProfile 中取出场景名。"""
    if scene is None:
        return "normal"
    if isinstance(scene, str):
        name = scene
    elif isinstance(scene, dict):
        name = str(scene.get("name") or scene.get("scene") or "normal")
    else:
        name = str(getattr(scene, "name", "") or "normal")
    return name if name in _SCENE_POLICIES else "normal"


def _apply_scene(opts: VisionOptions, scene: Any) -> VisionOptions:
    import dataclasses

    name = _scene_name(scene)
    overrides = dict(_SCENE_POLICIES.get(name, {}))
    if isinstance(scene, dict):
        for key in ("max_tokens", "target_ratio", "mode", "low_width", "max_output_width"):
            if scene.get(key) is not None:
                overrides[key] = scene[key]
    elif scene is not None and not isinstance(scene, str):
        # adaptive.SceneProfile: quality / max_width / budget_tokens
        budget = getattr(scene, "budget_tokens", None)
        if budget:
            overrides["max_tokens"] = max(160, int(budget))
    return dataclasses.replace(opts, **overrides) if overrides else opts


def _token_area(tokens: int) -> float:
    """把 token 预算折算成像素面积上限（estimate_image_tokens 的逆函数）。"""
    tokens = max(_IMAGE_TOKEN_MIN, int(tokens))
    return (tokens / _IMAGE_TOKEN_COEF) ** 2 * 1_000_000.0


class VisionConverter(DocumentConverter):
    """视觉画面帧转换器：**输出增强像素，不把画面转文本**。

    三条策略（引擎按内容与场景自动选择）：

    * ``sparse``：语义分块（xy-cut）→ 按关注度排序 → 高信息块重排成马赛克，
      小字块单独放大；空白与冗余被丢弃，坐标经 ``CoordinateMapper`` 还原。
    * ``dual``：全帧低清（视杆）+ 焦点高清（视锥），与既有双路喂图一致。
    * ``full``：整帧缩放（动态场景/照片/满画面），只做增强不裁内容。

    结果 ``kind=IMAGE``，``markdown`` 恒为空。
    """

    name = "vision"

    _IMAGE_EXT = {
        ".png", ".jpg", ".jpeg", ".webp", ".bmp", ".tif", ".tiff", ".gif", ".jfif",
    }

    def __init__(self, options: VisionOptions | None = None) -> None:
        self.options = options or VisionOptions()

    # -- 流输入（图片文件）---------------------------------------------------
    def accepts(self, file_stream: BinaryIO, stream_info: StreamInfo, **kwargs: Any) -> bool:
        if kwargs.get("as_document"):
            return False  # 明确要求"当文档看"时交给 ImageDocumentConverter
        mimetype = (stream_info.mimetype or "").lower()
        ext = (stream_info.extension or "").lower()
        if mimetype.startswith("image/"):
            return True
        return ext in self._IMAGE_EXT

    def convert(
        self, file_stream: BinaryIO, stream_info: StreamInfo, **kwargs: Any
    ) -> DocumentConverterResult:
        data = file_stream.read()
        if not data:
            raise UnsupportedContentError("图片内容为空")
        return self._convert_bytes(
            data,
            name=stream_info.filename or stream_info.local_path or "image",
            **kwargs,
        )

    def _convert_bytes(self, data: bytes, *, name: str = "image", **kwargs: Any) -> DocumentConverterResult:
        pixels = decode_image(data)
        return self.convert_pixels(pixels, name=name, raw=data, **kwargs)

    # -- 对象输入（视网膜帧 / numpy / PIL）-----------------------------------
    def accepts_object(self, obj: Any, stream_info: StreamInfo, **kwargs: Any) -> bool:
        if isinstance(obj, np.ndarray):
            return obj.ndim in (2, 3)
        if hasattr(obj, "pixels"):
            return True
        return hasattr(obj, "convert") and hasattr(obj, "size")  # PIL.Image

    def convert_object(
        self, obj: Any, stream_info: StreamInfo, **kwargs: Any
    ) -> DocumentConverterResult:
        frame_meta: dict[str, Any] = {}
        pixels = obj
        if hasattr(obj, "pixels"):
            pixels = obj.pixels
            for attr in ("frame_id", "timestamp", "source", "scene", "change_ratio"):
                value = getattr(obj, attr, None)
                if value is not None:
                    frame_meta[attr] = value
        extra_meta = kwargs.pop("frame_meta", None)
        if extra_meta:
            frame_meta.update(dict(extra_meta))
        return self.convert_pixels(pixels, frame_meta=frame_meta, **kwargs)

    # -- 主流程 --------------------------------------------------------------
    def convert_pixels(
        self,
        pixels: Any,
        *,
        name: str = "frame",
        frame_meta: dict[str, Any] | None = None,
        scene: Any = None,
        dirty_rects: Sequence[Any] = (),
        mode: str | None = None,
        hint: str = "",
        enhanced: bool | None = None,
        **kwargs: Any,
    ) -> DocumentConverterResult:
        """把像素转成"对大脑最优"的视觉载荷。"""
        bgr = _as_bgr(pixels)
        height, width = bgr.shape[:2]
        if width < 8 or height < 8:
            raise UnsupportedContentError(f"帧尺寸过小：{width}x{height}")

        opts = _apply_scene(self.options, scene)
        scene_name = _scene_name(scene)
        dirty = [r for r in (_coerce_rect(d) for d in dirty_rects) if r is not None]

        # 基线：不改造时大脑收到的编码（按 encode.max_width 缩放）
        base_w = min(width, max(64, int(opts.baseline_width)))
        base_h = max(1, int(round(height * base_w / float(width))))
        baseline_tokens = estimate_image_tokens(base_w, base_h)

        chosen_mode = (mode or opts.mode or "auto").lower()
        if chosen_mode not in ("auto", "sparse", "full", "dual"):
            chosen_mode = "auto"
        if chosen_mode == "auto":
            chosen_mode = "full" if scene_name == "dynamic" else "sparse"
        if scene_name == "dynamic" and chosen_mode == "sparse":
            chosen_mode = "full"  # 动态场景不降级：整帧保真
        if not opts.sparse:
            chosen_mode = "full" if chosen_mode != "dual" else chosen_mode
        if enhanced is False:
            opts.enhance = False

        result_meta: dict[str, Any] = {
            "scene": scene_name,
            "mode": chosen_mode,
            "original_size": [width, height],
            "baseline_tokens": baseline_tokens,
            "frame": dict(frame_meta or {}),
            "hint": hint,
        }

        frames: list[EncodedFrame]
        if chosen_mode == "dual":
            frames, strategy, extra = self._dual(bgr, opts, dirty)
        elif chosen_mode == "sparse":
            frames, strategy, extra = self._sparse(bgr, opts, dirty)
            if not frames:  # 无法分块（照片/满画面）→ 整帧
                frames, strategy, extra = self._full(bgr, opts)
                chosen_mode = "full"
                result_meta["mode"] = "full"
                result_meta["mode_fallback"] = "no_blocks"
        else:
            frames, strategy, extra = self._full(bgr, opts)

        mapper = extra.get("mapper")
        result_meta.update({k: v for k, v in extra.items() if k != "mapper"})
        result_meta["mapper"] = mapper
        # 可读性：在同一几何上比较"增强前/后"，避免把裁剪缩放混进指标
        sample_pixels = bgr
        if max(width, height) > 1280:
            sample_scale = 1280.0 / max(width, height)
            sample_pixels = _resize(
                bgr, max(8, int(width * sample_scale)), max(8, int(height * sample_scale))
            )
        result_meta["readability"] = readability(sample_pixels)
        if opts.enhance:
            enhanced_sample = enhance_pixels(
                sample_pixels,
                EnhanceOptions(
                    enabled=True,
                    sharpen=opts.sharpen,
                    min_text_px=opts.min_text_px,
                    max_scale_up=opts.max_scale_up,
                ),
            )
            result_meta["readability_enhanced"] = readability(enhanced_sample)

        output_tokens = sum(f.tokens for f in frames)
        report = TokenReport(
            original=baseline_tokens,
            output=output_tokens,
            strategy=strategy,
            detail={
                "native": [width, height],
                "encoded": [[f.width, f.height] for f in frames],
                "blocks": extra.get("blocks_kept", 0),
                "coverage": extra.get("keep_ratio", 1.0),
                "scene": scene_name,
            },
        )

        result = DocumentConverterResult(
            kind=ResultKind.IMAGE,
            markdown="",  # 画面绝不转文本
            title=name,
            image=frames[0],
            images=list(frames[1:]),
            meta=result_meta,
            converter=self.name,
            tokens=report,
        )
        return result

    # -- 策略：整帧 ----------------------------------------------------------
    def _full(
        self, bgr: np.ndarray, opts: VisionOptions
    ) -> tuple[list[EncodedFrame], str, dict[str, Any]]:
        h, w = bgr.shape[:2]
        limit_area = _token_area(opts.max_tokens)
        max_w = min(int(opts.max_output_width) or w, w)
        scale = min(1.0, max_w / float(w))
        if w * h * scale * scale > limit_area:
            scale = min(scale, math.sqrt(limit_area / float(w * h)))
        scale = max(0.2, scale)
        out_w = max(8, int(round(w * scale)))
        out_h = max(8, int(round(h * scale)))
        pixels = _resize(bgr, out_w, out_h)
        if opts.enhance:
            pixels = enhance_pixels(
                pixels,
                EnhanceOptions(enabled=True, sharpen=opts.sharpen,
                               min_text_px=opts.min_text_px, max_scale_up=opts.max_scale_up),
            )
        data = encode_image(pixels, quality=opts.jpeg_quality,
                            mimetype="image/png" if opts.png else "image/jpeg")
        frame = EncodedFrame(
            data=data,
            width=out_w,
            height=out_h,
            mimetype="image/png" if opts.png else "image/jpeg",
            rect=_rect(0, 0, w, h),
            scale=out_w / float(w),
            strategy="full",
            meta={"native_size": [w, h]},
        )
        mapper = CoordinateMapper(
            mosaic_width=out_w,
            mosaic_height=out_h,
            native_width=w,
            native_height=h,
            placements=[
                BlockPlacement(index=0, native=_rect(0, 0, w, h),
                               mosaic=_rect(0, 0, out_w, out_h),
                               scale=out_w / float(w), salience=1.0)
            ],
            strategy="full",
        )
        return [frame], "full", {
            "mapper": mapper,
            "blocks_detected": 0,
            "blocks_kept": 1,
            "keep_ratio": 1.0,
            "enhanced": bool(opts.enhance),
        }

    # -- 策略：稀疏编码 ------------------------------------------------------
    def _sparse(
        self, bgr: np.ndarray, opts: VisionOptions, dirty: Sequence[Rect]
    ) -> tuple[list[EncodedFrame], str, dict[str, Any]]:
        h, w = bgr.shape[:2]
        blocks = detect_blocks(
            bgr,
            min_area=opts.min_block_area,
            pad=max(4, opts.gap),
            max_blocks=opts.max_blocks,
        )
        if not blocks:
            return [], "sparse", {}
        # 只有一个块且几乎盖满整帧 → 稀疏编码无从下手，交给 _full 按预算缩放
        if len(blocks) == 1:
            covered = _area(_intersect(blocks[0].rect, _rect(0, 0, w, h)))
            if covered >= 0.9 * float(w * h):
                return [], "sparse", {}

        # 关注度：脏矩形内的块优先（增量帧只传变化区域）
        if dirty:
            for block in blocks:
                block.salience = _salience(bgr, block.rect, dirty)
            blocks.sort(key=lambda b: (-b.salience, b.rect.y, b.rect.x))

        # 只为墨迹付费：把块内的留白收掉，再按内容面积给预算。
        # 拼图允许 25% 的间距/超分冗余；文字极密时内容≈整帧，自然不降级。
        trimmed: list[ContentBlock] = []
        for block in blocks:
            ink = _ink_rect(bgr, block.rect, margin=max(2, int(opts.gap) // 2))
            if _area(ink) < max(16, int(opts.min_block_area) // 4):
                continue
            short_side = min(ink.w, ink.h)
            long_side = max(ink.w, ink.h)
            # 分隔线/边条：几乎没有信息量，却能把整张画布撑宽（1280×9 的横条
            # 会让每块都被迫缩到 991 宽），直接丢掉。
            if (block.kind != "text" and float(block.text_px) <= 0.0
                    and short_side < 12 and long_side >= 6 * max(1, short_side)):
                continue
            block.rect = ink
            trimmed.append(block)
        if trimmed:
            blocks = trimmed

        keep_min = max(1, int(math.ceil(len(blocks) * float(opts.keep_floor))))
        area_cap = _token_area(opts.max_tokens)
        target_area = w * h * max(0.02, float(opts.target_ratio) ** 2)
        soft_area = min(area_cap, target_area)
        # 硬预算 = 基线（整帧按 baseline_width 缩到大脑吃的尺寸）面积：
        # 任何情况下都不该比直接发一整帧更贵，否则"转化"纯亏。
        base_scale = min(1.0, float(opts.baseline_width) / float(w)) if w else 1.0
        baseline_area = (w * base_scale) * (h * base_scale)
        hard_area = min(area_cap, baseline_area)
        content_area = sum(_area(b.rect) for b in blocks) or 1
        budget_area = min(hard_area, max(soft_area, content_area * 1.25))

        selected = list(blocks)
        canvas, placements = self._pack(bgr, selected, opts)
        area = canvas.shape[0] * canvas.shape[1]
        # 超预算时按"信息价值/成本"逐级让步：先收超分倍数，再整体降采样，
        # 只有比整帧基线还贵才丢块——任何一步都不轻易删掉用户正在看的文字。
        if area > budget_area:
            plain, plain_placements = self._pack(bgr, selected, opts, max_upscale=1.0)
            plain_area = plain.shape[0] * plain.shape[1]
            if plain_area <= budget_area:
                allow = math.sqrt(budget_area / float(max(1, plain_area)))
                canvas, placements = self._pack(
                    bgr, selected, opts, max_upscale=max(1.0, allow)
                )
            else:
                max_text = max((float(b.text_px) for b in selected), default=0.0)
                floor = max(0.55, min(1.0, 12.0 / max_text)) if max_text > 0 else 0.55
                factor = max(floor, math.sqrt(budget_area / float(max(1, plain_area))))
                if factor < 0.99:
                    canvas, placements = self._pack(
                        bgr, selected, opts, max_upscale=1.0, factor=factor
                    )
                else:
                    canvas, placements = plain, plain_placements
            area = canvas.shape[0] * canvas.shape[1]

        while area > hard_area and len(selected) > keep_min:
            selected.pop()  # 已按显著度降序，丢最后一个（最不重要）
            canvas, placements = self._pack(bgr, selected, opts)
            area = canvas.shape[0] * canvas.shape[1]

        out_w, out_h = canvas.shape[1], canvas.shape[0]
        if out_w < opts.min_output_width:
            grow = min(float(opts.max_scale_up) + 0.5,
                       opts.min_output_width / float(max(1, out_w)))
            # 放大只为"太小看不清"，不能反过来把预算撑破。
            if grow > 1.01 and (out_w * grow) * (out_h * grow) <= hard_area * 1.02:
                canvas = _resize(canvas, int(round(out_w * grow)), int(round(out_h * grow)))
                factor = canvas.shape[1] / float(out_w)
                for p in placements:
                    p.mosaic = _rect(p.mosaic.x * factor, p.mosaic.y * factor,
                                     p.mosaic.w * factor, p.mosaic.h * factor)
                    p.scale *= factor
                out_w, out_h = canvas.shape[1], canvas.shape[0]

        data = encode_image(canvas, quality=opts.jpeg_quality,
                            mimetype="image/png" if opts.png else "image/jpeg")
        frame = EncodedFrame(
            data=data,
            width=out_w,
            height=out_h,
            mimetype="image/png" if opts.png else "image/jpeg",
            rect=placements[0].native if placements else None,
            scale=placements[0].scale if placements else 1.0,
            strategy="sparse",
            meta={"blocks": [p.to_dict() for p in placements]},
        )
        covered = sum(_area(_intersect(p.native, _rect(0, 0, w, h))) for p in placements)
        coverage = round(min(1.0, covered / float(w * h)), 4)
        mapper = CoordinateMapper(
            mosaic_width=out_w,
            mosaic_height=out_h,
            native_width=w,
            native_height=h,
            placements=placements,
            strategy="sparse",
        )
        return [frame], "sparse", {
            "mapper": mapper,
            "blocks_detected": len(blocks),
            "blocks_kept": len(placements),
            "keep_ratio": coverage,
            "enhanced": bool(opts.enhance),
            "layout": {"canvas": [out_w, out_h], "gap": opts.gap},
        }

    def _pack(
        self,
        bgr: np.ndarray,
        blocks: Sequence[ContentBlock],
        opts: VisionOptions,
        *,
        factor: float = 1.0,
        max_upscale: float | None = None,
    ) -> tuple[np.ndarray, list[BlockPlacement]]:
        """把内容块按关注度顺序拼成马赛克（shelf packing）。

        ``max_upscale`` 给"小字超分"设上限：预算紧时先收放大倍数，保住内容。
        """
        h, w = bgr.shape[:2]
        gap = max(0, int(opts.gap))
        background = _background_color(bgr)
        rendered: list[tuple[ContentBlock, np.ndarray, int, int]] = []
        for block in blocks:
            r = _clip_rect(block.rect, w, h)
            if r.w <= 0 or r.h <= 0:
                continue
            crop = bgr[r.y:r.y + r.h, r.x:r.x + r.w]
            up = _text_upscale(block, opts)
            if max_upscale is not None:
                up = min(up, float(max_upscale))
            scale = max(1.0, up) * factor
            if opts.enhance:
                crop = enhance_pixels(
                    crop,
                    EnhanceOptions(enabled=True, sharpen=opts.sharpen,
                                   min_text_px=opts.min_text_px,
                                   max_scale_up=opts.max_scale_up),
                )
            tw = max(6, int(round(r.w * scale)))
            th = max(6, int(round(r.h * scale)))
            rendered.append((block, _resize(crop, tw, th), tw, th))
        if not rendered:
            return np.full((8, 8, 3), 255, np.uint8), []

        total_area = sum(tw * th for _, _, tw, th in rendered)
        # shelf 拼图按高度降序摆放：行高由最高块决定，先放高块能少浪费空白。
        rendered.sort(key=lambda item: (-item[3], -item[2]))
        target_w = int(np.clip(math.sqrt(max(1.0, total_area) * 1.7),
                               opts.min_output_width, opts.max_output_width))
        rows: list[list[tuple[ContentBlock, np.ndarray, int, int]]] = []
        current: list[tuple[ContentBlock, np.ndarray, int, int]] = []
        current_w = 0
        for item in rendered:
            item_w = item[2]
            extra = item_w + (gap if current else 0)
            if current and current_w + extra > target_w:
                rows.append(current)
                current = []
                current_w = 0
                extra = item_w
            current.append(item)
            current_w += extra
        if current:
            rows.append(current)

        canvas_w = max(
            (sum(i[2] for i in row) + gap * (len(row) - 1)) for row in rows
        )
        canvas_h = sum(max(i[3] for i in row) for row in rows) + gap * (len(rows) - 1)
        canvas = np.empty((max(8, canvas_h), max(8, canvas_w), 3), np.uint8)
        canvas[:, :] = background

        placements: list[BlockPlacement] = []
        y = 0
        for row in rows:
            x = 0
            for index, (block, crop, tw, th) in enumerate(row):
                canvas[y:y + th, x:x + tw] = crop
                placements.append(
                    BlockPlacement(
                        index=len(placements),
                        native=_rect(block.rect.x, block.rect.y, block.rect.w, block.rect.h),
                        mosaic=_rect(x, y, tw, th),
                        scale=tw / float(max(1, block.rect.w)),
                        salience=block.salience,
                    )
                )
                x += tw + gap
            y += max(i[3] for i in row) + gap
        return canvas, placements

    # -- 策略：双路（低清全帧 + 高清焦点）------------------------------------
    def _dual(
        self, bgr: np.ndarray, opts: VisionOptions, dirty: Sequence[Rect]
    ) -> tuple[list[EncodedFrame], str, dict[str, Any]]:
        h, w = bgr.shape[:2]
        budget = _token_area(opts.max_tokens)

        # 焦点：脏矩形并集 > 最显著的内容块簇
        focus: Rect | None = None
        if dirty:
            x1 = min(r.x for r in dirty)
            y1 = min(r.y for r in dirty)
            x2 = max(r.x + r.w for r in dirty)
            y2 = max(r.y + r.h for r in dirty)
            focus = _clip_rect(_rect(x1, y1, x2 - x1, y2 - y1), w, h)
        blocks = detect_blocks(bgr, min_area=opts.min_block_area, pad=8,
                               max_blocks=opts.max_blocks, merge_gap=12)
        if focus is None or _area(focus) < 0.02 * w * h:
            if blocks:
                x1 = min(b.rect.x for b in blocks[:3])
                y1 = min(b.rect.y for b in blocks[:3])
                x2 = max(b.rect.x + b.rect.w for b in blocks[:3])
                y2 = max(b.rect.y + b.rect.h for b in blocks[:3])
                focus = _clip_rect(_rect(x1, y1, x2 - x1, y2 - y1), w, h)
        if focus is None or _area(focus) <= 0:
            return self._full(bgr, opts)

        # 低清全帧
        low_scale = min(1.0, float(opts.low_width) / float(w))
        low_w = max(64, int(round(w * low_scale)))
        low_h = max(36, int(round(h * low_scale)))
        low_pixels = _resize(bgr, low_w, low_h)
        low_data = encode_image(low_pixels, quality=opts.low_quality)

        # 焦点高清（必要时放大）
        crop = bgr[focus.y:focus.y + focus.h, focus.x:focus.x + focus.w]
        upscale = max(1.0, min(opts.max_scale_up,
                               max(700.0 / max(1, focus.w), 1.0)))
        focus_w = max(64, min(opts.focus_max_width, int(round(focus.w * upscale))))
        focus_h = max(36, int(round(focus.h * (focus_w / float(max(1, focus.w))))))
        focus_pixels = enhance_pixels(crop) if opts.enhance else crop
        focus_pixels = _resize(focus_pixels, focus_w, focus_h)

        # 预算：两张图合起来不超过 max_tokens，也不超过"直接发整帧"的基线
        total = estimate_image_tokens(low_w, low_h) + estimate_image_tokens(focus_w, focus_h)
        base_scale = min(1.0, float(opts.baseline_width) / float(w)) if w else 1.0
        baseline_area = (w * base_scale) * (h * base_scale)
        limit = min(_token_area(opts.max_tokens), baseline_area)
        if (low_w * low_h + focus_w * focus_h) > limit:
            ratio = math.sqrt(limit / float(low_w * low_h + focus_w * focus_h))
            low_w = max(64, int(low_w * ratio))
            low_h = max(36, int(low_h * ratio))
            focus_w = max(64, int(focus_w * ratio))
            focus_h = max(36, int(focus_h * ratio))
            low_pixels = _resize(bgr, low_w, low_h)
            low_data = encode_image(low_pixels, quality=opts.low_quality)
            focus_pixels = _resize(focus_pixels, focus_w, focus_h)
            total = estimate_image_tokens(low_w, low_h) + estimate_image_tokens(focus_w, focus_h)

        focus_data = encode_image(focus_pixels, quality=opts.focus_quality)
        low_frame = EncodedFrame(
            data=low_data, width=low_w, height=low_h, rect=_rect(0, 0, w, h),
            scale=low_w / float(w), strategy="dual-low", meta={"role": "context"},
        )
        focus_frame = EncodedFrame(
            data=focus_data, width=focus_w, height=focus_h, rect=focus,
            scale=focus_w / float(max(1, focus.w)), strategy="dual-focus",
            meta={"role": "focus", "upscale": round(focus_w / float(max(1, focus.w)), 3)},
        )
        mapper = CoordinateMapper(
            mosaic_width=focus_w, mosaic_height=focus_h,
            native_width=w, native_height=h,
            placements=[BlockPlacement(index=0, native=focus,
                                       mosaic=_rect(0, 0, focus_w, focus_h),
                                       scale=focus_w / float(max(1, focus.w)), salience=1.0)],
            strategy="dual",
        )
        return [low_frame, focus_frame], "dual", {
            "mapper": mapper,
            "blocks_detected": len(blocks),
            "blocks_kept": len(blocks),
            "keep_ratio": 1.0,  # 低清全帧覆盖整个画面（焦点是额外细节，不算覆盖率）
            "enhanced": bool(opts.enhance),
            "dual": {
                "low_size": [low_w, low_h],
                "focus_size": [focus_w, focus_h],
                "focus_rect": [focus.x, focus.y, focus.w, focus.h],
                "focus_coverage": round(_area(focus) / float(w * h), 4),
            },
        }

    # -- 指令式变焦-----------------------------------------------------------
    def convert_zoom(
        self,
        pixels: Any,
        rect: Sequence[float] | Rect,
        *,
        min_width: int = 720,
        max_tokens: int | None = None,
        hint: str = "",
    ) -> DocumentConverterResult:
        """对原始帧的指定区域做"增强 + 放大"，用于模型发起的变焦。"""
        bgr = _as_bgr(pixels)
        h, w = bgr.shape[:2]
        r = _coerce_rect(rect)
        if r is None:
            raise UnsupportedContentError("变焦矩形无效")
        r = _clip_rect(r, w, h)
        if r.w <= 0 or r.h <= 0:
            raise UnsupportedContentError("变焦矩形超出画面")
        opts = self.options
        crop = bgr[r.y:r.y + r.h, r.x:r.x + r.w]
        scale = max(1.0, min(opts.max_scale_up, float(min_width) / max(1, r.w)))
        out_w = min(int(opts.focus_max_width), max(64, int(round(r.w * scale))))
        out_h = max(36, int(round(r.h * (out_w / float(max(1, r.w))))))
        pixels_out = enhance_pixels(crop) if opts.enhance else crop
        pixels_out = _resize(pixels_out, out_w, out_h)
        data = encode_image(pixels_out, quality=opts.focus_quality)
        frame = EncodedFrame(
            data=data, width=out_w, height=out_h, rect=r,
            scale=out_w / float(max(1, r.w)), strategy="zoom",
            meta={"role": "zoom"},
        )
        mapper = CoordinateMapper(
            mosaic_width=out_w, mosaic_height=out_h, native_width=w, native_height=h,
            placements=[BlockPlacement(index=0, native=r, mosaic=_rect(0, 0, out_w, out_h),
                                       scale=out_w / float(max(1, r.w)), salience=1.0)],
            strategy="zoom",
        )
        baseline = estimate_image_tokens(w, h)
        return DocumentConverterResult(
            kind=ResultKind.IMAGE,
            markdown="",
            title="zoom",
            image=frame,
            meta={"strategy": "zoom", "mapper": mapper, "hint": hint,
                  "original_size": [w, h], "zoom_rect": [r.x, r.y, r.w, r.h]},
            converter=self.name,
            tokens=TokenReport(original=baseline, output=frame.tokens, strategy="zoom"),
        )


def _text_upscale(block: ContentBlock, opts: VisionOptions) -> float:
    """小字块单独放大：让字高回到模型可读区间。"""
    if block.text_px and block.text_px < opts.min_text_px:
        return min(float(opts.max_scale_up), max(1.0, float(opts.min_text_px) / block.text_px))
    if block.kind == "text" and block.rect.h <= opts.min_text_px:
        return min(float(opts.max_scale_up), 1.4)
    return 1.0


def _background_color(bgr: np.ndarray) -> tuple[int, int, int]:
    """用画面四边像素的中位数作为拼图底色（深色主题也不会突兀）。"""
    h, w = bgr.shape[:2]
    border = np.concatenate(
        [
            bgr[: max(1, h // 20), :, :].reshape(-1, 3),
            bgr[-max(1, h // 20):, :, :].reshape(-1, 3),
            bgr[:, : max(1, w // 20), :].reshape(-1, 3),
            bgr[:, -max(1, w // 20):, :].reshape(-1, 3),
        ],
        axis=0,
    )
    med = np.median(border, axis=0).astype(np.uint8)
    return int(med[0]), int(med[1]), int(med[2])


# ===========================================================================
# 7. 异常
# ===========================================================================


class OpticNerveError(Exception):
    """视神经引擎的错误基类。"""


class UnsupportedContentError(OpticNerveError):
    """没有转换器能处理该内容。"""


class MissingDependencyException(OpticNerveError):
    """该类型需要可选依赖（给出安装提示，而不是静默降级）。"""


# ===========================================================================
# 8. 结构化转换器：表格 / CSV / JSON（行列关系不丢）
# ===========================================================================


def _decode_text(data: bytes, charset: str | None = None) -> str:
    """尽力解码文本：显式 charset → BOM → utf-8 → gbk → latin-1。"""
    if not data:
        return ""
    candidates: list[str] = []
    if charset:
        candidates.append(charset)
    if data.startswith(b"\xef\xbb\xbf"):
        candidates.append("utf-8-sig")
    if data.startswith(b"\xff\xfe") or data.startswith(b"\xfe\xff"):
        candidates.append("utf-16")
    candidates += ["utf-8", "gb18030", "latin-1"]
    for enc in candidates:
        try:
            return data.decode(enc)
        except (UnicodeDecodeError, LookupError):
            continue
    try:  # charset_normalizer 在 markitdown 依赖里，通常可用
        from charset_normalizer import from_bytes  # type: ignore

        best = from_bytes(data).best()
        if best is not None:
            return str(best)
    except Exception:
        pass
    return data.decode("utf-8", errors="replace")


class StructuredConverter(DocumentConverter):
    """表格 / CSV / TSV / JSON → 结构化表示（列、行、类型、截断标记）。

    与"转成 markdown 表"的区别：行列关系是**结构化保留**的，渲染 markdown 只是
    其中一种视图；对大脑还可给更省 token 的 ``to_compact()``（TSV）。
    """

    name = "structured"
    _EXT = {".csv", ".tsv", ".tab", ".json", ".jsonl", ".ndjson"}

    def __init__(self, *, max_rows: int = 5000, max_bytes: int = 16_000_000) -> None:
        self.max_rows = int(max_rows)
        self.max_bytes = int(max_bytes)

    def accepts(self, file_stream: BinaryIO, stream_info: StreamInfo, **kwargs: Any) -> bool:
        if kwargs.get("as_document"):
            return False
        mimetype = (stream_info.mimetype or "").lower()
        ext = (stream_info.extension or "").lower()
        if ext in self._EXT:
            return True
        return mimetype in (
            "text/csv", "text/tab-separated-values", "application/json",
            "application/x-ndjson", "text/x-json", "application/jsonl",
        )

    def accepts_object(self, obj: Any, stream_info: StreamInfo, **kwargs: Any) -> bool:
        return isinstance(obj, (dict, list, tuple))

    def convert_object(self, obj: Any, stream_info: StreamInfo, **kwargs: Any) -> DocumentConverterResult:
        if isinstance(obj, tuple):
            obj = list(obj)
        payload = self._from_json_value(obj)
        return self._result(payload, stream_info, original_tokens=None)

    def convert(
        self, file_stream: BinaryIO, stream_info: StreamInfo, **kwargs: Any
    ) -> DocumentConverterResult:
        data = file_stream.read()
        if len(data) > self.max_bytes:
            raise UnsupportedContentError(f"结构化文件过大：{len(data)} bytes")
        text = _decode_text(data, stream_info.charset)
        ext = (stream_info.extension or "").lower()
        mimetype = (stream_info.mimetype or "").lower()
        if ext in (".json", ".jsonl", ".ndjson") or "json" in mimetype:
            payload = self._from_json_text(text)
        else:
            payload = self._from_delimited(text, ext=ext)
        return self._result(payload, stream_info, original_tokens=estimate_text_tokens(text))

    # -- CSV / TSV -----------------------------------------------------------
    def _from_delimited(self, text: str, *, ext: str = "") -> StructuredPayload:
        import csv

        sample = text[:8192]
        dialect: Any = None
        try:
            dialect = csv.Sniffer().sniff(sample, delimiters=",;\t|")
        except Exception:
            dialect = None
        if dialect is None:
            class _D(csv.excel):
                delimiter = "\t" if ext in (".tsv", ".tab") else ","
            dialect = _D
        rows: list[list[str]] = []
        try:
            reader = csv.reader(io.StringIO(text), dialect)
            for row in reader:
                rows.append([c.strip() for c in row])
        except Exception as exc:
            raise UnsupportedContentError(f"CSV 解析失败：{exc}") from exc
        while rows and not any(cell.strip() for cell in rows[-1]):
            rows.pop()
        if not rows:
            return StructuredPayload(format="table", meta={"empty": True})
        header = rows[0]
        body = rows[1:]
        # 列名去空、去重
        columns: list[str] = []
        seen: dict[str, int] = {}
        for i, name in enumerate(header):
            label = name.strip() or f"col_{i + 1}"
            if label in seen:
                seen[label] += 1
                label = f"{label}_{seen[label]}"
            else:
                seen[label] = 0
            columns.append(label)
        width = len(columns)
        truncated = False
        if len(body) > self.max_rows:
            body = body[: self.max_rows]
            truncated = True
        norm_rows: list[list[Any]] = []
        for row in body:
            if not any(cell.strip() for cell in row):
                continue
            padded = list(row[:width]) + [""] * max(0, width - len(row))
            norm_rows.append([_typed(cell) for cell in padded])
        return StructuredPayload(
            format="table",
            columns=columns,
            rows=norm_rows,
            truncated=truncated,
            meta={"row_count": len(norm_rows), "col_count": width},
        )

    # -- JSON ----------------------------------------------------------------
    def _from_json_text(self, text: str) -> StructuredPayload:
        stripped = text.strip()
        if not stripped:
            return StructuredPayload(format="records")
        if stripped[0] == "{" and "\n" in stripped:
            # JSONL / NDJSON 先按行试
            lines = [ln for ln in stripped.splitlines() if ln.strip()]
            if len(lines) > 1:
                try:
                    values = [json.loads(ln) for ln in lines]
                    if all(isinstance(v, dict) for v in values):
                        return self._records_payload(values)
                except Exception:
                    pass
        try:
            value = json.loads(stripped)
        except Exception:
            # 宽松：按行解析
            try:
                values = [json.loads(ln) for ln in stripped.splitlines() if ln.strip()]
            except Exception as exc:
                raise UnsupportedContentError(f"JSON 解析失败：{exc}") from exc
            return self._from_json_value(values)
        return self._from_json_value(value)

    def _from_json_value(self, value: Any) -> StructuredPayload:
        if isinstance(value, list):
            if value and all(isinstance(v, dict) for v in value):
                return self._records_payload(value)
            if value and all(isinstance(v, (list, tuple)) for v in value):
                width = max(len(v) for v in value)
                columns = [f"col_{i + 1}" for i in range(width)]
                rows = [list(v) + [None] * (width - len(v)) for v in value]
                return StructuredPayload(format="table", columns=columns, rows=rows)
            return StructuredPayload(
                format="table", columns=["value"], rows=[[_scalar(v)] for v in value]
            )
        if isinstance(value, dict):
            if value and all(isinstance(v, (list, dict)) for v in value.values()):
                # {"items": [...]} 常见外层包裹：若只有一个列表值则展开
                lists = [v for v in value.values() if isinstance(v, list)]
                if len(lists) == 1 and all(isinstance(v, dict) for v in lists[0]):
                    payload = self._records_payload(lists[0])
                    payload.schema = {"envelope_keys": sorted(k for k in value if not isinstance(value[k], (list, dict)))}
                    return payload
            scalar_items = {k: _scalar(v) for k, v in value.items()
                            if not isinstance(v, (list, dict))}
            if len(scalar_items) == len(value) or len(scalar_items) >= max(1, len(value) // 2):
                return StructuredPayload(
                    format="key_value",
                    columns=["key", "value"],
                    rows=[[k, v] for k, v in scalar_items.items()],
                    records=[{"key": k, "value": v} for k, v in scalar_items.items()],
                    schema={"keys": list(value.keys())},
                )
            return StructuredPayload(
                format="json",
                schema={"keys": list(value.keys())},
                meta={"json": value},
            )
        return StructuredPayload(format="key_value", columns=["value"], rows=[[_scalar(value)]])

    def _records_payload(self, records: Sequence[dict[str, Any]]) -> StructuredPayload:
        truncated = False
        records = list(records)
        if len(records) > self.max_rows:
            records = records[: self.max_rows]
            truncated = True
        columns: list[str] = []
        for rec in records:
            for key in rec:
                if key not in columns:
                    columns.append(str(key))
        rows = [[_scalar(rec.get(c)) for c in columns] for rec in records]
        return StructuredPayload(
            format="records",
            columns=columns,
            rows=rows,
            records=[{str(k): _scalar(v) for k, v in rec.items()} for rec in records],
            truncated=truncated,
            meta={"row_count": len(records), "col_count": len(columns)},
        )

    # -- 结果 ----------------------------------------------------------------
    def _result(
        self,
        payload: StructuredPayload,
        stream_info: StreamInfo,
        *,
        original_tokens: int | None,
    ) -> DocumentConverterResult:
        markdown = payload.to_markdown()
        if payload.format == "json":
            markdown = "```json\n" + json.dumps(payload.meta.get("json"), ensure_ascii=False,
                                                indent=2)[:20000] + "\n```"
        compact = payload.to_compact()
        title = stream_info.filename or stream_info.local_path or "structured"
        output_tokens = estimate_text_tokens(compact) if compact else 0
        if original_tokens is None:
            original_tokens = output_tokens
        return DocumentConverterResult(
            kind=ResultKind.STRUCTURED,
            markdown=markdown,
            title=title,
            structured=payload,
            meta={
                "format": payload.format,
                "shape": list(payload.shape),
                "truncated": payload.truncated,
                "compact_tokens": output_tokens,
            },
            converter=self.name,
            tokens=TokenReport(
                original=original_tokens,
                output=output_tokens,
                strategy="structured",
                detail={"format": payload.format, "rows": payload.shape[0]},
            ),
        )


def _scalar(value: Any) -> Any:
    if isinstance(value, (dict, list)):
        try:
            return json.dumps(value, ensure_ascii=False, separators=(",", ":"))
        except Exception:
            return str(value)
    return value


def _typed(cell: str) -> Any:
    """CSV 单元格轻量类型推断（整数/浮点/空），保留原始文本不丢。"""
    text = cell.strip()
    if text == "":
        return ""
    try:
        if re.fullmatch(r"[+-]?\d{1,18}", text):
            return int(text)
    except Exception:
        pass
    try:
        if re.fullmatch(r"[+-]?\d+\.\d+([eE][+-]?\d+)?", text):
            return float(text)
    except Exception:
        pass
    return cell


# ===========================================================================
# 9. 代码转换器：语言识别 + 符号大纲 + 截断
# ===========================================================================


_LANG_BY_EXT = {
    ".py": "python", ".pyi": "python", ".pyw": "python",
    ".js": "javascript", ".mjs": "javascript", ".cjs": "javascript", ".jsx": "jsx",
    ".ts": "typescript", ".tsx": "tsx", ".mts": "typescript",
    ".java": "java", ".kt": "kotlin", ".kts": "kotlin", ".scala": "scala",
    ".c": "c", ".h": "c", ".cc": "cpp", ".cpp": "cpp", ".cxx": "cpp", ".hpp": "cpp", ".hh": "cpp",
    ".cs": "csharp", ".go": "go", ".rs": "rust", ".rb": "ruby", ".php": "php",
    ".swift": "swift", ".m": "objectivec", ".mm": "objectivec", ".dart": "dart",
    ".sh": "bash", ".bash": "bash", ".zsh": "bash", ".ps1": "powershell",
    ".sql": "sql", ".lua": "lua", ".r": "r", ".pl": "perl", ".ex": "elixir", ".exs": "elixir",
    ".css": "css", ".scss": "scss", ".less": "less", ".vue": "vue", ".svelte": "svelte",
    ".yml": "yaml", ".yaml": "yaml", ".toml": "toml", ".ini": "ini", ".cfg": "ini",
    ".dockerfile": "dockerfile", ".tf": "hcl", ".proto": "protobuf", ".asm": "asm",
}

_LANG_BY_MIME = {
    "text/x-python": "python",
    "text/x-python3": "python",
    "application/javascript": "javascript",
    "text/javascript": "javascript",
    "application/typescript": "typescript",
    "text/x-java": "java",
    "text/x-c": "c",
    "text/x-c++": "cpp",
    "text/x-csharp": "csharp",
    "text/x-rust": "rust",
    "text/x-go": "go",
    "text/x-ruby": "ruby",
    "text/x-sh": "bash",
    "text/x-shellscript": "bash",
    "text/x-sql": "sql",
    "application/x-sh": "bash",
}

#: 各语言族的符号正则：(模式, 种类标签)
_SYMBOL_PATTERNS: dict[str, list[tuple[str, str]]] = {
    "python": [
        (r"^\s*(?:async\s+)?def\s+([A-Za-z_]\w*)\s*\(([^)]*)\)", "function"),
        (r"^\s*class\s+([A-Za-z_]\w*)\s*(?:\(([^)]*)\))?", "class"),
        (r"^([A-Z][A-Z0-9_]{2,})\s*=", "constant"),
    ],
    "javascript": [
        (r"^\s*(?:export\s+)?(?:async\s+)?function\s+([A-Za-z_$][\w$]*)\s*\(([^)]*)\)", "function"),
        (r"^\s*(?:export\s+)?class\s+([A-Za-z_$][\w$]*)", "class"),
        (r"^\s*(?:export\s+)?const\s+([A-Za-z_$][\w$]*)\s*=\s*(?:async\s*)?\(([^)]*)\)\s*=>", "function"),
        (r"^\s*(?:export\s+)?(?:const|let|var)\s+([A-Za-z_$][\w$]*)\s*=", "variable"),
    ],
    "java": [
        (r"^\s*(?:public|private|protected|static|final|abstract|synchronized|\s)+(?:[\w<>\[\],\s.]+\s+)?([A-Za-z_]\w*)\s*\(([^)]*)\)\s*(?:throws [\w,\s]+)?\{", "method"),
        (r"^\s*(?:public|private|protected|static|final|abstract|\s)*(?:class|interface|enum|record)\s+([A-Za-z_]\w*)", "type"),
    ],
    "csharp": [
        (r"^\s*(?:public|private|protected|internal|static|virtual|override|async|partial|\s)+[\w<>\[\],?\s]+\s+([A-Za-z_]\w*)\s*\(([^)]*)\)", "method"),
        (r"^\s*(?:public|private|protected|internal|static|abstract|sealed|partial|\s)*(?:class|interface|struct|enum|record)\s+([A-Za-z_]\w*)", "type"),
    ],
    "go": [
        (r"^func\s+(?:\([^)]*\)\s*)?([A-Za-z_]\w*)\s*\(([^)]*)\)", "function"),
        (r"^type\s+([A-Za-z_]\w*)\s+(?:struct|interface)", "type"),
    ],
    "rust": [
        (r"^\s*(?:pub\s+)?(?:async\s+)?fn\s+([A-Za-z_]\w*)\s*\(([^)]*)\)", "function"),
        (r"^\s*(?:pub\s+)?(?:struct|enum|trait|impl|mod)\s+([A-Za-z_]\w*)", "type"),
    ],
    "cpp": [
        (r"^\s*(?:[\w:<>~*&\s]+)\s+([A-Za-z_]\w*)\s*\(([^;{}]*)\)\s*(?:const)?\s*\{", "function"),
        (r"^\s*(?:class|struct|enum|namespace)\s+([A-Za-z_]\w*)", "type"),
    ],
    "ruby": [
        (r"^\s*def\s+(?:self\.)?([A-Za-z_]\w*[?!=]?)", "method"),
        (r"^\s*(?:class|module)\s+([A-Za-z_]\w*)", "type"),
    ],
    "bash": [
        (r"^\s*(?:function\s+)?([A-Za-z_][\w-]*)\s*\(\)\s*\{", "function"),
    ],
    "powershell": [
        (r"^\s*function\s+([A-Za-z_][\w-]*)", "function"),
        (r"^\s*class\s+([A-Za-z_]\w*)", "class"),
    ],
    "sql": [
        (r"^\s*(?:create|alter)\s+(?:or\s+replace\s+)?(?:table|view|procedure|function)\s+([\w.\[\]\"]+)", "ddl"),
    ],
}

_IMPORT_RE = re.compile(
    r"^\s*(?:import\b|from\s+\S+\s+import\b|require\s*\(|using\s+[\w.]+|use\s+[\w:]+|#include\b|export\s+.*\bfrom\b)"
)


class CodeConverter(DocumentConverter):
    """代码 → 结构化大纲 + （可截断的）高亮代码块。

    大文件不再整份塞给大脑：``outline`` 模式保留符号表与首尾片段，
    中间用 ``# ... 省略 N 行 ...`` 标记，token 大幅下降且结构不丢。
    """

    name = "code"

    def __init__(self, *, max_lines: int = 400, max_symbols: int = 80,
                 max_bytes: int = 8_000_000) -> None:
        self.max_lines = int(max_lines)
        self.max_symbols = int(max_symbols)
        self.max_bytes = int(max_bytes)

    def accepts(self, file_stream: BinaryIO, stream_info: StreamInfo, **kwargs: Any) -> bool:
        if kwargs.get("as_document"):
            return False
        ext = (stream_info.extension or "").lower()
        mimetype = (stream_info.mimetype or "").lower()
        if ext in _LANG_BY_EXT:
            return True
        return mimetype in _LANG_BY_MIME or mimetype.startswith("text/x-")

    def convert(
        self, file_stream: BinaryIO, stream_info: StreamInfo, **kwargs: Any
    ) -> DocumentConverterResult:
        data = file_stream.read()
        if len(data) > self.max_bytes:
            raise UnsupportedContentError(f"代码文件过大：{len(data)} bytes")
        text = _decode_text(data, stream_info.charset)
        language = self._language(text, stream_info)
        outline = self._outline(text, language)
        lines = text.splitlines()
        loc = len(lines)
        truncated = loc > self.max_lines
        body = self._render_code(lines, truncated)
        markdown = f"```{language}\n{body}\n```"
        compact = "\n".join(
            f"{row[0]}\t{row[1]}\t{row[2]}" for row in outline["rows"]
        )
        return DocumentConverterResult(
            kind=ResultKind.STRUCTURED,
            markdown=markdown,
            title=stream_info.filename or stream_info.local_path or f"code.{language}",
            structured=StructuredPayload(
                format="code",
                columns=["line", "kind", "name", "signature"],
                rows=outline["rows"],
                schema={
                    "language": language,
                    "loc": loc,
                    "imports": outline["imports"],
                    "symbol_count": len(outline["rows"]),
                },
                truncated=truncated,
                meta={"outline": truncated},
            ),
            meta={
                "language": language,
                "loc": loc,
                "truncated": truncated,
                "symbols": len(outline["rows"]),
                "imports": outline["imports"],
                "outline": outline["rows"],
            },
            converter=self.name,
            tokens=TokenReport(
                original=estimate_text_tokens(text),
                output=estimate_text_tokens(markdown),
                strategy="code-outline" if truncated else "code",
                detail={"loc": loc, "language": language},
            ),
        )

    # -- 内部 ----------------------------------------------------------------
    def _language(self, text: str, info: StreamInfo) -> str:
        ext = (info.extension or "").lower()
        if ext in _LANG_BY_EXT:
            return _LANG_BY_EXT[ext]
        mime = (info.mimetype or "").lower()
        if mime in _LANG_BY_MIME:
            return _LANG_BY_MIME[mime]
        head = text[:400]
        if head.startswith("#!"):
            first = head.splitlines()[0]
            for key, lang in (("python", "python"), ("node", "javascript"), ("bash", "bash"),
                              ("sh", "bash"), ("pwsh", "powershell"), ("ruby", "ruby")):
                if key in first:
                    return lang
        if re.search(r"^\s*(?:from\s+\S+\s+import|def\s+\w+\s*\()", text, re.M):
            return "python"
        if re.search(r"^\s*#include\b", text, re.M):
            return "cpp"
        if re.search(r"^\s*(?:public\s+class|package\s+[\w.]+;)", text, re.M):
            return "java"
        if re.search(r"^\s*func\s+\w+\s*\(", text, re.M):
            return "go"
        if re.search(r"^\s*(?:const|let|function|interface)\b", text, re.M):
            return "javascript"
        return "text"

    def _outline(self, text: str, language: str) -> dict[str, Any]:
        patterns = _SYMBOL_PATTERNS.get(language)
        if patterns is None:
            for fallback in ("javascript", "python", "cpp"):
                if language in ("jsx", "tsx", "typescript", "vue", "svelte", "dart"):
                    patterns = _SYMBOL_PATTERNS.get(fallback) or _SYMBOL_PATTERNS["javascript"]
                    break
        patterns = patterns or _SYMBOL_PATTERNS["javascript"]

        rows: list[list[Any]] = []
        imports: list[str] = []
        for index, line in enumerate(text.splitlines(), 1):
            if _IMPORT_RE.match(line):
                if len(imports) < 40:
                    imports.append(line.strip())
                continue
            for pattern, kind in patterns:
                match = re.match(pattern, line)
                if not match:
                    continue
                name = match.group(1)
                signature = line.strip()
                if len(signature) > 160:
                    signature = signature[:157] + "..."
                rows.append([index, kind, name, signature])
                break
            if len(rows) >= self.max_symbols:
                break
        return {"rows": rows, "imports": imports}

    def _render_code(self, lines: list[str], truncated: bool) -> str:
        if not truncated:
            return "\n".join(lines)
        head_n = int(self.max_lines * 0.7)
        tail_n = max(10, self.max_lines - head_n)
        head = lines[:head_n]
        tail = lines[-tail_n:]
        omitted = len(lines) - len(head) - len(tail)
        marker = f"# ... 省略 {omitted} 行（共 {len(lines)} 行）..."
        return "\n".join(head + [marker] + tail)


# ===========================================================================
# 10. 视频转换器：关键帧采样 + 抽帧 + 转写钩子
# ===========================================================================


class _Cv2VideoReader:
    """OpenCV 视频读取器（默认后端）。"""

    def __init__(self, path: str) -> None:
        self.path = path
        self.meta: dict[str, Any] = {}
        self._cap = None

    def __iter__(self) -> Iterator[Any]:
        try:
            import cv2  # type: ignore
        except Exception as exc:  # pragma: no cover
            raise MissingDependencyException("视频抽帧需要 opencv-python") from exc
        cap = cv2.VideoCapture(self.path)
        if not cap.isOpened():
            raise UnsupportedContentError("无法打开视频（编码不支持或文件损坏）")
        self._cap = cap
        fps = float(cap.get(cv2.CAP_PROP_FPS) or 0.0) or 25.0
        count = int(cap.get(cv2.CAP_PROP_FRAME_COUNT) or 0)
        width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH) or 0)
        height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT) or 0)
        self.meta.update({"fps": fps, "frame_count": count,
                          "duration_s": round(count / fps, 3) if count else 0.0,
                          "size": [width, height]})
        index = 0
        try:
            while True:
                ok, frame = cap.read()
                if not ok:
                    break
                yield index / fps, frame
                index += 1
        finally:
            cap.release()
            self._cap = None


class VideoConverter(DocumentConverter):
    """视频 → 关键帧序列（``kind=FRAME``）+ 转写挂点。

    分阶段实现：Phase 2 只做关键帧（均匀 + 场景切换），转写留 ``transcribe``
    回调；音频轨抽离与 ASR 在 Phase 3 补全。视频**不转文本**，仍以帧喂大脑。
    """

    name = "video"
    _EXT = {".mp4", ".mov", ".avi", ".mkv", ".webm", ".flv", ".wmv", ".m4v", ".mpg", ".mpeg"}

    def __init__(
        self,
        *,
        max_frames: int = 6,
        frame_max_width: int = 960,
        jpeg_quality: int = 82,
        reader_factory: Callable[[str | bytes], Any] | None = None,
        transcribe: Callable[[str | bytes], str | None] | None = None,
        scene_threshold: float = 16.0,
    ) -> None:
        self.max_frames = max(2, int(max_frames))
        self.frame_max_width = int(frame_max_width)
        self.jpeg_quality = int(jpeg_quality)
        self.reader_factory = reader_factory
        self.transcribe = transcribe
        self.scene_threshold = float(scene_threshold)

    def accepts(self, file_stream: BinaryIO, stream_info: StreamInfo, **kwargs: Any) -> bool:
        mimetype = (stream_info.mimetype or "").lower()
        ext = (stream_info.extension or "").lower()
        return mimetype.startswith("video/") or ext in self._EXT

    def convert(
        self, file_stream: BinaryIO, stream_info: StreamInfo, **kwargs: Any
    ) -> DocumentConverterResult:
        data = file_stream.read()
        if not data:
            raise UnsupportedContentError("视频内容为空")
        samples = self._read_samples(data, stream_info)
        if not samples:
            raise UnsupportedContentError("未能从视频中解码出任何帧")
        picked = self._select_keyframes(samples)
        frames = [self._encode_frame(item) for item in picked]
        meta_reader = getattr(self, "_last_reader_meta", {})
        transcript = self._transcribe(data, stream_info)
        notes = []
        if transcript is None:
            notes.append("转写未启用：可注入 transcribe 回调（Phase 3 接 whisper）")
        original = estimate_image_tokens(self.frame_max_width,
                                        int(self.frame_max_width * 9 / 16)) * max(1, len(samples))
        return DocumentConverterResult(
            kind=ResultKind.FRAME,
            markdown="",
            title=(transcript.splitlines()[0][:80] if transcript else None)
            or stream_info.filename
            or stream_info.local_path
            or "video",
            frames=frames,
            meta={
                "fps": meta_reader.get("fps"),
                "duration_s": meta_reader.get("duration_s"),
                "frame_count": len(samples),
                "sampled": len(frames),
                "keyframes": [
                    {"index": f.meta.get("index"), "timestamp_s": f.meta.get("timestamp_s")}
                    for f in frames
                ],
                "transcript": transcript,
                "notes": notes,
            },
            converter=self.name,
            tokens=TokenReport(
                original=original,
                output=sum(f.tokens for f in frames),
                strategy="keyframes",
                detail={"frame_count": len(samples), "keyframes": len(frames)},
            ),
        )

    # -- 内部 ----------------------------------------------------------------
    def _read_samples(self, data: bytes, info: StreamInfo) -> list[tuple[float, Any]]:
        path = info.local_path
        tmp_path = None
        if self.reader_factory is None and (path is None or not os.path.exists(str(path))):
            fd, tmp_path = tempfile.mkstemp(suffix=(info.extension or ".mp4"))
            with os.fdopen(fd, "wb") as fh:
                fh.write(data)
            path = tmp_path
        try:
            if self.reader_factory is not None:
                reader = self.reader_factory(path if path else data)
            else:
                reader = _Cv2VideoReader(str(path))
            samples: list[tuple[float, Any]] = []
            for index, item in enumerate(reader):
                if isinstance(item, tuple) and len(item) == 2:
                    timestamp, pixels = item
                else:
                    pixels = item
                    fps = float(getattr(reader, "meta", {}).get("fps", 25.0) or 25.0)
                    timestamp = index / fps
                samples.append((float(timestamp), pixels))
            self._last_reader_meta = dict(getattr(reader, "meta", {}) or {})
            return samples
        finally:
            if tmp_path and os.path.exists(tmp_path):
                try:
                    os.remove(tmp_path)
                except OSError:
                    pass

    def _select_keyframes(self, samples: Sequence[tuple[float, Any]]) -> list[tuple[float, Any]]:
        count = len(samples)
        if count <= self.max_frames:
            return list(samples)
        picked: list[int] = [0]
        last_gray = self._thumb(samples[0][1])
        for index in range(1, count - 1):
            if len(picked) >= self.max_frames - 1:
                break
            gray = self._thumb(samples[index][1])
            if gray is None or last_gray is None:
                continue
            diff = float(np.mean(np.abs(gray - last_gray)))
            if diff >= self.scene_threshold:
                picked.append(index)
                last_gray = gray
        # 均匀补位到 max_frames（含末帧）
        if len(picked) < self.max_frames:
            remaining = [i for i in range(count) if i not in picked]
            need = self.max_frames - len(picked)
            if remaining:
                step = max(1, len(remaining) // max(1, need))
                for i in remaining[::step][:max(0, need - 1)]:
                    picked.append(i)
        if (count - 1) not in picked:
            picked.append(count - 1)
        picked = sorted(set(picked))[: self.max_frames]
        return [samples[i] for i in picked]

    def _thumb(self, pixels: Any) -> np.ndarray | None:
        try:
            bgr = _as_bgr(pixels)
        except Exception:
            return None
        small = _resize(bgr, 64, 36)
        return _luma(small)

    def _encode_frame(self, item: tuple[float, Any]) -> EncodedFrame:
        timestamp, pixels = item
        bgr = _as_bgr(pixels)
        h, w = bgr.shape[:2]
        scale = min(1.0, self.frame_max_width / float(w))
        out_w, out_h = max(16, int(round(w * scale))), max(9, int(round(h * scale)))
        resized = _resize(bgr, out_w, out_h)
        data = encode_image(resized, quality=self.jpeg_quality)
        return EncodedFrame(
            data=data,
            width=out_w,
            height=out_h,
            rect=_rect(0, 0, w, h),
            scale=out_w / float(w),
            strategy="keyframe",
            meta={"timestamp_s": round(float(timestamp), 3)},
        )

    def _transcribe(self, data: bytes, info: StreamInfo) -> str | None:
        if self.transcribe is None:
            return None
        try:
            return self.transcribe(info.local_path or data)
        except Exception:
            return None


# ===========================================================================
# 11. 文档转换器（保留 MarkItDown 原有能力）
# ===========================================================================


class PlainTextConverter(DocumentConverter):
    """纯文本 → markdown（内容原样保留，只做行尾与空行规整）。"""

    name = "plain_text"
    _EXT = {".txt", ".text", ".log", ".rst", ".nfo", ".asc", ".srt", ".vtt", ".csv~"}

    def accepts(self, file_stream: BinaryIO, stream_info: StreamInfo, **kwargs: Any) -> bool:
        mimetype = (stream_info.mimetype or "").lower()
        ext = (stream_info.extension or "").lower()
        if ext in self._EXT:
            return True
        return mimetype.startswith("text/") and not mimetype.startswith("text/x-")

    def convert(
        self, file_stream: BinaryIO, stream_info: StreamInfo, **kwargs: Any
    ) -> DocumentConverterResult:
        text = _decode_text(file_stream.read(), stream_info.charset)
        title = stream_info.filename or stream_info.local_path
        return DocumentConverterResult(
            kind=ResultKind.MARKDOWN,
            markdown=text.strip(),
            title=os.path.splitext(os.path.basename(title))[0] if title else None,
            meta={"chars": len(text), "lines": text.count("\n") + 1},
            converter=self.name,
        )


class MarkdownConverter(DocumentConverter):
    """markdown → markdown（原样透传，标题取第一个 H1）。"""

    name = "markdown"
    _EXT = {".md", ".markdown", ".mdown", ".mkd"}

    def accepts(self, file_stream: BinaryIO, stream_info: StreamInfo, **kwargs: Any) -> bool:
        return (stream_info.extension or "").lower() in self._EXT or \
            (stream_info.mimetype or "").lower() in ("text/markdown", "text/x-markdown")

    def convert(
        self, file_stream: BinaryIO, stream_info: StreamInfo, **kwargs: Any
    ) -> DocumentConverterResult:
        text = _decode_text(file_stream.read(), stream_info.charset)
        title = None
        match = re.search(r"^#\s+(.+)$", text, re.M)
        if match:
            title = match.group(1).strip()
        return DocumentConverterResult(
            kind=ResultKind.MARKDOWN,
            markdown=text.strip(),
            title=title or stream_info.filename,
            meta={"chars": len(text)},
            converter=self.name,
        )


class _HtmlToMarkdown:
    """极简 HTML → markdown 转换（标题/段落/列表/表格/链接/代码/图片）。"""

    _SKIP = {"script", "style", "noscript", "head", "svg", "iframe"}

    def __init__(self) -> None:
        self.parts: list[str] = []
        self.title = ""
        self._skip_depth = 0
        self._in_title = False
        self._pre = 0
        self._code = 0
        self._href: str | None = None
        self._list_depth = 0
        self._ordered: list[bool] = []
        self._table: list[list[str]] | None = None
        self._row: list[str] | None = None
        self._cell: list[str] | None = None
        self._quote = 0

    # -- HTMLParser 接口 -----------------------------------------------------
    def feed(self, html: str) -> str:
        from html.parser import HTMLParser

        outer = self

        class _Parser(HTMLParser):
            def handle_starttag(self, tag, attrs):  # type: ignore[override]
                outer._start(tag, dict(attrs))

            def handle_startendtag(self, tag, attrs):  # type: ignore[override]
                outer._start(tag, dict(attrs), void=True)

            def handle_endtag(self, tag):  # type: ignore[override]
                outer._end(tag)

            def handle_data(self, data):  # type: ignore[override]
                outer._data(data)

            def handle_entityref(self, name):  # type: ignore[override]
                outer._data(f"&{name};")

        parser = _Parser(convert_charrefs=True)
        parser.feed(html)
        parser.close()
        return self._finish()

    # -- 内部 ----------------------------------------------------------------
    def _emit(self, text: str) -> None:
        if self._cell is not None:
            self._cell.append(text)
            return
        self.parts.append(text)

    def _start(self, tag: str, attrs: dict[str, Any], void: bool = False) -> None:
        tag = tag.lower()
        if tag in self._SKIP:
            self._skip_depth += 1 if not void else 0
            return
        if tag == "title":
            self._in_title = True
            return
        if re.fullmatch(r"h[1-6]", tag):
            self._emit("\n\n" + "#" * int(tag[1]) + " ")
        elif tag == "p":
            self._emit("\n\n")
        elif tag == "br":
            self._emit("  \n")
        elif tag == "hr":
            self._emit("\n\n---\n\n")
        elif tag in ("ul", "ol"):
            self._list_depth += 1
            self._ordered.append(tag == "ol")
            self._emit("\n")
        elif tag == "li":
            indent = "  " * max(0, self._list_depth - 1)
            marker = "1. " if (self._ordered and self._ordered[-1]) else "- "
            self._emit(f"\n{indent}{marker}")
        elif tag == "blockquote":
            self._quote += 1
            self._emit("\n\n> ")
        elif tag == "pre":
            self._pre += 1
            lang = attrs.get("data-lang") or ""
            if not lang:
                classes = str(attrs.get("class") or "")
                match = re.search(r"(?:language|lang)-(\w+)", classes)
                lang = match.group(1) if match else ""
            self._emit(f"\n\n```{lang}\n")
        elif tag == "code":
            if not self._pre:
                self._code += 1
                self._emit("`")
        elif tag == "a":
            self._href = attrs.get("href") or ""
            self._emit("[")
        elif tag == "img":
            alt = attrs.get("alt") or ""
            src = attrs.get("src") or ""
            self._emit(f"![{alt}]({src})")
        elif tag in ("strong", "b"):
            self._emit("**")
        elif tag in ("em", "i"):
            self._emit("*")
        elif tag == "table":
            self._table = []
        elif tag == "tr" and self._table is not None:
            self._row = []
        elif tag in ("td", "th") and self._row is not None:
            self._cell = []
        elif tag == "thead":
            pass

    def _end(self, tag: str) -> None:
        tag = tag.lower()
        if tag in self._SKIP:
            self._skip_depth = max(0, self._skip_depth - 1)
            return
        if tag == "title":
            self._in_title = False
            return
        if re.fullmatch(r"h[1-6]", tag):
            self._emit("\n\n")
        elif tag in ("p", "div", "section", "article"):
            self._emit("\n\n")
        elif tag in ("ul", "ol"):
            self._list_depth = max(0, self._list_depth - 1)
            if self._ordered:
                self._ordered.pop()
            self._emit("\n")
        elif tag == "blockquote":
            self._quote = max(0, self._quote - 1)
            self._emit("\n\n")
        elif tag == "pre":
            self._pre = max(0, self._pre - 1)
            self._emit("\n```\n\n")
        elif tag == "code" and not self._pre:
            self._code = max(0, self._code - 1)
            self._emit("`")
        elif tag == "a":
            href = self._href or ""
            self._href = None
            self._emit(f"]({href})" if href else "]")
        elif tag in ("strong", "b"):
            self._emit("**")
        elif tag in ("em", "i"):
            self._emit("*")
        elif tag in ("td", "th"):
            if self._row is not None and self._cell is not None:
                cell = re.sub(r"\s+", " ", "".join(self._cell)).strip()
                self._row.append(cell)
            self._cell = None
        elif tag == "tr":
            if self._table is not None and self._row is not None:
                self._table.append(self._row)
            self._row = None
        elif tag == "table":
            self._emit(self._render_table())
            self._table = None

    def _data(self, data: str) -> None:
        if self._skip_depth:
            return
        if self._in_title:
            self.title += data
            return
        text = data if (self._pre or self._code) else re.sub(r"[ \t\r\n]+", " ", data)
        if text.strip() or self._pre:
            self._emit(text)

    def _render_table(self) -> str:
        table = self._table or []
        if not table:
            return ""
        width = max(len(r) for r in table)
        rows = [r + [""] * (width - len(r)) for r in table]
        head, body = rows[0], rows[1:]
        lines = ["", "| " + " | ".join(head) + " |",
                 "| " + " | ".join("---" for _ in head) + " |"]
        lines += ["| " + " | ".join(r) + " |" for r in body]
        return "\n".join(lines) + "\n\n"

    def _finish(self) -> str:
        text = "".join(self.parts)
        text = re.sub(r"[ \t]+\n", "\n", text)
        text = re.sub(r"\n{3,}", "\n\n", text)
        return text.strip()


class HtmlConverter(DocumentConverter):
    """HTML / 网页 → markdown。"""

    name = "html"
    _EXT = {".html", ".htm", ".xhtml", ".shtml"}

    def accepts(self, file_stream: BinaryIO, stream_info: StreamInfo, **kwargs: Any) -> bool:
        mimetype = (stream_info.mimetype or "").lower()
        ext = (stream_info.extension or "").lower()
        return ext in self._EXT or mimetype in ("text/html", "application/xhtml+xml")

    def convert(
        self, file_stream: BinaryIO, stream_info: StreamInfo, **kwargs: Any
    ) -> DocumentConverterResult:
        text = _decode_text(file_stream.read(), stream_info.charset)
        converter = _HtmlToMarkdown()
        markdown = converter.feed(text)
        return DocumentConverterResult(
            kind=ResultKind.MARKDOWN,
            markdown=markdown,
            title=(converter.title or stream_info.filename or "").strip() or None,
            meta={"chars": len(text), "links": len(re.findall(r"\]\(", markdown))},
            converter=self.name,
        )


class DocxConverter(DocumentConverter):
    """Word (.docx) → markdown（标题/段落/表格；不依赖 python-docx）。"""

    name = "docx"
    _EXT = {".docx", ".docm"}
    _W = "{http://schemas.openxmlformats.org/wordprocessingml/2006/main}"

    def accepts(self, file_stream: BinaryIO, stream_info: StreamInfo, **kwargs: Any) -> bool:
        ext = (stream_info.extension or "").lower()
        mimetype = (stream_info.mimetype or "").lower()
        return ext in self._EXT or "wordprocessingml" in mimetype or \
            mimetype == "application/vnd.openxmlformats-officedocument.wordprocessingml.document"

    def convert(
        self, file_stream: BinaryIO, stream_info: StreamInfo, **kwargs: Any
    ) -> DocumentConverterResult:
        data = file_stream.read()
        try:
            with zipfile.ZipFile(io.BytesIO(data)) as archive:
                xml = archive.read("word/document.xml")
                core_title = self._core_title(archive)
        except (zipfile.BadZipFile, KeyError) as exc:
            raise UnsupportedContentError(f"不是有效的 docx：{exc}") from exc
        root = ET.fromstring(xml)
        body = root.find(f"{self._W}body")
        blocks: list[str] = []
        W = self._W
        if body is not None:
            for child in body:
                if child.tag == f"{W}p":
                    text = "".join(node.text or "" for node in child.iter(f"{W}t")).strip()
                    if not text:
                        continue
                    style = child.find(f"{W}pPr/{W}pStyle")
                    level = self._heading_level(style.get(f"{W}val") if style is not None else None)
                    if level:
                        blocks.append("#" * level + " " + text)
                    elif child.find(f"{W}pPr/{W}numPr") is not None:
                        blocks.append("- " + text)
                    else:
                        blocks.append(text)
                elif child.tag == f"{W}tbl":
                    blocks.append(self._render_table(child))
        markdown = "\n\n".join(b for b in blocks if b).strip()
        return DocumentConverterResult(
            kind=ResultKind.MARKDOWN,
            markdown=markdown,
            title=core_title or stream_info.filename,
            meta={"blocks": len(blocks)},
            converter=self.name,
        )

    def _core_title(self, archive: zipfile.ZipFile) -> str | None:
        try:
            core = ET.fromstring(archive.read("docProps/core.xml"))
        except Exception:
            return None
        for tag in ("{http://purl.org/dc/elements/1.1/}title", "title"):
            node = core.find(tag)
            if node is not None and node.text:
                return node.text.strip()
        return None

    def _heading_level(self, style: str | None) -> int:
        if not style:
            return 0
        match = re.search(r"(?:Heading|heading|标题)\s*(\d)", style)
        if match:
            return max(1, min(6, int(match.group(1))))
        if style.lower() in ("title", "subtitle"):
            return 1 if style.lower() == "title" else 2
        return 0

    def _render_table(self, node: ET.Element) -> str:
        W = self._W
        rows: list[list[str]] = []
        for tr in node.findall(f"{W}tr"):
            cells: list[str] = []
            for tc in tr.findall(f"{W}tc"):
                text = "".join(t.text or "" for t in tc.iter(f"{W}t")).strip()
                cells.append(re.sub(r"\s+", " ", text))
            if cells:
                rows.append(cells)
        if not rows:
            return ""
        width = max(len(r) for r in rows)
        rows = [r + [""] * (width - len(r)) for r in rows]
        lines = ["| " + " | ".join(rows[0]) + " |",
                 "| " + " | ".join("---" for _ in rows[0]) + " |"]
        lines += ["| " + " | ".join(r) + " |" for r in rows[1:]]
        return "\n".join(lines)


class XlsxConverter(DocumentConverter):
    """Excel (.xlsx) → 结构化（多工作表，行列不丢）+ markdown 视图。"""

    name = "xlsx"
    _EXT = {".xlsx", ".xlsm"}

    def __init__(self, *, max_rows: int = 2000) -> None:
        self.max_rows = int(max_rows)

    def accepts(self, file_stream: BinaryIO, stream_info: StreamInfo, **kwargs: Any) -> bool:
        ext = (stream_info.extension or "").lower()
        mimetype = (stream_info.mimetype or "").lower()
        return ext in self._EXT or "spreadsheetml" in mimetype

    def convert(
        self, file_stream: BinaryIO, stream_info: StreamInfo, **kwargs: Any
    ) -> DocumentConverterResult:
        try:
            import openpyxl  # type: ignore
        except Exception as exc:  # pragma: no cover
            raise MissingDependencyException("读取 xlsx 需要 openpyxl（pip install openpyxl）") from exc
        data = file_stream.read()
        workbook = openpyxl.load_workbook(io.BytesIO(data), data_only=True, read_only=True)
        sheets: list[dict[str, Any]] = []
        sections: list[str] = []
        total_rows = 0
        truncated = False
        try:
            for sheet in workbook.worksheets:
                rows: list[list[Any]] = []
                for row in sheet.iter_rows(values_only=True):
                    if row is None or not any(v is not None and str(v).strip() != "" for v in row):
                        continue
                    rows.append([_scalar(v) for v in row])
                    if len(rows) > self.max_rows:
                        truncated = True
                        break
                if not rows:
                    continue
                header = [str(v) if v is not None else f"col_{i + 1}" for i, v in enumerate(rows[0])]
                body = rows[1:]
                total_rows += len(body)
                sheets.append({"name": sheet.title, "columns": header, "rows": body})
                table = StructuredPayload(format="table", columns=header, rows=body)
                sections.append(f"## {sheet.title}\n\n{table.to_markdown()}")
        finally:
            try:
                workbook.close()
            except Exception:
                pass
        payload = StructuredPayload(
            format="sheets",
            columns=sheets[0]["columns"] if sheets else [],
            rows=sheets[0]["rows"] if sheets else [],
            schema={"sheets": [{"name": s["name"], "rows": len(s["rows"]),
                                "cols": len(s["columns"])} for s in sheets]},
            truncated=truncated,
            meta={"sheets": sheets, "row_count": total_rows},
        )
        markdown = "\n\n".join(sections)
        return DocumentConverterResult(
            kind=ResultKind.STRUCTURED,
            markdown=markdown,
            title=stream_info.filename,
            structured=payload,
            meta={"sheets": len(sheets), "rows": total_rows, "truncated": truncated},
            converter=self.name,
            tokens=TokenReport(
                original=estimate_text_tokens(markdown),
                output=estimate_text_tokens(markdown),
                strategy="worksheet",
                detail={"sheets": len(sheets)},
            ),
        )


class PptxConverter(DocumentConverter):
    """PowerPoint (.pptx) → markdown（每页标题 + 要点 + 表格）。"""

    name = "pptx"
    _EXT = {".pptx", ".pptm"}

    def accepts(self, file_stream: BinaryIO, stream_info: StreamInfo, **kwargs: Any) -> bool:
        ext = (stream_info.extension or "").lower()
        mimetype = (stream_info.mimetype or "").lower()
        return ext in self._EXT or "presentationml" in mimetype

    def convert(
        self, file_stream: BinaryIO, stream_info: StreamInfo, **kwargs: Any
    ) -> DocumentConverterResult:
        try:
            from pptx import Presentation  # type: ignore
        except Exception as exc:  # pragma: no cover
            raise MissingDependencyException("读取 pptx 需要 python-pptx（pip install python-pptx）") from exc
        data = file_stream.read()
        presentation = Presentation(io.BytesIO(data))
        blocks: list[str] = []
        for index, slide in enumerate(presentation.slides, 1):
            title = ""
            try:
                if slide.shapes.title is not None:
                    title = (slide.shapes.title.text or "").strip()
            except Exception:
                title = ""
            lines: list[str] = [f"## Slide {index}" + (f": {title}" if title else "")]
            for shape in slide.shapes:
                if not getattr(shape, "has_text_frame", False):
                    continue
                for paragraph in shape.text_frame.paragraphs:
                    text = "".join(run.text for run in paragraph.runs).strip()
                    if not text or text == title:
                        continue
                    lines.append("- " + text)
            blocks.append("\n".join(lines))
        markdown = "\n\n".join(blocks)
        return DocumentConverterResult(
            kind=ResultKind.MARKDOWN,
            markdown=markdown,
            title=stream_info.filename,
            meta={"slides": len(presentation.slides)},
            converter=self.name,
        )


class PdfConverter(DocumentConverter):
    """PDF → markdown（pypdf 可选依赖；缺失时给出明确安装提示）。"""

    name = "pdf"
    _EXT = {".pdf"}

    def accepts(self, file_stream: BinaryIO, stream_info: StreamInfo, **kwargs: Any) -> bool:
        return (stream_info.extension or "").lower() in self._EXT or \
            (stream_info.mimetype or "").lower() == "application/pdf"

    def convert(
        self, file_stream: BinaryIO, stream_info: StreamInfo, **kwargs: Any
    ) -> DocumentConverterResult:
        try:
            from pypdf import PdfReader  # type: ignore
        except Exception as exc:
            raise MissingDependencyException(
                "读取 PDF 需要 pypdf（pip install pypdf）"
            ) from exc
        data = file_stream.read()
        reader = PdfReader(io.BytesIO(data))
        pages: list[str] = []
        for page in reader.pages:
            try:
                pages.append((page.extract_text() or "").strip())
            except Exception:
                pages.append("")
        title = None
        try:
            meta = reader.metadata or {}
            title = (meta.get("/Title") or None) if hasattr(meta, "get") else None
        except Exception:
            title = None
        markdown = "\n\n---\n\n".join(p for p in pages if p).strip()
        return DocumentConverterResult(
            kind=ResultKind.MARKDOWN,
            markdown=markdown,
            title=title or stream_info.filename,
            meta={"pages": len(pages)},
            converter=self.name,
            tokens=TokenReport(original=estimate_text_tokens(markdown),
                               output=estimate_text_tokens(markdown), strategy="pdf"),
        )


class EpubConverter(DocumentConverter):
    """EPUB → markdown（按 spine 顺序拼接章节）。"""

    name = "epub"
    _EXT = {".epub"}

    def accepts(self, file_stream: BinaryIO, stream_info: StreamInfo, **kwargs: Any) -> bool:
        return (stream_info.extension or "").lower() in self._EXT or \
            (stream_info.mimetype or "").lower() == "application/epub+zip"

    def convert(
        self, file_stream: BinaryIO, stream_info: StreamInfo, **kwargs: Any
    ) -> DocumentConverterResult:
        data = file_stream.read()
        try:
            archive = zipfile.ZipFile(io.BytesIO(data))
        except zipfile.BadZipFile as exc:
            raise UnsupportedContentError(f"不是有效的 EPUB：{exc}") from exc
        with archive:
            opf_path = self._rootfile(archive)
            if not opf_path:
                raise UnsupportedContentError("EPUB 缺少 OPF 根文件")
            opf = ET.fromstring(archive.read(opf_path))
            ns = {"opf": "http://www.idpf.org/2007/opf"}
            base = os.path.dirname(opf_path)
            manifest: dict[str, str] = {}
            for item in opf.findall(".//opf:manifest/opf:item", ns):
                manifest[item.get("id", "")] = item.get("href", "")
            chapters: list[str] = []
            for ref in opf.findall(".//opf:spine/opf:itemref", ns):
                href = manifest.get(ref.get("idref", ""))
                if not href:
                    continue
                path = os.path.normpath(os.path.join(base, href)).replace("\\", "/")
                try:
                    html = archive.read(path).decode("utf-8", errors="replace")
                except KeyError:
                    continue
                text = _HtmlToMarkdown().feed(html)
                if text.strip():
                    chapters.append(text.strip())
            markdown = "\n\n".join(chapters)
        return DocumentConverterResult(
            kind=ResultKind.MARKDOWN,
            markdown=markdown,
            title=stream_info.filename,
            meta={"chapters": len(chapters)},
            converter=self.name,
        )

    def _rootfile(self, archive: zipfile.ZipFile) -> str | None:
        try:
            container = ET.fromstring(archive.read("META-INF/container.xml"))
        except Exception:
            return None
        for node in container.iter():
            if node.tag.endswith("rootfile"):
                return node.get("full-path")
        return None


class ImageDocumentConverter(DocumentConverter):
    """图片当"文档"看 → markdown 图片引用（``as_document=True`` 时启用）。

    默认图片走 :class:`VisionConverter`（增强像素帧）；只有明确要文档形态
    （例如把图片塞进 markdown 报告）才用这里。
    """

    name = "image_document"
    _EXT = VisionConverter._IMAGE_EXT

    def accepts(self, file_stream: BinaryIO, stream_info: StreamInfo, **kwargs: Any) -> bool:
        if not kwargs.get("as_document"):
            return False
        mimetype = (stream_info.mimetype or "").lower()
        return mimetype.startswith("image/") or (stream_info.extension or "").lower() in self._EXT

    def convert(
        self, file_stream: BinaryIO, stream_info: StreamInfo, **kwargs: Any
    ) -> DocumentConverterResult:
        data = file_stream.read()
        mimetype = (stream_info.mimetype or "").lower() or "image/png"
        name = stream_info.filename or "image"
        if len(data) <= 1_500_000:
            encoded = base64.b64encode(data).decode("ascii")
            ref = f"data:{mimetype};base64,{encoded}"
        else:
            ref = f"embedded:{mimetype};{len(data)}bytes"
        return DocumentConverterResult(
            kind=ResultKind.MARKDOWN,
            markdown=f"![{os.path.splitext(name)[0]}]({ref})",
            title=name,
            meta={"bytes": len(data), "mimetype": mimetype},
            converter=self.name,
        )


class ZipConverter(DocumentConverter):
    """ZIP 等归档 → 清单 + 内嵌文件（骨架，Phase 3 递归转换）。"""

    name = "zip"
    _EXT = {".zip", ".tar", ".gz", ".tgz", ".7z", ".rar"}

    def accepts(self, file_stream: BinaryIO, stream_info: StreamInfo, **kwargs: Any) -> bool:
        ext = (stream_info.extension or "").lower()
        return ext in self._EXT or (stream_info.mimetype or "").lower() in (
            "application/zip", "application/x-tar", "application/gzip",
        )

    def convert(
        self, file_stream: BinaryIO, stream_info: StreamInfo, **kwargs: Any
    ) -> DocumentConverterResult:
        data = file_stream.read()
        entries: list[tuple[str, int]] = []
        if (stream_info.extension or "").lower() == ".zip" or data[:2] == b"PK":
            try:
                with zipfile.ZipFile(io.BytesIO(data)) as archive:
                    entries = [(i.filename, i.file_size) for i in archive.infolist() if not i.is_dir()]
            except zipfile.BadZipFile as exc:
                raise UnsupportedContentError(f"不是有效的 ZIP：{exc}") from exc
        else:
            raise MissingDependencyException("该归档格式需要额外依赖（Phase 3 支持）")
        lines = ["# Archive", ""]
        lines += [f"- `{name}` ({size} bytes)" for name, size in entries[:200]]
        if len(entries) > 200:
            lines.append(f"- ... 共 {len(entries)} 个文件")
        return DocumentConverterResult(
            kind=ResultKind.MARKDOWN,
            markdown="\n".join(lines),
            title=stream_info.filename,
            meta={"entries": len(entries)},
            converter=self.name,
        )


class FallbackConverter(DocumentConverter):
    """兜底：能当文本读就当文本，不能就明确报错。"""

    name = "fallback"
    generic = True

    def accepts(self, file_stream: BinaryIO, stream_info: StreamInfo, **kwargs: Any) -> bool:
        return True

    def convert(
        self, file_stream: BinaryIO, stream_info: StreamInfo, **kwargs: Any
    ) -> DocumentConverterResult:
        data = file_stream.read()
        if not data:
            return DocumentConverterResult(kind=ResultKind.MARKDOWN, markdown="",
                                           title=stream_info.filename, converter=self.name)
        if b"\x00" in data[:8192]:
            raise UnsupportedContentError(
                f"二进制内容（{stream_info.mimetype or '未知'}）不能当文本读，"
                "请注册对应的 DocumentConverter"
            )
        text = _decode_text(data, stream_info.charset)
        printable = sum(1 for ch in text[:8192] if ch.isprintable() or ch in "\n\r\t")
        ratio = printable / max(1, len(text[:8192]))
        if ratio < 0.92 or not _is_probably_text(data[:8192]):
            raise UnsupportedContentError(
                f"无法识别的内容类型（{stream_info.mimetype or '未知'}），"
                "请注册对应的 DocumentConverter"
            )
        return DocumentConverterResult(
            kind=ResultKind.MARKDOWN,
            markdown=text.strip(),
            title=stream_info.filename,
            meta={"fallback": True, "chars": len(text)},
            converter=self.name,
        )


# ===========================================================================
# 12. 类型嗅探（magika + 魔数回退）
# ===========================================================================

_MAGIKA_INSTANCE: Any = None
_MAGIKA_TRIED = False

_MAGIC_IMAGE = [
    (b"\x89PNG\r\n\x1a\n", "image/png", ".png"),
    (b"\xff\xd8\xff", "image/jpeg", ".jpg"),
    (b"GIF87a", "image/gif", ".gif"),
    (b"GIF89a", "image/gif", ".gif"),
    (b"BM", "image/bmp", ".bmp"),
    (b"II*\x00", "image/tiff", ".tiff"),
    (b"MM\x00*", "image/tiff", ".tiff"),
]

_VIDEO_HINTS = {".mp4", ".mov", ".avi", ".mkv", ".webm", ".flv", ".wmv", ".m4v"}


def _magika_guess(data: bytes) -> dict[str, Any] | None:
    """用 magika 识别类型（可选依赖；失败返回 None，不抛错）。"""
    global _MAGIKA_INSTANCE, _MAGIKA_TRIED
    if not _MAGIKA_TRIED:
        _MAGIKA_TRIED = True
        try:
            from magika import Magika  # type: ignore

            _MAGIKA_INSTANCE = Magika()
        except Exception:
            _MAGIKA_INSTANCE = None
    if _MAGIKA_INSTANCE is None:
        return None
    try:
        result = _MAGIKA_INSTANCE.identify_bytes(data[:8192])
    except Exception:
        return None
    mimetype = getattr(result, "predicted_mime_type", None)
    extension = getattr(result, "predicted_extension", None)
    score = getattr(result, "score", None)
    output = getattr(result, "output", None)
    if output is not None:
        mimetype = mimetype or getattr(output, "mime_type", None)
        extension = extension or getattr(output, "extension", None)
    if not mimetype:
        return None
    if extension and not str(extension).startswith("."):
        extension = "." + str(extension)
    return {"mimetype": str(mimetype), "extension": extension, "score": score, "backend": "magika"}


def _is_probably_text(head: bytes) -> bool:
    if not head:
        return False
    if b"\x00" in head:
        return False
    try:
        text = head.decode("utf-8")
    except UnicodeDecodeError:
        return False
    printable = sum(1 for ch in text if ch.isprintable() or ch in "\n\r\t")
    return printable / max(1, len(text)) >= 0.92


# ===========================================================================
# 13. 引擎
# ===========================================================================


class OpticNerve:
    """视神经转化引擎（注册表 + 分流 + 预算）。

    >>> nerve = OpticNerve()
    >>> nerve.convert("a.csv").kind           # STRUCTURED
    >>> nerve.convert("a.docx").markdown      # markdown
    >>> nerve.convert(frame).image.tokens     # 增强帧
    """

    def __init__(
        self,
        *,
        config: Any = None,
        options: VisionOptions | None = None,
        enable_builtins: bool = True,
        converters: Sequence[DocumentConverter] = (),
        load_plugins: bool = False,
    ) -> None:
        self._config = config
        self._registry: list[ConverterRegistration] = []
        self._builtins_loaded = False
        section = _config_section(config, "optic_nerve")
        self.options = options or VisionOptions.from_mapping(section)
        self.stats: dict[str, Any] = {
            "conversions": 0,
            "by_kind": {k.value: 0 for k in ResultKind},
            "tokens_in": 0,
            "tokens_out": 0,
            "errors": 0,
        }
        self.last_errors: list[str] = []
        for converter in converters:
            self.register_converter(converter)
        if enable_builtins:
            self.enable_builtins()
        if load_plugins:
            self.load_plugins()

    # -- 注册 ----------------------------------------------------------------
    def register_converter(
        self, converter: DocumentConverter, *, priority: float = PRIORITY_SPECIFIC
    ) -> "OpticNerve":
        """注册转换器（同类只保留最新一次，插件可热替换）。"""
        if not isinstance(converter, DocumentConverter):
            raise TypeError("converter 必须是 DocumentConverter 子类实例")
        self._registry = [r for r in self._registry if type(r.converter) is not type(converter)]
        self._registry.insert(0, ConverterRegistration(converter=converter, priority=float(priority)))
        self._registry.sort(key=lambda r: r.priority)
        return self

    def unregister_converter(self, converter_type: type) -> bool:
        before = len(self._registry)
        self._registry = [r for r in self._registry if type(r.converter) is not converter_type]
        return len(self._registry) != before

    def enable_builtins(self) -> "OpticNerve":
        """注册内置转换器（幂等）。"""
        if self._builtins_loaded:
            return self
        self._builtins_loaded = True
        vision = VisionConverter(self.options)
        builtins: list[tuple[DocumentConverter, float]] = [
            (vision, PRIORITY_SPECIFIC),
            (VideoConverter(), PRIORITY_SPECIFIC + 1.0),
            (StructuredConverter(), PRIORITY_STRUCTURED),
            (CodeConverter(), PRIORITY_STRUCTURED + 2.0),
            (MarkdownConverter(), PRIORITY_DOCUMENT - 10.0),
            (HtmlConverter(), PRIORITY_DOCUMENT),
            (DocxConverter(), PRIORITY_DOCUMENT),
            (EpubConverter(), PRIORITY_DOCUMENT + 1.0),
            (PptxConverter(), PRIORITY_DOCUMENT + 2.0),
            (PdfConverter(), PRIORITY_DOCUMENT + 3.0),
            (XlsxConverter(), PRIORITY_DOCUMENT + 4.0),
            (ImageDocumentConverter(), PRIORITY_DOCUMENT + 10.0),
            (ZipConverter(), PRIORITY_DOCUMENT + 20.0),
            (PlainTextConverter(), PRIORITY_PLAIN_TEXT),
            (FallbackConverter(), PRIORITY_FALLBACK),
        ]
        for converter, priority in builtins:
            self.register_converter(converter, priority=priority)
        return self

    def load_plugins(self, group: str = "optic_nerve.plugin") -> "OpticNerve":
        """加载第三方插件（``entry_points(group="optic_nerve.plugin")``）。"""
        try:
            from importlib.metadata import entry_points

            eps = entry_points()
            selected = eps.select(group=group) if hasattr(eps, "select") else eps.get(group, [])
            for ep in selected:
                try:
                    factory = ep.load()
                    converter = factory() if isinstance(factory, type) else factory
                    self.register_converter(converter)
                except Exception as exc:  # 插件坏了不能拖垮引擎
                    self.last_errors.append(f"plugin {ep.name}: {exc}")
        except Exception:
            pass
        return self

    @property
    def converters(self) -> list[ConverterRegistration]:
        return list(self._registry)

    # -- 类型嗅探 ------------------------------------------------------------
    def sniff(self, file_stream: BinaryIO, base: StreamInfo | None = None) -> list[StreamInfo]:
        """返回候选 StreamInfo（按可信度排序），供 accept 逐个尝试。"""
        position = file_stream.tell() if hasattr(file_stream, "tell") else None
        try:
            head = file_stream.read(8192) or b""
        except Exception:
            head = b""
        container = head
        if position is not None and head[:2] == b"PK":
            # ZIP 族要读中央目录，8KB 前缀不够：能 seek 且不太大时读全量
            try:
                file_stream.seek(0, os.SEEK_END)
                size = int(file_stream.tell())
                if 0 < size <= 64 * 1024 * 1024:
                    file_stream.seek(0)
                    container = file_stream.read(size) or head
            except Exception:
                container = head
        if position is not None:
            try:
                file_stream.seek(position)
            except Exception:
                pass

        candidates: list[StreamInfo] = []
        if base is not None:
            candidates.append(base)

        guess = _magika_guess(head) if head else None
        if guess:
            candidates.append(
                StreamInfo(
                    mimetype=guess["mimetype"],
                    extension=guess.get("extension"),
                    filename=(base.filename if base else None),
                    local_path=(base.local_path if base else None),
                    url=(base.url if base else None),
                )
            )

        magic = self._magic_guess(head)
        if magic:
            candidates.append(
                StreamInfo(mimetype=magic[0], extension=magic[1],
                           filename=(base.filename if base else None),
                           local_path=(base.local_path if base else None))
            )

        if base is not None and base.filename:
            guessed_mime, _ = mimetypes.guess_type(base.filename)
            if guessed_mime:
                candidates.append(base.copy_and_update(mimetype=guessed_mime))

        if head[:2] == b"PK":
            candidates.extend(self._zip_variants(container, base))

        if _is_probably_text(head):
            text = head.lstrip()
            if text[:15].lower().startswith((b"<!doctype html", b"<html")):
                candidates.append(StreamInfo(mimetype="text/html", extension=".html",
                                             filename=(base.filename if base else None)))
            elif text[:1] in (b"{", b"["):
                candidates.append(StreamInfo(mimetype="application/json", extension=".json",
                                             filename=(base.filename if base else None)))
            else:
                candidates.append(StreamInfo(mimetype="text/plain", extension=".txt",
                                             filename=(base.filename if base else None)))

        candidates.append(StreamInfo(mimetype="application/octet-stream",
                                     filename=(base.filename if base else None),
                                     local_path=(base.local_path if base else None)))

        seen: set[tuple[str, str | None]] = set()
        unique: list[StreamInfo] = []
        for info in candidates:
            key = ((info.mimetype or "").lower(), (info.extension or "").lower())
            if key in seen:
                continue
            seen.add(key)
            unique.append(info)
        return unique

    def _magic_guess(self, head: bytes) -> tuple[str, str] | None:
        for signature, mimetype, extension in _MAGIC_IMAGE:
            if head.startswith(signature):
                return mimetype, extension
        if head.startswith(b"%PDF"):
            return "application/pdf", ".pdf"
        if head.startswith(b"RIFF") and head[8:12] in (b"WEBP", b"AVI "):
            return ("image/webp", ".webp") if head[8:12] == b"WEBP" else ("video/x-msvideo", ".avi")
        if head.startswith(b"ID3") or head.startswith(b"\xff\xfb"):
            return "audio/mpeg", ".mp3"
        if head.startswith(b"OggS"):
            return "audio/ogg", ".ogg"
        if head.startswith(b"fLaC"):
            return "audio/flac", ".flac"
        if head.startswith(b"\x1aE\xdf\xa3"):
            return "video/webm", ".webm"
        return None

    def _zip_variants(self, head: bytes, base: StreamInfo | None) -> list[StreamInfo]:
        """ZIP 容器族：docx / xlsx / pptx / epub / 普通 zip。"""
        filename = base.filename if base else None
        local_path = base.local_path if base else None
        try:
            stream = io.BytesIO(head)
            if hasattr(base, "local_path") and base and base.local_path:
                with open(base.local_path, "rb") as fh:
                    stream = io.BytesIO(fh.read(200_000))
            with zipfile.ZipFile(stream) as archive:
                names = archive.namelist()
                if "mimetype" in names:
                    try:
                        if archive.read("mimetype").strip() == b"application/epub+zip":
                            return [StreamInfo(mimetype="application/epub+zip", extension=".epub",
                                               filename=filename, local_path=local_path)]
                    except KeyError:
                        pass
                if any(n.startswith("word/") for n in names):
                    return [StreamInfo(mimetype="application/vnd.openxmlformats-officedocument.wordprocessingml.document",
                                       extension=".docx", filename=filename, local_path=local_path)]
                if any(n.startswith("xl/") for n in names):
                    return [StreamInfo(mimetype="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet",
                                       extension=".xlsx", filename=filename, local_path=local_path)]
                if any(n.startswith("ppt/") for n in names):
                    return [StreamInfo(mimetype="application/vnd.openxmlformats-officedocument.presentationml.presentation",
                                       extension=".pptx", filename=filename, local_path=local_path)]
                return [StreamInfo(mimetype="application/zip", extension=".zip",
                                   filename=filename, local_path=local_path)]
        except Exception:
            return [StreamInfo(mimetype="application/zip", extension=".zip",
                               filename=filename, local_path=local_path)]

    # -- 转换 ----------------------------------------------------------------
    def convert(self, source: Any, *, stream_info: StreamInfo | None = None, **kwargs: Any) -> DocumentConverterResult:
        """万能入口：路径 / URL / data URI / bytes / 流 / 帧 / 结构化对象。"""
        started = time.perf_counter()
        stream_info = _coerce_stream_info(stream_info)
        if _looks_like_observation(source):
            # 总线观测直接进引擎：unchanged / silence 返回 None（零 token）
            observed = self.convert_observation(source, **kwargs)
            if observed is None:
                return None  # type: ignore[return-value]
            observed.meta.setdefault(
                "elapsed_ms", round((time.perf_counter() - started) * 1000, 3)
            )
            self._account(observed)
            return observed
        if self._is_object(source):
            result = self._convert_object(source, stream_info, kwargs)
        else:
            stream, info, close = self._open(source, stream_info)
            try:
                result = self._convert_stream(stream, info, kwargs)
            finally:
                if close:
                    try:
                        stream.close()  # type: ignore[union-attr]
                    except Exception:
                        pass
        result.meta.setdefault("elapsed_ms", round((time.perf_counter() - started) * 1000, 3))
        self._account(result)
        return result

    #: 帧 / 结构化对象的捷入口
    def convert_frame(self, frame: Any, **kwargs: Any) -> DocumentConverterResult:
        return self.convert(frame, **kwargs)

    def convert_file(self, path: str | os.PathLike[str], **kwargs: Any) -> DocumentConverterResult:
        return self.convert(path, **kwargs)

    def convert_observation(self, observation: Any, **kwargs: Any) -> DocumentConverterResult | None:
        """接入统一总线 ``bus.Observation``：unchanged / silence 直接返回 None（零 token）。

        增量语义：``dirty_rects`` 进入稀疏编码的"脏区加权"，只把变化块排到前面，
        不需要大脑自己判断"哪里变了"。
        """
        kind = getattr(observation, "kind", "frame")
        kind_value = str(getattr(kind, "value", kind)).lower()
        if kind_value in ("unchanged", "silence", "none"):
            return None
        frame = getattr(observation, "frame", None)
        if frame is None and hasattr(observation, "pixels"):
            frame = observation
        if frame is None:
            return None
        options = dict(kwargs)
        if "dirty_rects" not in options:
            dirty = getattr(observation, "dirty_rects", None)
            if dirty:
                options["dirty_rects"] = tuple(dirty)
        if "change_ratio" not in options:
            ratio = getattr(observation, "changed_ratio", None)
            if ratio is not None:
                options["change_ratio"] = float(ratio)
        if "scene" not in options:
            scene = getattr(observation, "scene", None) or getattr(observation, "profile", None)
            if scene is not None:
                options["scene"] = scene
        options.setdefault(
            "frame_meta",
            {
                "seq": getattr(observation, "seq", None),
                "source": getattr(observation, "source", None),
                "is_key_frame": getattr(observation, "is_key_frame", None),
                "fovea": getattr(observation, "fovea", None),
                "reason": getattr(observation, "reason", ""),
            },
        )
        return self.convert(frame, **options)

    def zoom(self, frame: Any, rect: Any, **kwargs: Any) -> DocumentConverterResult:
        """指令式变焦：对原始帧的指定区域增强放大。"""
        for registration in self._registry:
            if isinstance(registration.converter, VisionConverter):
                pixels = getattr(frame, "pixels", frame)
                return registration.converter.convert_zoom(pixels, rect, **kwargs)
        raise UnsupportedContentError("未注册视觉转换器")

    def stream(self, frames: Iterable[Any], **kwargs: Any) -> Iterator[DocumentConverterResult | None]:
        """实时流式处理：逐帧产出结果（空帧/重复帧产出 None）。"""
        for frame in frames:
            yield self.convert(frame, **kwargs)

    # -- 内部：打开与分流 ----------------------------------------------------
    def _is_object(self, source: Any) -> bool:
        if isinstance(source, np.ndarray):
            return True
        if isinstance(source, (dict, list, tuple)):
            return True
        if hasattr(source, "pixels"):
            return True
        return hasattr(source, "convert") and hasattr(source, "size")

    def _convert_object(
        self, source: Any, stream_info: StreamInfo | None, kwargs: dict[str, Any]
    ) -> DocumentConverterResult:
        info = stream_info or StreamInfo(mimetype="application/x-optic-object")
        for registration in self._registry:
            try:
                if not registration.converter.accepts_object(source, info, **kwargs):
                    continue
                result = registration.converter.convert_object(source, info, **kwargs)
                return self._finalize(result, registration)
            except MissingDependencyException:
                raise
            except Exception as exc:
                self._record_error(registration, exc)
                continue
        raise UnsupportedContentError(f"没有转换器接受该对象：{type(source).__name__}")

    def _open(
        self, source: Any, stream_info: StreamInfo | None
    ) -> tuple[BinaryIO, StreamInfo, bool]:
        if isinstance(source, (str, os.PathLike)):
            location = os.fspath(source)
            if location.startswith("data:"):
                return self._open_data_uri(location, stream_info)
            if re.match(r"^https?://", location, re.I):
                return self._open_url(location, stream_info)
            if not os.path.exists(location):
                raise FileNotFoundError(f"文件不存在：{location}")
            info = stream_info or self._info_for_path(location)
            return open(location, "rb"), info, True
        if isinstance(source, (bytes, bytearray, memoryview)):
            return io.BytesIO(bytes(source)), stream_info or StreamInfo(), False
        if hasattr(source, "read"):
            info = stream_info or StreamInfo(filename=getattr(source, "name", None))
            return source, info, False  # type: ignore[return-value]
        raise UnsupportedContentError(f"不支持的输入类型：{type(source).__name__}")

    def _open_data_uri(
        self, uri: str, stream_info: StreamInfo | None
    ) -> tuple[BinaryIO, StreamInfo, bool]:
        match = re.match(r"^data:([^;,]+)?(;base64)?,(.*)$", uri, re.S)
        if not match:
            raise UnsupportedContentError("data URI 格式错误")
        mimetype = match.group(1) or "text/plain"
        payload = match.group(3)
        try:
            data = base64.b64decode(payload) if match.group(2) else payload.encode("utf-8")
        except (binascii.Error, ValueError) as exc:
            raise UnsupportedContentError(f"data URI 解码失败：{exc}") from exc
        extension = mimetypes.guess_extension(mimetype) or None
        info = stream_info or StreamInfo(mimetype=mimetype, extension=extension)
        return io.BytesIO(data), info, False

    def _open_url(
        self, url: str, stream_info: StreamInfo | None
    ) -> tuple[BinaryIO, StreamInfo, bool]:
        import urllib.parse
        import urllib.request

        request = urllib.request.Request(url, headers={"User-Agent": "OpticNerve/2.0"})
        with urllib.request.urlopen(request, timeout=20) as response:  # noqa: S310
            data = response.read()
            content_type = response.headers.get("Content-Type", "")
        mimetype = content_type.split(";")[0].strip() or "application/octet-stream"
        name = os.path.basename(urllib.parse.urlparse(url).path) or None  # type: ignore[name-defined]
        info = stream_info or StreamInfo(mimetype=mimetype, filename=name, url=url)
        return io.BytesIO(data), info, False

    def _info_for_path(self, path: str) -> StreamInfo:
        name = os.path.basename(path)
        mimetype, _ = mimetypes.guess_type(name)
        extension = os.path.splitext(name)[1].lower() or None
        return StreamInfo(
            mimetype=mimetype or "",
            extension=extension,
            filename=name,
            local_path=os.path.abspath(path),
        )

    def _convert_stream(
        self, stream: BinaryIO, info: StreamInfo, kwargs: dict[str, Any]
    ) -> DocumentConverterResult:
        candidates = self.sniff(stream, info)
        # 两轮：先所有候选 × 专用转换器，再所有候选 × 兜底转换器。
        # 兜底转换器"什么都能接"，若与专用转换器同轮会抢在嗅探结果之前。
        for generic in (False, True):
            for candidate in candidates:
                for registration in self._registry:
                    if bool(getattr(registration.converter, "generic", False)) != generic:
                        continue
                    position = self._safe_tell(stream)
                    try:
                        if not registration.converter.accepts(stream, candidate, **kwargs):
                            continue
                    except Exception as exc:
                        self._record_error(registration, exc)
                        continue
                    finally:
                        self._safe_seek(stream, position)
                    try:
                        result = registration.converter.convert(stream, candidate, **kwargs)
                        if result is not None:
                            return self._finalize(result, registration)
                    except MissingDependencyException:
                        raise
                    except UnsupportedContentError:
                        continue
                    except Exception as exc:
                        self._record_error(registration, exc)
                        continue
                    finally:
                        self._safe_seek(stream, position)
        raise UnsupportedContentError(
            f"没有转换器能处理该内容（{info.mimetype or '未知类型'}）"
        )

    def _safe_tell(self, stream: Any) -> int:
        try:
            return int(stream.tell())
        except Exception:
            return 0

    def _safe_seek(self, stream: Any, position: int) -> None:
        try:
            stream.seek(position)
        except Exception:
            pass

    def _record_error(self, registration: ConverterRegistration, exc: Exception) -> None:
        self.stats["errors"] += 1
        message = f"{registration.name}: {type(exc).__name__}: {exc}"
        self.last_errors.append(message)
        if len(self.last_errors) > 20:
            self.last_errors = self.last_errors[-20:]

    def _finalize(
        self, result: DocumentConverterResult, registration: ConverterRegistration
    ) -> DocumentConverterResult:
        result.converter = result.converter or registration.name
        if result.kind == ResultKind.MARKDOWN:
            result.markdown = _normalize_markdown(result.markdown or "")
        elif result.kind.is_visual:
            result.markdown = ""  # 画面绝不转文本
        if result.tokens is None:
            if result.kind.is_visual:
                frames = result.all_frames()
                result.tokens = TokenReport(
                    original=sum(f.tokens for f in frames) or 1,
                    output=sum(f.tokens for f in frames),
                    strategy=result.meta.get("strategy", "visual"),
                )
            else:
                text = result.markdown or ""
                tokens = estimate_text_tokens(text)
                result.tokens = TokenReport(original=tokens, output=tokens, strategy="text")
        return result

    def _account(self, result: DocumentConverterResult) -> None:
        self.stats["conversions"] += 1
        self.stats["by_kind"][result.kind.value] = self.stats["by_kind"].get(result.kind.value, 0) + 1
        if result.tokens:
            self.stats["tokens_in"] += result.tokens.original
            self.stats["tokens_out"] += result.tokens.output

    # -- 观测 ----------------------------------------------------------------
    def status(self) -> dict[str, Any]:
        return {
            "name": "optic_nerve",
            "version": "2.0",
            "converters": [
                {"name": r.name, "priority": r.priority, "class": type(r.converter).__name__}
                for r in self._registry
            ],
            "stats": json.loads(json.dumps(self.stats)),
            "options": {
                "max_tokens": self.options.max_tokens,
                "mode": self.options.mode,
                "target_ratio": self.options.target_ratio,
            },
            "errors": list(self.last_errors[-5:]),
        }


def _config_section(config: Any, name: str) -> dict[str, Any]:
    if config is None:
        return {}
    try:
        section = config.section(name)
    except Exception:
        return {}
    if not section:
        return {}
    try:
        return dict(section)
    except Exception:
        return {}


def _normalize_markdown(text: str) -> str:
    text = (text or "").replace("\r\n", "\n").replace("\r", "\n")
    text = re.sub(r"\n{3,}", "\n\n", text)
    return text.strip()


# ===========================================================================
# 14. 实时流（帧 → 视神经 → 对大脑最优的表示）
# ===========================================================================


class OpticNerveStream:
    """把采集层帧流变成转化流：去重、场景跟随、token 记账。

    >>> stream = OpticNerveStream()
    >>> for frame in retina.frames():
    ...     payload = stream.push(frame, scene=frame.scene)
    ...     if payload is not None:
    ...         brain.ask_images(payload.image_bytes(), question)
    """

    def __init__(
        self,
        engine: OpticNerve | None = None,
        *,
        dedupe: bool = True,
        max_frames: int | None = None,
        scene: Any = None,
        change_threshold: float = 0.0015,
    ) -> None:
        self.engine = engine or default_engine()
        self.dedupe = bool(dedupe)
        self.max_frames = max_frames
        self.scene = scene
        self.change_threshold = float(change_threshold)
        self._last_signature: str | None = None
        self.counters = {
            "pushed": 0,
            "emitted": 0,
            "deduped": 0,
            "tokens_in": 0,
            "tokens_out": 0,
            "errors": 0,
        }

    def push(
        self,
        frame: Any,
        *,
        change_ratio: float = 0.0,
        scene: Any = None,
        dirty_rects: Sequence[Any] = (),
        hint: str = "",
        force: bool = False,
    ) -> DocumentConverterResult | None:
        self.counters["pushed"] += 1
        if self.max_frames is not None and self.counters["emitted"] >= self.max_frames:
            return None
        pixels = getattr(frame, "pixels", frame)
        signature = _frame_signature(pixels)
        if (
            self.dedupe
            and not force
            and signature is not None
            and signature == self._last_signature
            and change_ratio <= self.change_threshold
        ):
            self.counters["deduped"] += 1
            return None
        self._last_signature = signature
        try:
            payload = self.engine.convert(
                frame,
                scene=scene if scene is not None else self.scene,
                change_ratio=change_ratio,
                dirty_rects=tuple(dirty_rects),
                hint=hint,
            )
        except Exception:
            self.counters["errors"] += 1
            raise
        self.counters["emitted"] += 1
        if payload.tokens:
            self.counters["tokens_in"] += payload.tokens.original
            self.counters["tokens_out"] += payload.tokens.output
        return payload

    def stats(self) -> dict[str, Any]:
        payload = dict(self.counters)
        saved = payload["tokens_in"] - payload["tokens_out"]
        payload["tokens_saved"] = saved
        payload["token_ratio"] = round(
            payload["tokens_out"] / payload["tokens_in"], 4
        ) if payload["tokens_in"] else 1.0
        return payload

    def reset(self) -> None:
        self._last_signature = None
        for key in self.counters:
            self.counters[key] = 0


def _frame_signature(pixels: Any) -> str | None:
    """低成本帧指纹（32×18 灰度）用于去重。"""
    try:
        bgr = _as_bgr(pixels)
    except Exception:
        return None
    thumb = _resize(bgr, 32, 18)
    return _sha8(np.ascontiguousarray(_luma(thumb).astype(np.uint8)).tobytes())


_DEFAULT_ENGINE: OpticNerve | None = None


def default_engine() -> OpticNerve:
    """进程级默认引擎（读 config/retina.yaml 的 ``optic_nerve`` 段）。"""
    global _DEFAULT_ENGINE
    if _DEFAULT_ENGINE is None:
        config = None
        try:
            from .config import load_config  # type: ignore

            config = load_config()
        except Exception:
            config = None
        _DEFAULT_ENGINE = OpticNerve(config=config)
    return _DEFAULT_ENGINE


def reset_default_engine() -> None:
    """测试用：丢弃默认引擎单例。"""
    global _DEFAULT_ENGINE
    _DEFAULT_ENGINE = None


