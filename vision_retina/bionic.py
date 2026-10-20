"""仿生双路视觉：全帧低清（视杆）+ 焦点高清（视锥）+ 指令式变焦 + token 预算。

设计要点（对应 V1.2–V1.4）：
  * 双路：第一次只发全帧低清，让模型看布局；需要细节时再发一张焦点高清。
  * 指令式变焦：模型输出 ``{"action":"zoom","region":{x,y,w,h}}``，视网膜按坐标裁剪、
    用语义 ROI 扩展避免切字，再按预算选择放大倍数。
  * 预算三条：按需（模型主动请求才发第二张）、裁剪到位（语义安全的最小矩形）、
    双路受控（低清 + 焦点 + 文本的估算 token 不超过上限，超了先缩小焦点图再拒绝）。

本模块只做图片编码与请求编排，不内置任何模型；大脑由调用方注入 VisionBrain。
"""
from __future__ import annotations

import io
import json
import math
import time
from dataclasses import dataclass, field
from typing import Any

import numpy as np
from PIL import Image

from .brain import BrainReply, VisionBrain
from .config import Config, load_config
from .frames import Rect, RetinaFrame
from .roi import expand_rect_to_content

ZOOM_ACTION = "zoom"

ZOOM_PROTOCOL = (
    "\n\n【双路视觉协议】你现在看到的是一张低清全帧，只用于判断版面与位置。"
    "如果回答需要读小字/数字/细节，而低清图不足以确定，必须先请求放大，并且只输出一个 JSON：\n"
    '{"action":"zoom","region":{"x":<整数>,"y":<整数>,"w":<整数>,"h":<整数>},"reason":"<一句话>"}\n'
    "坐标系：你看到的低清图左上角为 (0,0)，图片尺寸 {width}x{height} 像素；region 必须紧贴要读的内容，不要框整屏。"
    "若低清图已经足够，直接给出最终答案；不要输出其它 JSON，也不要在请求放大时附带答案。"
    "注意：放大本身消耗 token 预算，低清图能读出答案时禁止请求放大，不要为保险而放大。"
)


def estimate_image_tokens(width: int, height: int) -> int:
    """按实测标定的图片 token 估算（±20%）。

    标定样本（deepseek-flash）：317²→184、800×600→346、2240×1400→962，
    近似 ``570 * sqrt(megapixels)``。仅用于预算预判，最终以 usage 为准。
    """
    if width <= 0 or height <= 0:
        return 0
    megapixels = (width * height) / 1_000_000.0
    return max(48, int(round(570.0 * math.sqrt(megapixels))))


def parse_zoom(text: str) -> "ZoomRequest | None":
    """从模型输出中提取 zoom 指令；兼容代码块与嵌套 JSON。"""
    if not text:
        return None
    decoder = json.JSONDecoder()
    for index, char in enumerate(text):
        if char != "{":
            continue
        try:
            obj, _ = decoder.raw_decode(text[index:])
        except ValueError:
            continue
        if not isinstance(obj, dict):
            continue
        action = str(obj.get("action") or obj.get("type") or "").lower()
        payload = obj.get("region") or obj.get("target") or obj.get("zoom")
        if action not in (ZOOM_ACTION, "magnify", "enlarge") and not (payload and "zoom" in obj):
            continue
        if isinstance(payload, dict):
            try:
                x = int(payload.get("x", 0))
                y = int(payload.get("y", 0))
                w = int(payload.get("w", payload.get("width", 0)))
                h = int(payload.get("h", payload.get("height", 0)))
            except (TypeError, ValueError):
                continue
            if w > 0 and h > 0:
                return ZoomRequest(
                    rect=Rect(x, y, w, h),
                    reason=str(obj.get("reason") or "")[:200],
                    raw=obj,
                )
        elif isinstance(payload, (list, tuple)) and len(payload) == 4:
            try:
                x, y, w, h = (int(v) for v in payload)
            except (TypeError, ValueError):
                continue
            if w > 0 and h > 0:
                return ZoomRequest(rect=Rect(x, y, w, h), reason=str(obj.get("reason") or "")[:200], raw=obj)
    return None


@dataclass(frozen=True)
class ZoomRequest:
    rect: Rect
    reason: str = ""
    raw: dict[str, Any] = field(default_factory=dict)


@dataclass
class EncodedImage:
    data: bytes
    width: int
    height: int
    quality: int
    rect: Rect | None = None
    mime: str = "image/jpeg"

    @property
    def tokens(self) -> int:
        return estimate_image_tokens(self.width, self.height)

    def to_dict(self) -> dict[str, Any]:
        return {
            "width": self.width,
            "height": self.height,
            "quality": self.quality,
            "bytes": len(self.data),
            "estimated_tokens": self.tokens,
            "rect": list(self.rect) if self.rect else None,
        }


@dataclass
class BionicConfig:
    low_width: int = 768
    low_quality: int = 72
    focus_max_width: int = 1024
    focus_quality: int = 88
    max_scale: float = 2.0
    budget_tokens: int = 1200
    text_reserve: int = 120
    max_area_ratio: float = 0.45
    zoom_margin: int = 6
    request_max_tokens: int = 384
    answer_max_tokens: int = 2048

    @classmethod
    def from_config(cls, config: Config | None) -> "BionicConfig":
        cfg = config or load_config()
        data = cfg.section("bus").get("fovea") or {}
        base = cls()
        return cls(
            low_width=int(data.get("low_width", base.low_width)),
            low_quality=int(data.get("low_quality", base.low_quality)),
            focus_max_width=int(data.get("focus_max_width", base.focus_max_width)),
            focus_quality=int(data.get("focus_quality", base.focus_quality)),
            max_scale=float(data.get("max_scale", base.max_scale)),
            budget_tokens=int(data.get("budget_tokens", base.budget_tokens)),
            text_reserve=int(data.get("text_reserve", base.text_reserve)),
            max_area_ratio=float(data.get("max_area_ratio", base.max_area_ratio)),
            zoom_margin=int(data.get("zoom_margin", base.zoom_margin)),
            request_max_tokens=int(data.get("request_max_tokens", base.request_max_tokens)),
            answer_max_tokens=int(data.get("answer_max_tokens", base.answer_max_tokens)),
        )

    def to_dict(self) -> dict[str, Any]:
        return {
            "low_width": self.low_width,
            "low_quality": self.low_quality,
            "focus_max_width": self.focus_max_width,
            "focus_quality": self.focus_quality,
            "max_scale": self.max_scale,
            "budget_tokens": self.budget_tokens,
            "max_area_ratio": self.max_area_ratio,
        }


@dataclass
class BionicResult:
    answer: str
    prompt: str
    low: EncodedImage
    focus: EncodedImage | None
    zoom: Rect | None
    zoom_reason: str
    calls: int
    prompt_tokens: int
    completion_tokens: int
    estimated_tokens: int
    budget_tokens: int
    budget_ok: bool
    denied_reason: str | None
    first_answer: str
    latency_ms: int
    model: str
    usage: dict[str, Any] = field(default_factory=dict)
    reply: BrainReply | None = None
    crop_safe: bool = True

    def to_dict(self) -> dict[str, Any]:
        return {
            "answer": self.answer,
            "calls": self.calls,
            "zoom": list(self.zoom) if self.zoom else None,
            "zoom_reason": self.zoom_reason,
            "denied_reason": self.denied_reason,
            "low": self.low.to_dict(),
            "focus": self.focus.to_dict() if self.focus else None,
            "prompt_tokens": self.prompt_tokens,
            "completion_tokens": self.completion_tokens,
            "estimated_tokens": self.estimated_tokens,
            "budget_tokens": self.budget_tokens,
            "budget_ok": self.budget_ok,
            "crop_safe": self.crop_safe,
            "latency_ms": self.latency_ms,
            "model": self.model,
        }


def _encode_pixels(pixels: np.ndarray, quality: int, max_width: int) -> EncodedImage:
    height, width = pixels.shape[:2]
    image = Image.fromarray(np.ascontiguousarray(pixels))
    if max_width and width > max_width:
        new_h = max(1, int(round(height * max_width / width)))
        image = image.resize((max_width, new_h), Image.LANCZOS)
    buffer = io.BytesIO()
    image.save(buffer, format="JPEG", quality=int(quality), optimize=True)
    return EncodedImage(
        data=buffer.getvalue(),
        width=image.width,
        height=image.height,
        quality=int(quality),
    )


def _encode_focus(pixels: np.ndarray, rect: Rect, target_width: int, quality: int) -> EncodedImage:
    crop = pixels[rect.y:rect.y + rect.h, rect.x:rect.x + rect.w]
    image = Image.fromarray(np.ascontiguousarray(crop))
    if target_width and target_width > image.width:
        new_h = max(1, int(round(image.height * target_width / image.width)))
        image = image.resize((target_width, new_h), Image.LANCZOS)
    elif target_width and target_width < image.width:
        new_h = max(1, int(round(image.height * target_width / image.width)))
        image = image.resize((target_width, new_h), Image.LANCZOS)
    buffer = io.BytesIO()
    image.save(buffer, format="JPEG", quality=int(quality), optimize=True)
    return EncodedImage(
        data=buffer.getvalue(),
        width=image.width,
        height=image.height,
        quality=int(quality),
        rect=rect,
    )


def _clean_answer(text: str) -> str:
    """去掉模型偶尔附带的 zoom JSON，只留自然语言答案。"""
    if not text:
        return ""
    stripped = text.strip()
    if stripped.startswith("```"):
        lines = stripped.splitlines()
        if len(lines) >= 3:
            stripped = "\n".join(lines[1:-1]).strip()
    return stripped


class BionicRetina:
    """把一帧编排成「低清全帧 → 按需焦点高清」的双路请求。"""

    def __init__(self, brain: VisionBrain, config: Config | None = None,
                 bionic_config: BionicConfig | None = None):
        self.brain = brain
        self.config = config or load_config()
        self.settings = bionic_config or BionicConfig.from_config(self.config)

    # ---- 编码 ----

    def low_view(self, frame: RetinaFrame) -> EncodedImage:
        return _encode_pixels(frame.pixels, self.settings.low_quality, self.settings.low_width)

    def focus_view(self, frame: RetinaFrame, rect: Rect, target_width: int) -> EncodedImage:
        return _encode_focus(frame.pixels, rect, target_width, self.settings.focus_quality)

    # ---- 主流程 ----

    def ask(
        self,
        prompt: str,
        frame: RetinaFrame,
        *,
        budget_tokens: int | None = None,
        temperature: float = 0.0,
    ) -> BionicResult:
        started = time.perf_counter()
        settings = self.settings
        budget = int(budget_tokens or settings.budget_tokens)
        low = self.low_view(frame)

        first_prompt = prompt + ZOOM_PROTOCOL.replace("{width}", str(low.width)).replace("{height}", str(low.height))
        first = self.brain.ask_images(
            [low.data], first_prompt,
            temperature=temperature,
            max_tokens=settings.request_max_tokens,
        )
        first_prompt_tokens = int((first.usage or {}).get("prompt_tokens") or 0)
        first_completion = int((first.usage or {}).get("completion_tokens") or 0)
        text_tokens = max(settings.text_reserve, first_prompt_tokens - low.tokens) if first_prompt_tokens else settings.text_reserve

        zoom = parse_zoom(first.text)
        if zoom is None:
            answer = _clean_answer(first.text)
            return BionicResult(
                answer=answer, prompt=prompt, low=low, focus=None, zoom=None, zoom_reason="",
                calls=1, prompt_tokens=first_prompt_tokens, completion_tokens=first_completion,
                estimated_tokens=low.tokens + text_tokens, budget_tokens=budget,
                budget_ok=(low.tokens + text_tokens) <= budget, denied_reason=None,
                first_answer=answer, latency_ms=int((time.perf_counter() - started) * 1000),
                model=first.model, usage=first.usage or {}, reply=first,
            )

        # 模型看到的是低清图，坐标按低清图计；换算回原生帧像素。
        sx = frame.width / max(1, low.width)
        sy = frame.height / max(1, low.height)
        native = Rect(
            int(round(zoom.rect.x * sx)), int(round(zoom.rect.y * sy)),
            max(1, int(round(zoom.rect.w * sx))), max(1, int(round(zoom.rect.h * sy))),
        )
        clipped = native.clip(frame.width, frame.height)
        # 1) 先做语义扩展；CAD 表格的长线会让扩展失控，因此加面积上限。
        safe_rect, safe = expand_rect_to_content(
            frame.pixels, clipped, margin=settings.zoom_margin,
            max_area_ratio=min(0.95, settings.max_area_ratio * 3),
        )
        crop_safe = safe
        area_cap = frame.width * frame.height * settings.max_area_ratio
        if not safe or safe_rect.area <= 0 or safe_rect.area > area_cap:
            # 2) 扩展失败时退回“请求框 + 边距”，保证仍能放大；标记 crop_safe=false 供审计。
            safe_rect = clipped.expand(settings.zoom_margin * 2).clip(frame.width, frame.height)
            crop_safe = False
        if safe_rect.area <= 0:
            return self._denied(first, low, prompt, zoom, budget, text_tokens, started,
                                "请求区域为空，已退回低清答案。")
        if safe_rect.area > area_cap:
            return self._denied(first, low, prompt, zoom, budget, text_tokens, started,
                                "请求区域过大（超过画面 45%），不符合“只放大必要小区域”。")

        focus, denied = self._fit_focus(frame, safe_rect, budget, text_tokens, low.tokens)
        if focus is None:
            return self._denied(first, low, prompt, zoom, budget, text_tokens, started, denied or "预算不足。")

        second_prompt = (
            f"问题：{prompt}\n\n"
            f"第二张图是你请求放大的区域（全帧坐标 x={safe_rect.x}, y={safe_rect.y}, "
            f"w={safe_rect.w}, h={safe_rect.h}）。请结合第一张低清全帧和第二张放大图直接回答；"
            "不要输出 JSON，不要解释。"
        )
        second = self.brain.ask_images(
            [low.data, focus.data], second_prompt,
            temperature=temperature,
            max_tokens=settings.answer_max_tokens,
        )
        prompt_tokens = first_prompt_tokens + int((second.usage or {}).get("prompt_tokens") or 0)
        completion_tokens = first_completion + int((second.usage or {}).get("completion_tokens") or 0)
        return BionicResult(
            answer=_clean_answer(second.text), prompt=prompt, low=low, focus=focus,
            zoom=safe_rect, zoom_reason=zoom.reason, calls=2, crop_safe=crop_safe,
            prompt_tokens=prompt_tokens, completion_tokens=completion_tokens,
            estimated_tokens=low.tokens + focus.tokens + text_tokens,
            budget_tokens=budget, budget_ok=prompt_tokens <= budget if prompt_tokens else True,
            denied_reason=None, first_answer=_clean_answer(first.text),
            latency_ms=int((time.perf_counter() - started) * 1000),
            model=second.model, usage=second.usage or {}, reply=second,
        )

    def _fit_focus(
        self,
        frame: RetinaFrame,
        rect: Rect,
        budget: int,
        text_tokens: int,
        low_tokens: int,
    ) -> tuple[EncodedImage | None, str | None]:
        settings = self.settings
        remaining = budget - text_tokens - low_tokens
        if remaining < 120:
            return None, f"预算不足：低清图与文本已占 {text_tokens + low_tokens} tokens，上限 {budget}。"
        natural = int(rect.w * settings.max_scale)
        start = max(rect.w, min(settings.focus_max_width, natural))
        width = start
        while width >= max(320, rect.w // 2):
            focus = _encode_focus(frame.pixels, rect, width, settings.focus_quality)
            if focus.tokens <= remaining:
                return focus, None
            width = int(width * 0.82)
        return None, "预算不足：即使缩到最小焦点图仍超出上限。"

    def _denied(
        self, first: BrainReply, low: EncodedImage, prompt: str, zoom: ZoomRequest,
        budget: int, text_tokens: int, started: float, reason: str,
    ) -> BionicResult:
        answer = _clean_answer(first.text)
        return BionicResult(
            answer=answer, prompt=prompt, low=low, focus=None, zoom=zoom.rect,
            zoom_reason=zoom.reason, calls=1,
            prompt_tokens=int((first.usage or {}).get("prompt_tokens") or 0),
            completion_tokens=int((first.usage or {}).get("completion_tokens") or 0),
            estimated_tokens=low.tokens + text_tokens, budget_tokens=budget,
            budget_ok=False, denied_reason=reason, first_answer=answer,
            latency_ms=int((time.perf_counter() - started) * 1000),
            model=first.model, usage=first.usage or {}, reply=first,
        )


__all__ = [
    "BionicRetina", "BionicConfig", "BionicResult", "EncodedImage",
    "ZoomRequest", "parse_zoom", "estimate_image_tokens", "ZOOM_PROTOCOL",
]
