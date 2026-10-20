"""大脑 API 接入口：OpenAI 兼容 /chat/completions（任务书 §3.3）。

边界（必须坚持）：
- 本模块**不内置任何模型**，只负责把 RetinaFrame 组装成标准 image_url 内容块，
  调用客户在 config 里填的 base_url / model / key。
- 任何家（DeepSeek / GPT / Qwen / GLM / 本地 vLLM ...）只要兼容 OpenAI Chat
  Completions 就能接；格式差异（data URI / 裸 base64）由这里统一抹平。

健康检查不是"能连上"，而是"确实能看见"：
发一张合成图（纯色圆 + 唯一颜色），要求只回答颜色；文本模型无法通过。
"""
from __future__ import annotations

import base64
import io
import json
import re
import time
from dataclasses import dataclass, field
from typing import Any, Iterable, Sequence

import httpx

from .config import Config, load_config
from .frames import RetinaFrame

ALLOWED_ACTIONS = {
    "click",
    "double_click",
    "right_click",
    "type",
    "key",
    "scroll",
    "drag",
    "wait",
    "observe",
}


class BrainError(RuntimeError):
    """大脑接入口的基础异常。"""


class BrainNotConfigured(BrainError):
    """base_url / model / api_key 缺任意一项。"""


class BrainAuthError(BrainError):
    """401/403：客户填的 key 无效或没有该模型权限。"""


class BrainHTTPError(BrainError):
    """上游返回非 2xx。"""

    def __init__(self, message: str, status_code: int | None = None, body: str = ""):
        super().__init__(message)
        self.status_code = status_code
        self.body = body


@dataclass
class BrainReply:
    """一次大脑调用的结果（给手/CLI/报告共用）。"""

    text: str
    model: str = ""
    latency_ms: int = 0
    usage: dict[str, Any] = field(default_factory=dict)
    finish_reason: str | None = None
    request_id: str | None = None
    image_bytes: int = 0
    raw: dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        return {
            "text": self.text,
            "model": self.model,
            "latency_ms": self.latency_ms,
            "usage": self.usage,
            "finish_reason": self.finish_reason,
            "request_id": self.request_id,
            "image_bytes": self.image_bytes,
        }

    @property
    def action(self) -> dict[str, Any] | None:
        """若模型按约定返回了动作 JSON，则解析出来（否则 None）。"""
        return parse_action(self.text)


def parse_action(text: str) -> dict[str, Any] | None:
    """从模型回复中提取第一个合法的动作 JSON（任务书 §4.2）。

    容忍 ```json 代码块、前后解释文字；只认 action 字段合法的对象。
    用 raw_decode 从每个 '{' 起尝试，正确处理嵌套对象（正则的非贪婪匹配会截断）。
    """
    if not text:
        return None
    candidates: list[str] = []
    for block in re.findall(r"```(?:json)?\s*(\{.*?\})\s*```", text, flags=re.S | re.I):
        candidates.append(block)
    candidates.append(text)
    decoder = json.JSONDecoder()
    for candidate in candidates:
        for index, char in enumerate(candidate):
            if char != "{":
                continue
            try:
                obj, _end = decoder.raw_decode(candidate[index:])
            except (json.JSONDecodeError, TypeError, ValueError):
                continue
            if isinstance(obj, dict) and obj.get("action") in ALLOWED_ACTIONS:
                return obj
    return None


def _tiny_square_png(color: tuple[int, int, int], size: int = 64) -> bytes:
    """生成纯色方块 PNG（健康检查用，不依赖 PIL 之外的库）。"""
    from PIL import Image

    image = Image.new("RGB", (size, size), color)
    buffer = io.BytesIO()
    image.save(buffer, format="PNG")
    return buffer.getvalue()


class VisionBrain:
    """OpenAI 兼容视觉模型接入口。"""

    def __init__(
        self,
        config: Config | None = None,
        *,
        base_url: str | None = None,
        api_key: str | None = None,
        model: str | None = None,
        http_client: httpx.Client | None = None,
    ):
        self.config = config or load_config()
        brain = self.config.section("brain")
        self.base_url = (base_url or brain.get("base_url") or "").strip()
        self.model = (model or brain.get("model") or "").strip()
        self._api_key = api_key if api_key is not None else self.config.api_key()
        self.temperature = float(brain.get("temperature", 0.2))
        self.timeout_s = float(brain.get("timeout_s", 60))
        self.max_tokens = int(brain.get("max_tokens", 1024))
        self.image_format = str(brain.get("image_format", "auto")).lower()
        self.system_prompt = brain.get("system_prompt") or None
        self.retries = int(brain.get("retries", 2))
        self.encode_format = str(self.config.get("encode.format", "jpeg"))
        self.encode_quality = int(self.config.get("encode.quality", 80))
        self.encode_max_width = int(self.config.get("encode.max_width", 1280))
        self._client = http_client

    # ---- 配置与状态 ----

    @property
    def endpoint(self) -> str:
        base = self.base_url.rstrip("/")
        if base.endswith("/chat/completions"):
            return base
        return f"{base}/chat/completions"

    def missing(self) -> list[str]:
        missing = []
        if not self.base_url:
            missing.append("brain.base_url")
        if not self.model:
            missing.append("brain.model")
        if not self._api_key:
            missing.append("brain.api_key（或环境变量 / config/secrets.local.env）")
        return missing

    def status(self) -> dict[str, Any]:
        return {
            "base_url": self.base_url,
            "model": self.model,
            "endpoint": self.endpoint if self.base_url else None,
            "api_key_configured": bool(self._api_key),
            "secrets": self.config.secrets_status(),
            "configured": not self.missing(),
            "image_format": self.image_format,
        }

    # ---- 图像内容块 ----

    def _image_url(self, frame: RetinaFrame | bytes) -> str:
        if isinstance(frame, RetinaFrame):
            payload = frame.encode(
                fmt=self.encode_format,
                quality=self.encode_quality,
                max_width=self.encode_max_width,
            )
            mime = "png" if self.encode_format.lower() == "png" else "jpeg"
        else:
            payload = frame
            mime = _sniff_mime(payload)
        if self.image_format == "base64":
            return base64.b64encode(payload).decode("ascii")
        return f"data:image/{mime};base64,{base64.b64encode(payload).decode('ascii')}"

    def image_part(self, frame: RetinaFrame | bytes, detail: str = "auto") -> dict[str, Any]:
        part: dict[str, Any] = {"type": "image_url", "image_url": {"url": self._image_url(frame)}}
        if detail and detail != "auto":
            part["image_url"]["detail"] = detail
        return part

    # ---- HTTP ----

    def _ensure_configured(self) -> None:
        missing = self.missing()
        if missing:
            raise BrainNotConfigured(
                "大脑接口未配置完整，缺少：" + "、".join(missing) + "。"
                "请在 config/retina.yaml 填 base_url/model，并把 key 写入 "
                "config/secrets.local.env（VISION_API_KEY=...）或同名环境变量。"
            )

    def _headers(self) -> dict[str, str]:
        return {
            "Authorization": f"Bearer {self._api_key}",
            "Content-Type": "application/json",
        }

    def _post(self, payload: dict[str, Any]) -> tuple[dict[str, Any], httpx.Response, int]:
        client = self._client or httpx.Client(timeout=self.timeout_s)
        owns = self._client is None
        started = time.perf_counter()
        last_error: Exception | None = None
        try:
            for attempt in range(self.retries + 1):
                try:
                    response = client.post(self.endpoint, headers=self._headers(), json=payload)
                except httpx.HTTPError as exc:
                    last_error = exc
                    if attempt >= self.retries:
                        raise BrainHTTPError(f"请求大脑接口失败：{exc}") from exc
                    time.sleep(min(2.0, 0.4 * (2**attempt)))
                    continue
                latency = int((time.perf_counter() - started) * 1000)
                if response.status_code in (401, 403):
                    raise BrainAuthError(
                        f"大脑接口鉴权失败（HTTP {response.status_code}）。请检查 key 是否有效、"
                        f"是否有模型 {self.model} 的权限。"
                    )
                if response.status_code == 429 or response.status_code >= 500:
                    last_error = BrainHTTPError(
                        f"大脑接口暂时不可用（HTTP {response.status_code}）",
                        status_code=response.status_code,
                        body=response.text[:500],
                    )
                    if attempt >= self.retries:
                        raise last_error
                    retry_after = response.headers.get("retry-after")
                    delay = float(retry_after) if (retry_after or "").isdigit() else min(4.0, 0.6 * (2**attempt))
                    time.sleep(delay)
                    continue
                if response.status_code >= 400:
                    raise BrainHTTPError(
                        f"大脑接口返回 HTTP {response.status_code}：{response.text[:800]}",
                        status_code=response.status_code,
                        body=response.text[:2000],
                    )
                try:
                    data = response.json()
                except ValueError as exc:
                    raise BrainHTTPError(f"大脑接口返回非 JSON：{response.text[:300]}") from exc
                return data, response, latency
            raise last_error or BrainHTTPError("大脑接口调用失败")
        finally:
            if owns:
                client.close()

    def chat(
        self,
        messages: Sequence[dict[str, Any]],
        *,
        temperature: float | None = None,
        max_tokens: int | None = None,
        extra: dict[str, Any] | None = None,
    ) -> BrainReply:
        self._ensure_configured()
        payload: dict[str, Any] = {
            "model": self.model,
            "messages": list(messages),
            "temperature": self.temperature if temperature is None else temperature,
            "max_tokens": self.max_tokens if max_tokens is None else max_tokens,
            "stream": False,
        }
        if extra:
            payload.update(extra)
        data, response, latency = self._post(payload)
        choices = data.get("choices") or []
        if not choices:
            raise BrainHTTPError(f"大脑接口响应没有 choices：{json.dumps(data, ensure_ascii=False)[:500]}")
        message = choices[0].get("message") or {}
        text = message.get("content") or ""
        if isinstance(text, list):  # 少数实现返回分段内容
            text = "".join(part.get("text", "") if isinstance(part, dict) else str(part) for part in text)
        return BrainReply(
            text=str(text).strip(),
            model=str(data.get("model") or self.model),
            latency_ms=latency,
            usage=data.get("usage") or {},
            finish_reason=choices[0].get("finish_reason"),
            request_id=response.headers.get("x-request-id") or data.get("id"),
            raw=data,
        )

    # ---- 业务方法 ----

    def ask(
        self,
        frame: RetinaFrame | bytes,
        prompt: str,
        *,
        system: str | None = None,
        detail: str = "auto",
        temperature: float | None = None,
        max_tokens: int | None = None,
    ) -> BrainReply:
        """看一张图并回答。frame 可以是 RetinaFrame，也可以是现成图片字节。"""
        system_text = system if system is not None else self.system_prompt
        messages: list[dict[str, Any]] = []
        if system_text:
            messages.append({"role": "system", "content": system_text})
        messages.append(
            {
                "role": "user",
                "content": [{"type": "text", "text": prompt}, self.image_part(frame, detail=detail)],
            }
        )
        reply = self.chat(messages, temperature=temperature, max_tokens=max_tokens)
        reply.image_bytes = len(_frame_payload(frame))
        return reply

    def ask_images(
        self,
        images: Sequence[RetinaFrame | bytes],
        prompt: str,
        *,
        system: str | None = None,
        detail: str = "auto",
        temperature: float | None = None,
        max_tokens: int | None = None,
    ) -> BrainReply:
        """一张消息里同时喂多张图（仿生双路：低清全帧 + 焦点高清）。"""
        frames = list(images)
        if not frames:
            raise ValueError("ask_images 至少需要一张图片")
        system_text = system if system is not None else self.system_prompt
        messages: list[dict[str, Any]] = []
        if system_text:
            messages.append({"role": "system", "content": system_text})
        content: list[dict[str, Any]] = [{"type": "text", "text": prompt}]
        total_bytes = 0
        for image in frames:
            payload = _frame_payload(image)
            total_bytes += len(payload)
            content.append(self.image_part(image, detail=detail))
        messages.append({"role": "user", "content": content})
        reply = self.chat(messages, temperature=temperature, max_tokens=max_tokens)
        reply.image_bytes = total_bytes
        return reply

    def ask_actions(
        self,
        frame: RetinaFrame,
        task: str,
        *,
        history: Iterable[str] | None = None,
        temperature: float | None = None,
        max_tokens: int | None = None,
    ) -> BrainReply:
        """让模型看当前画面并返回一个标准动作 JSON（手可直接消费）。"""
        instruction = (
            f"当前任务：{task}\n"
            "请只输出一个 JSON 对象，不要解释、不要 Markdown。格式：\n"
            '{"action":"click|double_click|right_click|type|key|scroll|drag|wait|observe",'
            '"target":{"x":640,"y":420},"params":{},"meta":{}}\n'
            f"坐标使用屏幕像素坐标，本帧左上角为 (0,0)，帧尺寸 {frame.width}x{frame.height}；"
            '若任务已完成请返回 {"action":"observe"}。'
        )
        if history:
            instruction += "\n最近动作：\n" + "\n".join(history)
        return self.ask(frame, instruction, temperature=temperature, max_tokens=max_tokens)

    def health_check(self, *, timeout_s: float | None = None) -> dict[str, Any]:
        """真实探测：模型是否能看见图片。成功返回 ok=True 与证据。

        用两张不同颜色的合成图分别提问（只回答颜色词）。文本模型可能靠猜命中
        一张，但两张都命中纯属偶然；同时避免推理模型把 max_tokens 花在思考上
        导致 content 为空（给足 512 token）。
        """
        status = self.status()
        missing = self.missing()
        if missing:
            return {
                "ok": False,
                "stage": "config",
                "missing": missing,
                "status": status,
                "detail": "配置不完整，尚未发起网络请求。",
            }
        probes = [
            ("红色", (220, 20, 30), ("红", "red", "crimson", "scarlet")),
            ("蓝色", (20, 60, 220), ("蓝", "blue", "navy", "azure")),
        ]
        question = "这张图片是什么颜色？只回答一个颜色词。"
        results: list[dict[str, Any]] = []
        started = time.perf_counter()
        for expected, rgb, words in probes:
            try:
                reply = self.ask(_tiny_square_png(rgb, 64), question, detail="low", max_tokens=512)
            except BrainAuthError as exc:
                return {"ok": False, "stage": "auth", "status": status, "detail": str(exc)}
            except BrainHTTPError as exc:
                detail = str(exc)
                vision_hint = ""
                if exc.status_code in (400, 404, 422) and re.search(r"image|vision|multimodal|content", detail, re.I):
                    vision_hint = "（上游错误信息与 image/vision 相关：该模型可能不支持视觉输入）"
                return {
                    "ok": False,
                    "stage": "http",
                    "status": status,
                    "detail": detail + vision_hint,
                    "latency_ms": int((time.perf_counter() - started) * 1000),
                }
            answer = reply.text.strip()
            ok = any(word in answer.lower() for word in words)
            results.append(
                {
                    "expected": expected,
                    "answer": answer,
                    "ok": ok,
                    "finish_reason": reply.finish_reason,
                    "latency_ms": reply.latency_ms,
                    "reasoning_tokens": (reply.usage.get("completion_tokens_details") or {}).get("reasoning_tokens"),
                }
            )
            if not ok:
                break
        passed = bool(results) and all(item["ok"] for item in results) and len(results) == len(probes)
        detail = (
            "模型正确识别了两张不同颜色的合成图，视觉输入可用。"
            if passed
            else "模型未能稳定识别合成色块：可能不支持图片、忽略了 image_url，或推理占满 token 未给出最终回答。"
        )
        if not passed and results and all(item["finish_reason"] == "length" for item in results):
            detail += " 本次 finish_reason=length：请调大 brain.max_tokens（推理模型需要更多预算）。"
        return {
            "ok": passed,
            "stage": "vision_probe" if passed else "vision_probe_failed",
            "status": status,
            "question": question,
            "results": results,
            "model": self.model,
            "latency_ms": int((time.perf_counter() - started) * 1000),
            "request_id": None,
            "detail": detail,
        }


def _sniff_mime(payload: bytes) -> str:
    if payload.startswith(b"\x89PNG\r\n\x1a\n"):
        return "png"
    if payload.startswith(b"\xff\xd8\xff"):
        return "jpeg"
    if payload[:4] == b"RIFF" and payload[8:12] == b"WEBP":
        return "webp"
    return "jpeg"


def _frame_payload(frame: RetinaFrame | bytes) -> bytes:
    if isinstance(frame, RetinaFrame):
        return frame.encode()
    return frame
