"""视网膜服务：把采集总线 + 大脑接入口组装成一个可被 MCP / WebSocket 共享的实例。

设计要点：
- 采集线程常驻（懒启动），多次 observe 之间保留差分基线与静默计时；
- 所有公开方法加锁，MCP 工具线程与 WebSocket 协程可以安全并发调用；
- 对外只返回标准结构（Observation 元数据 + JPEG 字节 + BrainReply 字段），
  不暴露任何具体"手"的实现，也不反向依赖它们。
"""
from __future__ import annotations

import base64
import hashlib
import threading
import time
from dataclasses import dataclass, field
from typing import Any

from .bionic import BionicConfig, BionicRetina
from .brain import BrainReply, VisionBrain
from .bus import Observation, RetinaBus
from .config import Config, load_config
from .frames import Rect, RetinaFrame


def _frame_signature(frame: RetinaFrame, quality: int | None = None,
                     max_width: int | None = None) -> bytes:
    """低成本帧签名：降采样像素 + 尺寸 + 编码参数，用于缓存命中判断。"""
    digest = hashlib.blake2b(digest_size=16)
    digest.update(f"{frame.width}x{frame.height}:{frame.source}:{quality}:{max_width}".encode("ascii"))
    digest.update(frame.pixels[::24, ::24].tobytes())
    return digest.digest()


@dataclass
class ObserveResult:
    """一次观察的完整结果。"""

    observation: Observation
    image: bytes | None = None
    mime: str = "image/jpeg"
    cached_image: bool = False

    @property
    def has_image(self) -> bool:
        return self.image is not None

    def meta(self) -> dict[str, Any]:
        data = self.observation.to_dict(include_pixels=False)
        data["image"] = {
            "included": self.has_image,
            "mime": self.mime,
            "bytes": len(self.image or b""),
            "cached": self.cached_image,
        }
        return data

    def to_dict(self, include_image_base64: bool = False) -> dict[str, Any]:
        data = self.meta()
        if include_image_base64 and self.image:
            data["image"]["data_base64"] = base64.b64encode(self.image).decode("ascii")
        return data


@dataclass
class AskResult:
    """一次"让大脑看画面并回答"的结果。"""

    reply: BrainReply
    observation: dict[str, Any]
    frame_id: int | None = None
    cached_frame: bool = False

    def to_dict(self) -> dict[str, Any]:
        data = self.reply.to_dict()
        data["observation"] = self.observation
        data["frame_id"] = self.frame_id
        data["cached_frame"] = self.cached_frame
        data["action"] = self.reply.action
        return data


class RetinaService:
    """采集 + 大脑的门面（MCP tools / WebSocket 共用）。"""

    def __init__(self, config: Config | None = None, *, brain: VisionBrain | None = None):
        self.config = config or load_config()
        self._brain = brain
        self._bus: RetinaBus | None = None
        self._lock = threading.RLock()
        self._started_at = 0.0
        self._last_frame: RetinaFrame | None = None
        self._last_obs: ObserveResult | None = None
        self._region: tuple[int, int, int, int] | None = None
        self._encode_cache: tuple[bytes, bytes, str] | None = None
        self._bionic: BionicRetina | None = None
        self._counters = {"observe": 0, "ask": 0, "fovea": 0, "errors": 0, "encode": 0, "encode_hits": 0}

    # ---- 生命周期 ----

    @property
    def bus(self) -> RetinaBus:
        if self._bus is None:
            self._bus = RetinaBus.create(self.config)
        return self._bus

    @property
    def brain(self) -> VisionBrain:
        if self._brain is None:
            self._brain = VisionBrain(self.config)
        return self._brain

    @property
    def bionic(self) -> BionicRetina:
        if self._bionic is None:
            self._bionic = BionicRetina(self.brain, self.config)
        return self._bionic

    def start(self) -> "RetinaService":
        with self._lock:
            if self._started_at:
                return self
            if self._region is None:
                whitelist = self.region_whitelist()
                if whitelist:
                    # 配置了白名单且未显式指定区域时，默认只采集第一块允许区域
                    self._region = whitelist[0]
                    self.config.data.setdefault("capture", {})["region"] = list(self._region)
            self.bus.start()
            self._started_at = time.monotonic()
            return self

    def close(self) -> None:
        with self._lock:
            if self._bus is not None:
                self._bus.close()
                self._bus = None
            self._started_at = 0.0

    def __enter__(self) -> "RetinaService":
        return self.start()

    def __exit__(self, *exc: Any) -> None:
        self.close()

    # ---- 观察 ----

    def observe(self, *, include_image: bool = True, force_keyframe: bool | None = None) -> ObserveResult:
        with self._lock:
            self.start()
            if force_keyframe is None:
                force_keyframe = self._last_frame is None
            observation = self.bus.observe(force_keyframe=bool(force_keyframe))
            self._counters["observe"] += 1
            image = None
            mime = "image/jpeg"
            cached_image = False
            if observation.has_image and observation.frame is not None:
                self._last_frame = observation.frame
                if include_image:
                    # 特征缓存：按帧内容签名（降采样 Blake2b）复用编码结果，
                    # 静止画面的重复帧不再重复 JPEG 编码。
                    scene = getattr(self.bus, "scene_profile", None)
                    quality = int(scene.quality) if scene else int(self.config.get("encode.quality", 80))
                    max_width = int(scene.max_width) if scene else int(self.config.get("encode.max_width", 1280))
                    signature = _frame_signature(observation.frame, quality, max_width)
                    if self._encode_cache and self._encode_cache[0] == signature:
                        image = self._encode_cache[1]
                        self._counters["encode_hits"] += 1
                    else:
                        image = observation.frame.encode(
                            fmt=self.config.get("encode.format", "jpeg"),
                            quality=quality,
                            max_width=max_width,
                        )
                        mime = "image/jpeg" if self.config.get("encode.format", "jpeg") == "jpeg" else "image/png"
                        self._encode_cache = (signature, image, mime)
                        self._counters["encode"] += 1
                    mime = self._encode_cache[2] if self._encode_cache else mime
            elif include_image and self._last_frame is not None and self._encode_cache:
                # 静默/未变化：工具层仍返回最近一帧，标记 cached，客户端不必重新解码。
                image = self._encode_cache[1]
                mime = self._encode_cache[2]
                cached_image = True
            result = ObserveResult(observation=observation, image=image, mime=mime, cached_image=cached_image)
            self._last_obs = result
            return result

    # ---- 提问 ----

    def ask(self, prompt: str, *, include_image: bool = True) -> AskResult:
        with self._lock:
            observe = self.observe(include_image=include_image)
            frame = observe.observation.frame if observe.has_image else None
            cached = False
            if frame is None:
                # 静默/无图时复用上一张画面（大脑仍需要视觉上下文）
                frame = self._last_frame
                cached = frame is not None
            if frame is None:
                raise RuntimeError("还没有任何可用画面，无法提问；请先 retina_observe。")
            reply = self.brain.ask(frame, prompt)
            self._counters["ask"] += 1
            return AskResult(
                reply=reply,
                observation=observe.meta(),
                frame_id=frame.frame_id,
                cached_frame=cached,
            )

    def ask_fovea(self, prompt: str, *, budget_tokens: int | None = None,
                  temperature: float = 0.0) -> dict[str, Any]:
        """仿生双路：低清全帧 → 模型按需请求变焦 → 焦点高清，受 token 预算约束。"""
        with self._lock:
            observe = self.observe(include_image=False)
            frame = observe.observation.frame if observe.has_image else None
            cached = False
            if frame is None:
                frame = self._last_frame
                cached = frame is not None
            if frame is None:
                raise RuntimeError("还没有任何可用画面，无法提问；请先 retina_observe。")
            result = self.bionic.ask(prompt, frame, budget_tokens=budget_tokens,
                                     temperature=temperature)
            self._counters["ask"] += 1
            self._counters["fovea"] += 1
            payload = result.to_dict()
            payload.update({
                "ok": True,
                "frame_id": frame.frame_id,
                "cached_frame": cached,
                "scene": getattr(self.bus, "scene_profile", None).to_dict()
                if getattr(self.bus, "scene_profile", None) else None,
                "observation": observe.meta(),
            })
            return payload

    def ask_actions(self, task: str) -> AskResult:
        with self._lock:
            observe = self.observe(include_image=True)
            frame = observe.observation.frame if observe.has_image else self._last_frame
            cached = observe.observation.frame is None and frame is not None
            if frame is None:
                raise RuntimeError("还没有任何可用画面，无法给出动作。")
            reply = self.brain.ask_actions(frame, task)
            self._counters["ask"] += 1
            return AskResult(reply=reply, observation=observe.meta(), frame_id=frame.frame_id, cached_frame=cached)

    # ---- 区域与元信息 ----

    def region_whitelist(self) -> list[tuple[int, int, int, int]]:
        raw = self.config.get("safety.region_whitelist") or []
        allowed: list[tuple[int, int, int, int]] = []
        for item in raw:
            try:
                x, y, w, h = (int(value) for value in item)
            except Exception:
                continue
            if w > 0 and h > 0:
                allowed.append((x, y, w, h))
        return allowed

    def _validate_region(self, region: tuple[int, int, int, int] | None) -> tuple[int, int, int, int] | None:
        whitelist = self.region_whitelist()
        if region is None or not whitelist:
            return region
        rect = Rect(*region)
        for x, y, w, h in whitelist:
            if rect.x >= x and rect.y >= y and rect.right <= x + w and rect.bottom <= y + h:
                return region
        raise ValueError(f"采集区域 {region} 超出安全白名单：{whitelist}")

    def set_region(self, region: tuple[int, int, int, int] | None) -> dict[str, Any]:
        with self._lock:
            if region is None and self.region_whitelist():
                # 白名单模式下“清除区域”回到第一块允许区域，而不是整屏
                region = self.region_whitelist()[0]
            region = self._validate_region(region)
            self._region = region
            if self._bus is not None:
                self._bus.set_region(region)
            self._last_frame = None
            self._encode_cache = None
            return {"region": list(region) if region else None, "capture": self.describe_capture()}

    def describe_capture(self) -> dict[str, Any]:
        with self._lock:
            if self._bus is None:
                return {"running": False}
            try:
                return self._bus.capture.describe()
            except Exception as exc:  # noqa: BLE001
                return {"running": False, "error": str(exc)}

    def list_sources(self) -> dict[str, Any]:
        from .capture import list_sources

        return {
            "sources": list_sources(),
            "active": self.config.get("capture.source", "desktop"),
            "region": list(self._region) if self._region else self.config.get("capture.region"),
        }

    def safety_status(self) -> dict[str, Any]:
        return {
            "region_whitelist": [list(item) for item in self.region_whitelist()],
            "persist_frames": bool(self.config.get("safety.persist_frames", False)),
            "allow_record": bool(self.config.get("safety.allow_record", False)),
            "default_region": list(self._region) if self._region else None,
        }

    def health(self, probe: bool = True) -> dict[str, Any]:
        """配置 + 可选的视觉能力真实探测（会消耗少量 token）。"""
        with self._lock:
            data: dict[str, Any] = dict(self.brain.status())
            data["safety"] = self.safety_status()
            data["vision_probe"] = self.brain.health_check() if probe else None
            return data

    def status(self) -> dict[str, Any]:
        with self._lock:
            data: dict[str, Any] = {
                "service": "vision-retina",
                "uptime_s": round(time.monotonic() - self._started_at, 3) if self._started_at else 0.0,
                "counters": dict(self._counters),
            "bionic": self._bionic.settings.to_dict() if self._bionic else None,
                "capture": self.describe_capture() if self._bus is not None else {"running": False},
                "brain": self.brain.status(),
                "hand": {
                    "mcp_tools": ["retina_observe", "retina_ask", "retina_set_region",
                                  "retina_list_sources", "retina_status", "retina_health"],
                    "ws": self.config.section("hand").get("ws"),
                },
                "region": list(self._region) if self._region else None,
                "safety": self.safety_status(),
                "last_frame_id": self._last_frame.frame_id if self._last_frame else None,
            }
            if self._bus is not None:
                data["bus"] = self._bus.status()
            return data
