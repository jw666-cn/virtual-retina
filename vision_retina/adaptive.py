"""场景自适应编码（仿“暗适应”）。

静态/低变化 → 降采样率、降质量、缩编码宽度、延长静默窗口；
动态/高变化 → 提高采样与编码保真度。
用 EMA + 进入/退出阈值避免在场景切换时抖动。
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Any

from .config import Config, load_config

SCENE_STATIC = "static"
SCENE_NORMAL = "normal"
SCENE_DYNAMIC = "dynamic"


@dataclass(frozen=True)
class SceneProfile:
    name: str
    fps: float
    quality: int
    max_width: int
    silence_timeout_s: float
    reason: str = ""

    def to_dict(self) -> dict[str, Any]:
        return {
            "name": self.name,
            "fps": self.fps,
            "quality": self.quality,
            "max_width": self.max_width,
            "silence_timeout_s": self.silence_timeout_s,
            "reason": self.reason,
        }


def _profile(name: str, data: dict[str, Any], defaults: dict[str, Any], reason: str) -> SceneProfile:
    merged = {**defaults, **(data or {})}
    return SceneProfile(
        name=name,
        fps=max(0.1, float(merged.get("fps", 2.0))),
        quality=max(10, min(95, int(merged.get("quality", 80)))),
        max_width=max(320, int(merged.get("max_width", 1280))),
        silence_timeout_s=max(0.5, float(merged.get("silence_timeout_s", 3.0))),
        reason=reason,
    )


class AdaptiveController:
    """把变化率序列映射成场景档位。"""

    def __init__(self, config: Config | None = None):
        cfg = config or load_config()
        data = cfg.section("adaptive") or {}
        self.enabled = bool(data.get("enabled", True))
        self.static_ratio = float(data.get("static_ratio", 0.002))
        self.static_exit_ratio = float(data.get("static_exit_ratio", 0.006))
        self.dynamic_ratio = float(data.get("dynamic_ratio", 0.02))
        self.dynamic_exit_ratio = float(data.get("dynamic_exit_ratio", 0.008))
        self.enter_static_s = float(data.get("enter_static_s", 3.0))
        self.enter_dynamic_s = float(data.get("enter_dynamic_s", 0.4))
        self.ema_alpha = float(data.get("ema_alpha", 0.3))
        defaults = {"fps": 2.0, "quality": 80, "max_width": 1280, "silence_timeout_s": 3.0}
        self.profiles = {
            SCENE_STATIC: _profile(SCENE_STATIC, data.get("static") or {},
                                   {**defaults, "fps": 0.5, "quality": 58, "max_width": 960,
                                    "silence_timeout_s": 6.0}, "静态/低变化"),
            SCENE_NORMAL: _profile(SCENE_NORMAL, data.get("normal") or {}, defaults, "常规"),
            SCENE_DYNAMIC: _profile(SCENE_DYNAMIC, data.get("dynamic") or {},
                                    {**defaults, "fps": 6.0, "quality": 86, "max_width": 1600,
                                     "silence_timeout_s": 2.0}, "动态/高变化"),
        }
        self.ema = 0.0
        self.static_for = 0.0
        self.dynamic_for = 0.0
        self.current = self.profiles[SCENE_NORMAL]
        self.updates = 0

    def reset(self) -> None:
        self.ema = 0.0
        self.static_for = 0.0
        self.dynamic_for = 0.0
        self.current = self.profiles[SCENE_NORMAL]
        self.updates = 0

    def update(self, dt_s: float, changed_ratio: float, *, keyframe: bool = False) -> SceneProfile:
        dt = max(0.0, min(float(dt_s or 0.0), 2.0))
        ratio = max(0.0, float(changed_ratio or 0.0))
        self.ema = (1 - self.ema_alpha) * self.ema + self.ema_alpha * ratio
        self.updates += 1
        if not self.enabled:
            self.current = self.profiles[SCENE_NORMAL]
            return self.current

        # 档位内用更宽的退出阈值，避免边界抖动。
        if self.current.name == SCENE_STATIC:
            if self.ema > self.static_exit_ratio:
                self.static_for = 0.0
                self.current = self.profiles[SCENE_NORMAL]
        elif self.current.name == SCENE_DYNAMIC:
            if self.ema < self.dynamic_exit_ratio:
                self.dynamic_for = 0.0
                self.current = self.profiles[SCENE_NORMAL]

        if self.current.name == SCENE_NORMAL:
            if self.ema < self.static_ratio:
                self.static_for += dt
                self.dynamic_for = 0.0
                if self.static_for >= self.enter_static_s:
                    self.current = self.profiles[SCENE_STATIC]
            elif self.ema > self.dynamic_ratio:
                self.dynamic_for += dt
                self.static_for = 0.0
                if self.dynamic_for >= self.enter_dynamic_s:
                    self.current = self.profiles[SCENE_DYNAMIC]
            else:
                self.static_for = max(0.0, self.static_for - dt * 0.5)
                self.dynamic_for = max(0.0, self.dynamic_for - dt * 0.5)
        return self.current

    def status(self) -> dict[str, Any]:
        return {
            "enabled": self.enabled,
            "scene": self.current.to_dict(),
            "ema_change_ratio": round(self.ema, 5),
            "static_for_s": round(self.static_for, 2),
            "dynamic_for_s": round(self.dynamic_for, 2),
            "updates": self.updates,
        }


__all__ = ["AdaptiveController", "SceneProfile", "SCENE_STATIC", "SCENE_NORMAL", "SCENE_DYNAMIC"]
