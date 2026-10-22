"""配置加载：YAML 默认值 → 环境变量覆盖 → 本地 secrets 文件覆盖。

设计原则：
- 代码里不写死任何客户的模型地址/密钥；
- 配置缺失时给出明确可读的错误，而不是静默失败；
- 任何密钥都不进入日志/报告（序列化时统一脱敏）。
"""
from __future__ import annotations

import copy
import os
from pathlib import Path
from typing import Any

PROJECT_ROOT = Path(__file__).resolve().parent.parent
DEFAULT_CONFIG_PATH = PROJECT_ROOT / "config" / "retina.yaml"
DEFAULT_SECRETS_PATH = PROJECT_ROOT / "config" / "secrets.local.env"
EXAMPLE_SECRETS_PATH = PROJECT_ROOT / "config" / "secrets.local.env.example"

# 与 config/retina.yaml 保持一致的兜底默认值（yaml 缺失也能跑）
DEFAULTS: dict[str, Any] = {
    "capture": {
        "source": "desktop",
        "backend": "auto",
        "output_index": 0,
        "fps": 2.0,
        "ring_size": 8,
        "region": None,
    },
    "camera": {"index": 0, "path": None, "width": 1280, "height": 720, "fps": 30},
    "bus": {
        "mode": "delta",
        "keyframe_interval_s": 30,
        "silence_timeout_s": 3.0,
        "delta_threshold": 12,
        "change_min_ratio": 0.002,
        "diff_grid": 32,
        "fovea": {
            "enabled": False, "focus": "mouse", "radius": 240, "scale": 0.66, "mode": "crop",
            "low_width": 768, "low_quality": 72,
            "focus_max_width": 1024, "focus_quality": 88,
            "max_scale": 2.0, "budget_tokens": 1200, "max_area_ratio": 0.45,
        },
    },
    "adaptive": {
        "enabled": True,
        "static_ratio": 0.006,
        "static_exit_ratio": 0.010,
        "dynamic_ratio": 0.02,
        "dynamic_exit_ratio": 0.008,
        "enter_static_s": 3.0,
        "enter_dynamic_s": 0.4,
        "ema_alpha": 0.3,
        "static": {"fps": 1.0, "quality": 60, "max_width": 1000, "silence_timeout_s": 6.0},
        "normal": {"fps": 2.0, "quality": 80, "max_width": 1280, "silence_timeout_s": 3.0},
        "dynamic": {"fps": 6.0, "quality": 86, "max_width": 1600, "silence_timeout_s": 2.0},
    },
    "encode": {"format": "jpeg", "quality": 80, "max_width": 1280},
    # 视神经转化引擎（OpticNerve）：视网膜与大脑之间的转化层。
    # 画面只输出增强像素/稀疏编码，文档类才转 markdown。
    "optic_nerve": {
        "mode": "auto",  # auto | sparse | full | dual
        "max_tokens": 900,
        "target_ratio": 0.62,
        "baseline_width": 1280,
        "enhance": True,
        "sharpen": 0.62,
        "min_text_px": 16,
        "max_scale_up": 2.0,
        "sparse": True,
        "gap": 6,
        "max_blocks": 48,
        "min_block_area": 256,
        "keep_floor": 0.2,
        "dynamic_keep": 0.85,
        "dirty_bonus": 0.35,
        "jpeg_quality": 82,
        "png": False,
        "min_output_width": 360,
        "max_output_width": 1280,
        "low_width": 640,
        "low_quality": 72,
        "focus_max_width": 960,
        "focus_quality": 88,
    },
    "brain": {
        "base_url": "",
        "model": "",
        "api_key_env": "VISION_API_KEY",
        "temperature": 0.2,
        "timeout_s": 60,
        "max_tokens": 1024,
        "image_format": "auto",
        "system_prompt": "你是一台视觉外设的感知后端。你看到的是用户桌面的实时截图。请直接、具体地回答，不要寒暄。",
    },
    "hand": {"server_name": "virtual-retina", "ws": {"host": "127.0.0.1", "port": 8765}},
    "safety": {"region_whitelist": None, "persist_frames": False, "allow_record": False},
}


def _deep_merge(base: dict, override: dict) -> dict:
    out = copy.deepcopy(base)
    for key, value in (override or {}).items():
        if isinstance(value, dict) and isinstance(out.get(key), dict):
            out[key] = _deep_merge(out[key], value)
        else:
            out[key] = value
    return out


def load_dotenv(path: Path) -> dict[str, str]:
    """极简 .env 解析（KEY=VALUE，支持 # 注释与引号）。"""
    values: dict[str, str] = {}
    if not path.exists():
        return values
    for raw in path.read_text(encoding="utf-8-sig").splitlines():
        line = raw.strip()
        if not line or line.startswith("#") or "=" not in line:
            continue
        key, _, value = line.partition("=")
        key = key.strip()
        value = value.strip().strip('"').strip("'")
        if key:
            values[key] = value
    return values


class Config:
    """合并后的运行时配置。"""

    def __init__(self, data: dict[str, Any], path: Path, secrets: dict[str, str] | None = None):
        self.data = data
        self.path = path
        self.secrets = secrets or {}

    def section(self, name: str) -> dict[str, Any]:
        return copy.deepcopy(self.data.get(name, {}))

    def get(self, dotted: str, default: Any = None) -> Any:
        node: Any = self.data
        for part in dotted.split("."):
            if not isinstance(node, dict) or part not in node:
                return default
            node = node[part]
        return node

    def api_key(self) -> str | None:
        """按优先级取密钥：环境变量 → secrets.local.env → 配置内联（不推荐）。"""
        env_name = self.get("brain.api_key_env", "VISION_API_KEY")
        if env_name and os.environ.get(env_name):
            return os.environ[env_name]
        if env_name and self.secrets.get(env_name):
            return self.secrets[env_name]
        inline = self.get("brain.api_key")
        return inline or None

    def secrets_status(self) -> dict[str, Any]:
        env_name = self.get("brain.api_key_env", "VISION_API_KEY")
        return {
            "env_name": env_name,
            "from_environment": bool(os.environ.get(env_name)),
            "from_local_file": bool(self.secrets.get(env_name)),
            "local_file": str(DEFAULT_SECRETS_PATH),
            "local_file_exists": DEFAULT_SECRETS_PATH.exists(),
            "configured": bool(self.api_key()),
        }

    def to_safe_dict(self) -> dict[str, Any]:
        """脱敏后的配置快照，可安全写入报告/日志。"""
        data = copy.deepcopy(self.data)
        brain = data.setdefault("brain", {})
        if brain.get("api_key"):
            brain["api_key"] = "***"
        return data


def load_config(path: str | Path | None = None, secrets_path: str | Path | None = None) -> Config:
    import yaml  # 延迟导入，只有真正读配置时才需要

    config_path = Path(path) if path else DEFAULT_CONFIG_PATH
    file_data: dict[str, Any] = {}
    if config_path.exists():
        file_data = yaml.safe_load(config_path.read_text(encoding="utf-8")) or {}
    else:
        raise FileNotFoundError(
            f"配置文件不存在：{config_path}\n请从 config/retina.yaml 开始，或设置 RETINA_CONFIG 环境变量。"
        )

    merged = _deep_merge(DEFAULTS, file_data)

    # 环境变量覆盖（便于容器/CI 注入，不写死在 yaml 里）
    env_map = {
        "VISION_API_BASE": ("brain", "base_url"),
        "VISION_MODEL": ("brain", "model"),
    }
    for env_name, (section, key) in env_map.items():
        if os.environ.get(env_name):
            merged[section][key] = os.environ[env_name]

    secrets_file = Path(secrets_path) if secrets_path else DEFAULT_SECRETS_PATH
    secrets = load_dotenv(secrets_file)
    # secrets 文件里的 VISION_API_BASE / VISION_MODEL 也可覆盖
    if secrets.get("VISION_API_BASE"):
        merged["brain"]["base_url"] = secrets["VISION_API_BASE"]
    if secrets.get("VISION_MODEL"):
        merged["brain"]["model"] = secrets["VISION_MODEL"]

    return Config(merged, config_path, secrets)
