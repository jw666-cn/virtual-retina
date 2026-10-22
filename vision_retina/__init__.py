"""虚拟视网膜（Vision Retina）：纯视觉输入外设。

只做「视网膜」：采集 → 统一输入总线 → 增量编码 → 标准接口。
不内置大脑（模型），不内置手（执行）；两者分别通过 OpenAI 兼容 API
与 MCP / WebSocket 协议接入，客户自选、可替换。

V2.0 起增加「视神经转化引擎」OpticNerve：视网膜与大脑之间的实时转化层，
把任何输入变成对当前大脑最优的形态（画面→增强像素/稀疏编码，文档→markdown）。
"""

__version__ = "0.1.0"

__all__ = [
    "__version__",
    "RetinaFrame",
    "Rect",
    "load_config",
    "RetinaService",
    "OpticNerve",
    "default_engine",
]


def __getattr__(name: str):
    # 延迟导入：只做 CLI/配置时不必加载 numpy / mcp / openai 相关模块
    if name in ("RetinaFrame", "Rect"):
        from .frames import Rect, RetinaFrame

        return {"RetinaFrame": RetinaFrame, "Rect": Rect}[name]
    if name == "load_config":
        from .config import load_config

        return load_config
    if name == "RetinaService":
        from .service import RetinaService

        return RetinaService
    if name in ("OpticNerve", "default_engine"):
        from .optic_nerve import OpticNerve, default_engine

        return {"OpticNerve": OpticNerve, "default_engine": default_engine}[name]
    raise AttributeError(name)
