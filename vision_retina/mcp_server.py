"""手接入协议之一：MCP Server（任务书 §3.4）。

对外只暴露稳定的 tools：
  retina_observe      请求当前画面/增量（可选带图）
  retina_ask          让大脑看画面并回答（可选返回动作 JSON）
  retina_ask_fovea    仿生双路：低清全帧 + 按需焦点高清（受 token 预算约束）
  retina_set_region   设定采集区域（隐私控制）
  retina_list_sources 列出可用输入源（desktop / camera）
  retina_status       运行状态（诊断扩展，非必需）
  retina_health       视觉能力健康检查（诊断扩展）

工具名与参数即契约；任何 MCP 客户端（dsh / openclaw / opencode / 自研）都能接，
本模块不 import 任何具体"手"的代码。
"""
from __future__ import annotations

import json
from typing import Any

from mcp.server.mcpserver import MCPServer
from mcp.server.mcpserver.utilities.types import Image
from mcp.types import ImageContent, TextContent

from . import __version__
from .service import RetinaService

INSTRUCTIONS = (
    "虚拟视网膜是一个纯视觉输入外设，不是 agent、不执行任何操作。"
    "用 retina_observe 取当前桌面/摄像头画面（图片内容块 + 元数据），"
    "用 retina_ask 让客户配置的视觉模型看图回答，"
    "用 retina_set_region 限定采集区域，用 retina_list_sources 查看输入源。"
    "返回的动作 JSON 由调用方（手）自行执行与审批，本服务不执行动作。"
)


def _json_text(payload: Any) -> str:
    return json.dumps(payload, ensure_ascii=False, indent=2)


def build_mcp_server(service: RetinaService, *, name: str = "virtual-retina") -> MCPServer:
    server = MCPServer(name=name, version=__version__, instructions=INSTRUCTIONS)

    @server.tool(
        description=(
            "获取当前视觉输入的一帧（默认桌面，摄像头需先配置）。返回图片内容块 + JSON 元数据。"
            "画面未变化时返回最近一帧并标记 image.cached=true；include_image=false 时只返回元数据。"
        )
    )
    def retina_observe(include_image: bool = True, force_keyframe: bool = False) -> list[TextContent | ImageContent]:
        result = service.observe(include_image=include_image, force_keyframe=force_keyframe)
        blocks: list[TextContent | ImageContent] = [
            TextContent(type="text", text=_json_text(result.meta()))
        ]
        if result.image:
            blocks.append(Image(data=result.image, format=result.mime.split("/")[-1]).to_image_content())
        return blocks

    @server.tool(
        description=(
            "让客户配置的视觉模型看当前画面并回答 prompt。返回 JSON：answer/action/usage/latency_ms/observation。"
            "若模型给出了标准动作 JSON，action 字段可直接交给手执行（本服务不执行）。"
        )
    )
    def retina_ask(prompt: str, include_image: bool = True) -> str:
        result = service.ask(prompt, include_image=include_image)
        return _json_text(result.to_dict())

    @server.tool(
        description=(
            "仿生双路提问：先给模型一张低清全帧；模型若请求放大，视网膜按语义安全 ROI 裁剪并回传焦点高清图，"
            "随后只让模型回答一次。总图片 token 受 budget_tokens 约束。返回 zoom/两路口径/用量。"
        )
    )
    def retina_ask_fovea(prompt: str, budget_tokens: int = 1200) -> str:
        return _json_text(service.ask_fovea(prompt, budget_tokens=budget_tokens))

    @server.tool(
        description=(
            "设定采集区域（屏幕像素坐标），用于隐私控制与降低 token。传 x=y=w=h=null 清除区域回到全屏。"
        )
    )
    def retina_set_region(
        x: int | None = None,
        y: int | None = None,
        w: int | None = None,
        h: int | None = None,
    ) -> str:
        values = [x, y, w, h]
        if all(value is None for value in values):
            region = None
        elif any(value is None for value in values):
            raise ValueError("x/y/w/h 必须同时提供，或全部留空以清除区域。")
        else:
            if w is None or h is None or w <= 0 or h <= 0:
                raise ValueError("w/h 必须为正整数。")
            region = (int(x), int(y), int(w), int(h))  # type: ignore[arg-type]
        try:
            return _json_text(service.set_region(region))
        except ValueError as exc:
            return _json_text({"ok": False, "error": str(exc), "safety": service.safety_status()})

    @server.tool(
        description=(
            "视觉能力健康检查：探测客户配置的模型是否真的能看图（用两张纯色小图验证）。"
            "probe=false 时只返回配置状态，不消耗 token。"
        )
    )
    def retina_health(probe: bool = True) -> str:
        return _json_text(service.health(probe=probe))

    @server.tool(description="列出可用输入源（desktop/camera）与当前活动源、区域。")
    def retina_list_sources() -> str:
        return _json_text(service.list_sources())

    @server.tool(description="查看服务状态（采集后端、总线计数、大脑配置、协议端口）。诊断用扩展。")
    def retina_status() -> str:
        return _json_text(service.status())

    return server


def run_stdio(service: RetinaService | None = None) -> None:
    """以 stdio 传输启动 MCP Server（适配绝大多数 MCP 客户端）。"""
    service = service or RetinaService()
    server = build_mcp_server(service)
    try:
        server.run("stdio")
    finally:
        service.close()


def run_http(service: RetinaService | None = None, host: str = "127.0.0.1", port: int = 8766) -> None:
    """以 Streamable HTTP 传输启动 MCP Server（不方便拉起子进程的客户端用）。"""
    service = service or RetinaService()
    server = build_mcp_server(service)
    try:
        server.run(
            "streamable-http",
            host=host,
            port=port,
            max_request_body_size=32 * 1024 * 1024,
        )
    finally:
        service.close()
