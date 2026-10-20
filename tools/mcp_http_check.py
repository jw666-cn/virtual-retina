"""通用 MCP 客户端自检（Streamable HTTP 传输）。

本机沙箱禁止 asyncio 创建命名管道，stdio 子进程传输无法在此环境验证；
因此另提供 HTTP 传输：先启动服务
  py -3.12 -m vision_retina serve --transport http --port 8766
再运行
  py -3.12 tools/mcp_http_check.py --url http://127.0.0.1:8766/mcp

校验内容与 stdio 版一致：tools 列表 → list_sources → observe（图片块）→ ask → set_region → observe → 清除。
"""
from __future__ import annotations

import argparse
import asyncio
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from mcp.client.session import ClientSession  # noqa: E402
from mcp.client.streamable_http import streamable_http_client  # noqa: E402

if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")


def _summarize(result) -> dict:
    items = []
    for block in result.content or []:
        kind = getattr(block, "type", type(block).__name__)
        if kind == "text":
            items.append({"type": "text", "chars": len(block.text), "preview": block.text[:280]})
        elif kind == "image":
            items.append({"type": "image", "mime": getattr(block, "mime_type", None), "base64_chars": len(block.data or "")})
        else:
            items.append({"type": kind})
    return {"is_error": bool(getattr(result, "isError", False)), "content": items}


def _first_json(result) -> dict:
    for block in result.content or []:
        if getattr(block, "type", "") == "text":
            try:
                return json.loads(block.text)
            except Exception:  # noqa: BLE001
                return {}
    return {}


async def main(url: str, prompt: str) -> int:
    async with streamable_http_client(url) as (read, write):
        async with ClientSession(read, write, read_timeout_seconds=180) as session:
            init = await session.initialize()
            print("== initialize ==")
            print(json.dumps({"server": str(init.server_info), "protocolVersion": init.protocol_version}, ensure_ascii=False, indent=2))
            tools = await session.list_tools()
            print("== tools ==")
            for tool in tools.tools:
                print(f"- {tool.name}: {(tool.description or '').splitlines()[0][:100]}")
            names = {tool.name for tool in tools.tools}
            expected = {
                "retina_observe", "retina_ask", "retina_set_region",
                "retina_list_sources", "retina_status", "retina_health",
            }
            if not expected.issubset(names):
                print(f"[FAIL] 缺少工具：{sorted(expected - names)}")
                return 2

            print("== retina_list_sources ==")
            print(json.dumps(_summarize(await session.call_tool("retina_list_sources", {})), ensure_ascii=False, indent=2))

            print("== retina_observe ==")
            observed = await session.call_tool("retina_observe", {"include_image": True})
            summary = _summarize(observed)
            print(json.dumps(summary, ensure_ascii=False, indent=2))
            if not any(item["type"] == "image" for item in summary["content"]):
                print("[FAIL] retina_observe 没有返回图片内容块")
                return 3

            print("== retina_ask ==")
            asked = await session.call_tool("retina_ask", {"prompt": prompt})
            ask_summary = _summarize(asked)
            print(json.dumps(ask_summary, ensure_ascii=False, indent=2))
            if ask_summary["is_error"]:
                return 4

            print("== retina_set_region 0,0,800,600 ==")
            print(json.dumps(_summarize(await session.call_tool("retina_set_region", {"x": 0, "y": 0, "w": 800, "h": 600})), ensure_ascii=False, indent=2))

            print("== retina_observe (region) ==")
            region_obs = await session.call_tool("retina_observe", {"include_image": False, "force_keyframe": True})
            print(json.dumps(_summarize(region_obs), ensure_ascii=False, indent=2))
            meta = _first_json(region_obs)
            print(f"region frame: {meta.get('width')}x{meta.get('height')} dirty={meta.get('dirty_rects')}")
            if (meta.get("width"), meta.get("height")) != (800, 600):
                print("[FAIL] 区域切换后帧尺寸仍不是 800x600")
                return 5

            print("== retina_set_region 清除 ==")
            print(json.dumps(_summarize(await session.call_tool("retina_set_region", {})), ensure_ascii=False, indent=2))

            print("== retina_status ==")
            print(json.dumps(_summarize(await session.call_tool("retina_status", {})), ensure_ascii=False, indent=2))

            print("== retina_health (probe=False) ==")
            health = await session.call_tool("retina_health", {"probe": False})
            health_summary = _summarize(health)
            print(json.dumps(health_summary, ensure_ascii=False, indent=2))
            if health_summary["is_error"]:
                print("[FAIL] retina_health 调用失败")
                return 6

            print("[PASS] MCP 客户端（Streamable HTTP）全流程通过")
            return 0


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--url", default="http://127.0.0.1:8766/mcp")
    parser.add_argument("--prompt", default="屏幕上现在是什么？用两三句话描述。")
    args = parser.parse_args()
    raise SystemExit(asyncio.run(main(args.url, args.prompt)))

