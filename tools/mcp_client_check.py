"""通用 MCP 客户端自检（不依赖任何具体"手"）。

它会以标准 stdio 方式启动视网膜 MCP Server，列出 tools，然后依次调用：
retina_list_sources → retina_observe → retina_ask → retina_set_region → retina_observe → 清除区域。

用法：
  py -3.12 tools/mcp_client_check.py
  py -3.12 tools/mcp_client_check.py --prompt "屏幕上现在是什么？"
"""
from __future__ import annotations

import argparse
import asyncio
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from mcp import StdioServerParameters  # noqa: E402
from mcp.client.session import ClientSession  # noqa: E402
from mcp.client.stdio import stdio_client  # noqa: E402

if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")


def _summarize(result) -> dict:
    items = []
    for block in result.content or []:
        kind = getattr(block, "type", type(block).__name__)
        if kind == "text":
            text = block.text
            items.append({"type": "text", "chars": len(text), "preview": text[:280]})
        elif kind == "image":
            items.append({"type": "image", "mime": getattr(block, "mimeType", None), "base64_chars": len(block.data or "")})
        else:
            items.append({"type": kind})
    return {"is_error": bool(getattr(result, "isError", False)), "content": items}


async def main(prompt: str) -> int:
    params = StdioServerParameters(
        command=sys.executable,
        args=["-m", "vision_retina", "serve", "--transport", "mcp"],
        cwd=str(ROOT),
        env=None,
    )
    async with stdio_client(params) as (read, write):
        async with ClientSession(read, write, read_timeout_seconds=180) as session:
            init = await session.initialize()
            print("== initialize ==")
            print(json.dumps(
                {
                    "server": init.server_info.model_dump() if hasattr(init.server_info, "model_dump") else str(init.server_info),
                    "protocolVersion": init.protocol_version,
                    "capabilities": list((init.capabilities.model_dump() if hasattr(init.capabilities, "model_dump") else {}).keys()),
                },
                ensure_ascii=False,
                indent=2,
            ))
            tools = await session.list_tools()
            print("== tools ==")
            for tool in tools.tools:
                print(f"- {tool.name}: {(tool.description or '').splitlines()[0][:100]}")
            names = {tool.name for tool in tools.tools}
            expected = {"retina_observe", "retina_ask", "retina_set_region", "retina_list_sources"}
            if not expected.issubset(names):
                print(f"[FAIL] 缺少工具：{sorted(expected - names)}")
                return 2

            print("== retina_list_sources ==")
            print(json.dumps(_summarize(await session.call_tool("retina_list_sources", {})), ensure_ascii=False, indent=2))

            print("== retina_observe ==")
            observed = await session.call_tool("retina_observe", {"include_image": True})
            summary = _summarize(observed)
            print(json.dumps(summary, ensure_ascii=False, indent=2))
            has_image = any(item["type"] == "image" for item in summary["content"])
            if not has_image:
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

            print("== retina_set_region 清除 ==")
            print(json.dumps(_summarize(await session.call_tool("retina_set_region", {})), ensure_ascii=False, indent=2))

            print("== retina_status ==")
            status = await session.call_tool("retina_status", {})
            print(json.dumps(_summarize(status), ensure_ascii=False, indent=2))
            print("[PASS] MCP 标准客户端全流程通过")
            return 0


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--prompt", default="屏幕上现在是什么？用两三句话描述。")
    args = parser.parse_args()
    raise SystemExit(asyncio.run(main(args.prompt)))

