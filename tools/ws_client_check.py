"""WebSocket JSON 兜底协议自检（不依赖 MCP）。

先启动服务：
  py -3.12 -m vision_retina serve --transport ws --port 8765
再运行：
  py -3.12 tools/ws_client_check.py --url ws://127.0.0.1:8765
"""
from __future__ import annotations

import argparse
import asyncio
import base64
import json
import sys

from websockets.asyncio.client import connect

if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")


def _brief(payload: dict) -> dict:
    brief = {k: v for k, v in payload.items() if k != "result"}
    result = payload.get("result")
    if isinstance(result, dict):
        copied = dict(result)
        image = copied.get("image")
        if isinstance(image, dict) and image.get("data_base64"):
            image["data_base64_chars"] = len(image.pop("data_base64"))
        brief["result"] = copied
    else:
        brief["result"] = result
    return brief


async def main(url: str, prompt: str) -> int:
    async with connect(url, max_size=32 * 1024 * 1024, open_timeout=10) as ws:
        hello = json.loads(await ws.recv())
        print("== hello ==")
        print(json.dumps(hello, ensure_ascii=False, indent=2))
        if hello.get("protocol") != "retina.ws/1":
            print("[FAIL] 协议标识不对")
            return 2

        async def call(request_id: int, kind: str, params: dict | None = None) -> dict:
            await ws.send(json.dumps({"id": request_id, "type": kind, "params": params or {}}, ensure_ascii=False))
            reply = json.loads(await ws.recv())
            print(f"== {kind} ==")
            print(json.dumps(_brief(reply), ensure_ascii=False, indent=2))
            return reply

        if not (await call(1, "ping")).get("ok"):
            return 3
        sources = await call(2, "list_sources")
        if not sources.get("ok"):
            return 4
        observed = await call(3, "observe", {"include_image": True})
        image = (observed.get("result") or {}).get("image") or {}
        if not image.get("data_base64_chars"):
            print("[FAIL] observe 没有返回图片 base64")
            return 5
        asked = await call(4, "ask", {"prompt": prompt})
        if not asked.get("ok"):
            return 6
        region = await call(5, "set_region", {"x": 0, "y": 0, "w": 800, "h": 600})
        if not region.get("ok"):
            return 7
        region_obs = await call(6, "observe", {"include_image": False, "force_keyframe": True})
        width = ((region_obs.get("result") or {}).get("width") or 0)
        if width != 800:
            print(f"[FAIL] 区域裁剪未生效，width={width}")
            return 8
        if not (await call(7, "set_region", {})).get("ok"):
            return 9
        if not (await call(8, "status")).get("ok"):
            return 10
        print("[PASS] WebSocket JSON 协议全流程通过")
        return 0


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--url", default="ws://127.0.0.1:8765")
    parser.add_argument("--prompt", default="屏幕上现在是什么？用两三句话描述。")
    args = parser.parse_args()
    raise SystemExit(asyncio.run(main(args.url, args.prompt)))
