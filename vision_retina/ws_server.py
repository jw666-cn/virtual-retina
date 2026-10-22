"""手接入协议之二：WebSocket JSON 兜底（任务书 §3.4）。

协议（retina.ws/1）：
  客户端 → 服务端：{"id": 1, "type": "observe", "params": {...}}
  服务端 → 客户端：{"id": 1, "ok": true, "type": "observe", "result": {...}}

type 支持：observe / ask / set_region / list_sources / status / ping
连接建立时服务端先发一条 {"type":"hello","protocol":"retina.ws/1",...}。

这个接口不依赖 MCP，任何语言/框架（Node/Python/Go/浏览器）都能消费；
服务的仍然是同一套标准结构与动作 JSON，不引入第二套语义。
"""
from __future__ import annotations

import asyncio
import json
import logging
import threading
from typing import Any

from websockets.asyncio.server import ServerConnection, serve

from .service import RetinaService

PROTOCOL = "retina.ws/1"
log = logging.getLogger("vision_retina.ws")


def _ok(request_id: Any, kind: str, result: Any) -> str:
    return json.dumps({"id": request_id, "ok": True, "type": kind, "result": result}, ensure_ascii=False)


def _err(request_id: Any, kind: str, exc: Exception) -> str:
    return json.dumps(
        {
            "id": request_id,
            "ok": False,
            "type": kind,
            "error": {"type": type(exc).__name__, "message": str(exc)},
        },
        ensure_ascii=False,
    )


class WsServerHandle:
    """后台 WebSocket 服务的句柄。"""

    def __init__(self, host: str, port: int):
        self.host = host
        self.port = port
        self.ready = threading.Event()
        self._loop: asyncio.AbstractEventLoop | None = None
        self._stop: asyncio.Event | None = None
        self._thread: threading.Thread | None = None
        self._error: BaseException | None = None

    @property
    def url(self) -> str:
        return f"ws://{self.host}:{self.port}"

    def start(self, service: RetinaService, *, timeout: float = 5.0) -> "WsServerHandle":
        self._thread = threading.Thread(target=self._run, args=(service,), name="retina-ws", daemon=True)
        self._thread.start()
        if not self.ready.wait(timeout):
            raise RuntimeError(f"WebSocket 服务启动超时：{self.url}")
        if self._error is not None:
            raise RuntimeError(f"WebSocket 服务启动失败：{self._error}") from self._error
        return self

    def _run(self, service: RetinaService) -> None:
        try:
            asyncio.run(self._main(service))
        except BaseException as exc:  # noqa: BLE001
            self._error = exc
            log.exception("WebSocket 服务异常退出")
        finally:
            self.ready.set()

    async def _main(self, service: RetinaService) -> None:
        self._loop = asyncio.get_running_loop()
        self._stop = asyncio.Event()

        async def handler(websocket: ServerConnection) -> None:
            await websocket.send(
                json.dumps(
                    {
                        "type": "hello",
                        "protocol": PROTOCOL,
                        "service": "vision-retina",
                        "types": ["observe", "ask", "ask_fovea", "ask_actions", "perceive",
                                  "set_region", "list_sources", "status", "health", "ping"],
                    },
                    ensure_ascii=False,
                )
            )
            async for raw in websocket:
                request_id: Any = None
                kind = "unknown"
                try:
                    request = json.loads(raw)
                    if not isinstance(request, dict):
                        raise ValueError("消息必须是 JSON 对象")
                    request_id = request.get("id")
                    kind = str(request.get("type") or request.get("op") or "unknown")
                    params = request.get("params") or {}
                    if not isinstance(params, dict):
                        raise ValueError("params 必须是对象")
                    result = await self._dispatch(service, kind, params)
                    await websocket.send(_ok(request_id, kind, result))
                except Exception as exc:  # noqa: BLE001
                    log.warning("WS 请求失败：%s", exc)
                    await websocket.send(_err(request_id, kind, exc))

        async with serve(handler, self.host, self.port, max_size=32 * 1024 * 1024) as server:
            self.port = int(server.sockets[0].getsockname()[1])
            self.ready.set()
            await self._stop.wait()

    async def _dispatch(self, service: RetinaService, kind: str, params: dict[str, Any]) -> Any:
        if kind == "ping":
            return {"pong": True, "protocol": PROTOCOL}
        if kind == "observe":
            include_image = bool(params.get("include_image", True))
            force = params.get("force_keyframe")
            result = await asyncio.to_thread(
                service.observe, include_image=include_image, force_keyframe=force
            )
            return result.to_dict(include_image_base64=include_image)
        if kind == "ask":
            prompt = params.get("prompt")
            if not prompt:
                raise ValueError("ask 需要 params.prompt")
            result = await asyncio.to_thread(service.ask, str(prompt), include_image=bool(params.get("include_image", True)))
            return result.to_dict()
        if kind == "ask_fovea":
            prompt = params.get("prompt")
            if not prompt:
                raise ValueError("ask_fovea 需要 params.prompt")
            budget = params.get("budget_tokens")
            return await asyncio.to_thread(
                service.ask_fovea, str(prompt), budget_tokens=int(budget) if budget else None
            )
        if kind == "ask_actions":
            task = params.get("task")
            if not task:
                raise ValueError("ask_actions 需要 params.task")
            result = await asyncio.to_thread(service.ask_actions, str(task))
            return result.to_dict()
        if kind == "perceive":
            return await asyncio.to_thread(
                service.perceive,
                force_keyframe=params.get("force_keyframe"),
                include_image=bool(params.get("include_image", False)),
            )
        if kind == "set_region":
            region = None
            provided = [params.get(key) for key in ("x", "y", "w", "h")]
            if any(value is not None for value in provided):
                if any(value is None for value in provided):
                    raise ValueError("set_region 需要同时提供 x/y/w/h，或全部省略以清除")
                region = (int(params["x"]), int(params["y"]), int(params["w"]), int(params["h"]))
            return await asyncio.to_thread(service.set_region, region)
        if kind == "list_sources":
            return await asyncio.to_thread(service.list_sources)
        if kind == "status":
            return await asyncio.to_thread(service.status)
        if kind == "health":
            return await asyncio.to_thread(service.health, bool(params.get("probe", True)))
        raise ValueError(f"不支持的 type：{kind}")

    def stop(self, timeout: float = 5.0) -> None:
        if self._loop is not None and self._stop is not None:
            self._loop.call_soon_threadsafe(self._stop.set)
        if self._thread is not None:
            self._thread.join(timeout)


def start_ws_server(
    service: RetinaService,
    host: str = "127.0.0.1",
    port: int = 8765,
) -> WsServerHandle:
    """在后台线程启动 WebSocket 服务（port=0 时由系统分配）。"""
    return WsServerHandle(host, port).start(service)
