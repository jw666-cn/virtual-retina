"""命令行入口：服务、观察、提问、诊断。"""
from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

from .config import load_config


def _print(payload) -> None:
    print(json.dumps(payload, ensure_ascii=False, indent=2))


def _parse_region(text: str | None) -> tuple[int, int, int, int] | None:
    if not text:
        return None
    parts = [int(v) for v in text.split(",")]
    if len(parts) != 4:
        raise SystemExit("--region 需要 x,y,w,h 四个整数")
    return (parts[0], parts[1], parts[2], parts[3])


def cmd_doctor(args) -> int:
    from .brain import VisionBrain
    from .capture import list_sources

    cfg = load_config()
    brain = VisionBrain(cfg)
    report = {
        "config": cfg.to_safe_dict(),
        "secrets": cfg.secrets_status(),
        "brain": brain.status(),
        "sources": list_sources(),
    }
    if not args.no_probe:
        report["health"] = brain.health_check()
    _print(report)
    health = report.get("health")
    return 0 if (health is None or health.get("ok")) else 2


def cmd_sources(args) -> int:
    from .capture import list_sources

    _print({"sources": list_sources()})
    return 0


def cmd_observe(args) -> int:
    from .service import RetinaService

    cfg = load_config()
    region = _parse_region(args.region)
    if region:
        cfg.data["capture"]["region"] = list(region)
    if args.fovea:
        cfg.data["bus"]["fovea"]["enabled"] = True
    service = RetinaService(cfg)
    try:
        result = service.observe(include_image=not args.no_image, force_keyframe=True)
        meta = result.meta()
        if result.image and args.out:
            Path(args.out).parent.mkdir(parents=True, exist_ok=True)
            Path(args.out).write_bytes(result.image)
            meta["saved"] = args.out
        _print(meta)
        return 0
    finally:
        service.close()


def cmd_ask(args) -> int:
    from .service import RetinaService

    cfg = load_config()
    region = _parse_region(args.region)
    if region:
        cfg.data["capture"]["region"] = list(region)
    service = RetinaService(cfg)
    try:
        result = service.ask(args.prompt, include_image=not args.no_image)
        _print(result.to_dict())
        return 0
    finally:
        service.close()


def cmd_set_region(args) -> int:
    from .service import RetinaService

    service = RetinaService(load_config())
    try:
        region = None if args.clear else (args.x, args.y, args.w, args.h)
        _print(service.set_region(region))
        return 0
    finally:
        service.close()


def cmd_serve(args) -> int:
    import logging

    from .service import RetinaService

    logging.basicConfig(level=logging.INFO, stream=sys.stderr, format="[%(name)s] %(levelname)s %(message)s")
    cfg = load_config()
    if args.source:
        cfg.data.setdefault("capture", {})["source"] = args.source
    service = RetinaService(cfg)
    handle = None
    try:
        if args.transport in ("ws", "both"):
            from .ws_server import start_ws_server

            ws_cfg = cfg.section("hand").get("ws") or {}
            host = args.host or ws_cfg.get("host", "127.0.0.1")
            port = args.port if args.port is not None else int(ws_cfg.get("port", 8765))
            handle = start_ws_server(service, host=host, port=port)
            print(f"[retina] WebSocket: {handle.url} (protocol retina.ws/1)", file=sys.stderr, flush=True)
        if args.transport in ("mcp", "both"):
            from .mcp_server import run_stdio

            run_stdio(service)
        elif args.transport == "http":
            from .mcp_server import run_http

            host = args.host or "127.0.0.1"
            port = args.port if args.port is not None else 8766
            print(f"[retina] MCP Streamable HTTP: http://{host}:{port}/mcp", file=sys.stderr, flush=True)
            run_http(service, host=host, port=port)
        elif handle is not None:
            print("[retina] 按 Ctrl+C 停止", file=sys.stderr, flush=True)
            while True:
                time.sleep(0.5)
    except KeyboardInterrupt:
        return 0
    finally:
        if handle is not None:
            handle.stop()
        service.close()
    return 0


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(prog="python -m vision_retina", description="虚拟视网膜：纯视觉输入外设")
    sub = parser.add_subparsers(dest="command", required=True)

    doctor = sub.add_parser("doctor", help="环境与大脑健康检查")
    doctor.add_argument("--no-probe", action="store_true", help="不发起模型探针请求")
    doctor.set_defaults(func=cmd_doctor)

    sources = sub.add_parser("sources", help="列出输入源")
    sources.set_defaults(func=cmd_sources)

    observe = sub.add_parser("observe", help="取一帧并输出元数据")
    observe.add_argument("--out", default=None, help="保存 JPEG 路径")
    observe.add_argument("--region", default=None, help="x,y,w,h")
    observe.add_argument("--fovea", action="store_true")
    observe.add_argument("--no-image", action="store_true")
    observe.set_defaults(func=cmd_observe)

    ask = sub.add_parser("ask", help="让大脑看画面并回答")
    ask.add_argument("prompt")
    ask.add_argument("--region", default=None, help="x,y,w,h")
    ask.add_argument("--no-image", action="store_true")
    ask.set_defaults(func=cmd_ask)

    region = sub.add_parser("set-region", help="设置/清除采集区域")
    region.add_argument("x", nargs="?", type=int)
    region.add_argument("y", nargs="?", type=int)
    region.add_argument("w", nargs="?", type=int)
    region.add_argument("h", nargs="?", type=int)
    region.add_argument("--clear", action="store_true")
    region.set_defaults(func=cmd_set_region)

    serve = sub.add_parser("serve", help="启动 MCP / WebSocket 服务")
    serve.add_argument("--transport", choices=["mcp", "ws", "http", "both"], default="mcp")
    serve.add_argument("--host", default=None)
    serve.add_argument("--port", type=int, default=None)
    serve.add_argument("--source", choices=["desktop", "camera"], default=None)
    serve.set_defaults(func=cmd_serve)
    return parser


def main(argv: list[str] | None = None) -> int:
    if hasattr(sys.stdout, "reconfigure"):
        sys.stdout.reconfigure(encoding="utf-8", errors="replace")
    if hasattr(sys.stderr, "reconfigure"):
        sys.stderr.reconfigure(encoding="utf-8", errors="replace")
    parser = build_parser()
    args = parser.parse_args(argv)
    return int(args.func(args) or 0)


if __name__ == "__main__":
    raise SystemExit(main())
