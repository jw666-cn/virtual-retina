"""大脑接入口自检工具（Step 2 / Step 4 用）。

用法：
  py -3.12 tools/probe_brain.py --health
  py -3.12 tools/probe_brain.py --ask "屏幕上现在显示什么？"
  py -3.12 tools/probe_brain.py --actions "打开设置并关闭 Wi-Fi" --save artifacts/brain_live.jpg
  py -3.12 tools/probe_brain.py --image path/to.png --ask "图里有什么？"
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from vision_retina.brain import BrainError, VisionBrain  # noqa: E402
from vision_retina.bus import RetinaBus  # noqa: E402
from vision_retina.config import load_config  # noqa: E402

if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")


def _capture_frame(args) -> tuple[object, dict]:
    cfg = load_config()
    cfg.data.setdefault("capture", {})["fps"] = 2.0
    if args.region:
        x, y, w, h = (int(v) for v in args.region.split(","))
        cfg.data["capture"]["region"] = [x, y, w, h]
    bus = RetinaBus.create(cfg, source="desktop", backend=args.backend)
    if args.fovea:
        bus.set_fovea(enabled=True)
    bus.start()
    try:
        observation = bus.observe(force_keyframe=True, timeout=8.0)
    finally:
        bus.close()
    if observation is None or not observation.has_image:
        raise RuntimeError("没有取到可用画面：" + (observation.reason if observation else "unknown"))
    return observation.frame, observation.to_dict()


def main() -> int:
    parser = argparse.ArgumentParser(description="大脑 API 接入口自检")
    parser.add_argument("--health", action="store_true", help="健康检查：验证模型确实能看见图片")
    parser.add_argument("--ask", default=None, help="对当前画面/指定图片提问")
    parser.add_argument("--actions", default=None, help="让模型给出下一步动作 JSON（传入任务描述）")
    parser.add_argument("--image", default=None, help="不用实时画面，改问一张本地图片")
    parser.add_argument("--save", default=None, help="把本次送给模型的画面另存为文件")
    parser.add_argument("--region", default=None, help="采集区域 x,y,w,h")
    parser.add_argument("--fovea", action="store_true", help="启用中央凹裁剪")
    parser.add_argument("--backend", default="auto", choices=["auto", "dxgi", "gdi"])
    parser.add_argument("--timeout", type=float, default=None)
    args = parser.parse_args()

    cfg = load_config()
    brain = VisionBrain(cfg)
    print(json.dumps({"status": brain.status()}, ensure_ascii=False, indent=2))
    if args.timeout:
        brain.timeout_s = args.timeout

    try:
        if args.health:
            result = brain.health_check()
            print("== health ==")
            print(json.dumps(result, ensure_ascii=False, indent=2))
            return 0 if result.get("ok") else 2

        if not (args.ask or args.actions):
            print("请指定 --health 或 --ask/--actions")
            return 1

        if args.image:
            payload = Path(args.image).read_bytes()
            reply = brain.ask(payload, args.ask or "请描述这张图片。")
        else:
            frame, meta = _capture_frame(args)
            print("== observation ==")
            print(json.dumps(meta, ensure_ascii=False))
            if args.save:
                frame.save(args.save)
                print(f"saved: {args.save}")
            if args.actions:
                reply = brain.ask_actions(frame, args.actions)
            else:
                reply = brain.ask(frame, args.ask)

        print("== reply ==")
        print(json.dumps(reply.to_dict(), ensure_ascii=False, indent=2))
        if reply.action:
            print("== action ==")
            print(json.dumps(reply.action, ensure_ascii=False, indent=2))
        return 0
    except BrainError as exc:
        print(f"[brain] {type(exc).__name__}: {exc}")
        return 3
    except Exception as exc:  # noqa: BLE001
        print(f"[error] {type(exc).__name__}: {exc}")
        return 4


if __name__ == "__main__":
    raise SystemExit(main())
