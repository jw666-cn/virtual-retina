"""Phase 3 现实识别闭环：摄像头（或视频文件模拟 UVC）→ 总线 → 大脑问答。"""
from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from vision_retina.config import load_config  # noqa: E402
from vision_retina.service import RetinaService  # noqa: E402

PROMPT = "画面里有哪些颜色和形状？有没有文字？请用两句话简短回答。"


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--video", default=str(ROOT / "artifacts" / "test_reality.avi"))
    ap.add_argument("--fps", type=float, default=5.0)
    ap.add_argument("--prompt", default=PROMPT)
    ap.add_argument("--save", default=str(ROOT / "artifacts" / "phase3_reality_frame.png"))
    ap.add_argument("--json", default=str(ROOT / "artifacts" / "phase3_reality.json"))
    args = ap.parse_args()

    cfg = load_config()
    cfg.data.setdefault("capture", {}).update({"source": "camera", "fps": args.fps, "region": None})
    cfg.data.setdefault("camera", {})["path"] = args.video
    cfg.data.setdefault("bus", {})["mode"] = "full"
    cfg.data["bus"].setdefault("fovea", {})["enabled"] = False
    cfg.data.setdefault("encode", {})["max_width"] = 640

    started = time.time()
    with RetinaService(cfg) as svc:
        first = svc.observe(include_image=True, force_keyframe=True)
        if not first.observation.has_image or first.image is None:
            print("[reality] 没有取到摄像头画面", flush=True)
            return 2
        frame = first.observation.frame
        Path(args.save).parent.mkdir(parents=True, exist_ok=True)
        Path(args.save).write_bytes(first.image)
        print(
            f"[reality] frame={frame.width}x{frame.height} source={frame.source} "
            f"jpg={len(first.image)} bytes capture={svc.describe_capture().get('backend')}",
            flush=True,
        )

        reply = svc.ask(args.prompt)
        text = (reply.reply.text or "").strip()
        usage = reply.reply.usage or {}
        answer = text.lower()
        checks = {
            "red": ("红" in text) or ("red" in answer),
            "blue": ("蓝" in text) or ("blue" in answer),
            "green": ("绿" in text) or ("green" in answer),
            "text": ("retina" in answer) or ("2026" in answer) or ("文字" in text),
        }
        verdict = "PASS" if (checks["red"] and checks["blue"]) else "PARTIAL"
        report = {
            "verdict": verdict,
            "video": args.video,
            "frame": {"width": frame.width, "height": frame.height, "source": frame.source},
            "answer": text,
            "checks": checks,
            "usage": usage,
            "latency_ms": round((time.time() - started) * 1000),
            "cached_frame": reply.cached_frame,
            "capture": svc.describe_capture(),
            "bus": svc.bus.status(),
        }
        Path(args.json).parent.mkdir(parents=True, exist_ok=True)
        Path(args.json).write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")
        print(f"[reality] {verdict} answer={text[:160]!r} usage={usage}", flush=True)
        print(f"[reality] report={args.json}", flush=True)
        return 0 if verdict == "PASS" else 1


if __name__ == "__main__":
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")
    sys.stderr.reconfigure(encoding="utf-8", errors="replace")
    raise SystemExit(main())
