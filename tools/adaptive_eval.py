"""V1.5 验证：场景自适应编码（仿暗适应）在不同变化率下的帧率/质量/token。

用可控的合成帧驱动同一套 RetinaBus：
  static   画面完全不变（时钟不跳）
  low      每个 tick 只有很小一块变化（0.3% 面积）
  dynamic  160px 方块每 tick 移动（约 2.7% 面积）
对比 adaptive=on / off 的发送帧数、有效帧率、单帧字节与图片 token。
"""
from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from vision_retina.bionic import estimate_image_tokens  # noqa: E402
from vision_retina.bus import RetinaBus  # noqa: E402
from vision_retina.config import load_config  # noqa: E402
from vision_retina.frames import Rect, RetinaFrame, now_ms  # noqa: E402

OUT = ROOT / "artifacts" / "adaptive"
W, H = 1200, 800


class FakeCapture:
    source = "desktop"
    region = None
    fps = 2.0

    def stats(self) -> dict:
        return {"captured": 0, "errors": 0}

    def close(self) -> None:
        return None


def scene_frame(kind: str, index: int) -> RetinaFrame:
    pixels = np.full((H, W, 3), 24, dtype=np.uint8)
    pixels[0:60, 0:520] = 240  # 静态标题条
    if kind == "low":
        pixels[300:310, 600:640] = 20 + (index % 2) * 220
    elif kind == "dynamic":
        x = 40 + (index * 37) % (W - 200)
        y = 300 + (index * 23) % (H - 260)
        pixels[y:y + 160, x:x + 160] = (230, 40, 40)
    return RetinaFrame(
        frame_id=index + 1, timestamp_ms=now_ms(), source="desktop", pixels=pixels,
        dirty_rects=[Rect(0, 0, W, H)], is_key_frame=index == 0,
    )


def encode_stats(frame: RetinaFrame, quality: int, max_width: int) -> tuple[int, int]:
    data = frame.encode(fmt="jpeg", quality=quality, max_width=max_width)
    width = min(frame.width, max_width)
    height = max(1, round(frame.height * width / frame.width))
    return len(data), estimate_image_tokens(width, height)


def run_scene(kind: str, enabled: bool, ticks: int, tick_s: float, cfg) -> dict:
    cfg.data["adaptive"]["enabled"] = enabled
    clock = {"t": 0.0}

    def now() -> float:
        return clock["t"]

    bus = RetinaBus(FakeCapture(), cfg, clock=now)  # type: ignore[arg-type]
    emitted = 0
    bytes_total = 0
    tokens_total = 0
    scenes: dict[str, int] = {}
    for index in range(ticks):
        frame = scene_frame(kind, index)
        obs = bus.evaluate(frame, force_keyframe=index == 0)
        profile = bus.scene_profile
        if profile:
            scenes[profile.name] = scenes.get(profile.name, 0) + 1
        if obs.kind == "frame" and obs.frame is not None:
            quality = int(profile.quality) if (enabled and profile) else 80
            max_width = int(profile.max_width) if (enabled and profile) else 1280
            size, tokens = encode_stats(obs.frame, quality, max_width)
            emitted += 1
            bytes_total += size
            tokens_total += tokens
        clock["t"] += tick_s
    duration = ticks * tick_s
    return {
        "scene": kind,
        "adaptive": enabled,
        "ticks": ticks,
        "duration_s": duration,
        "emitted": emitted,
        "effective_fps": round(emitted / duration, 2),
        "avg_bytes": round(bytes_total / emitted, 1) if emitted else 0,
        "avg_tokens": round(tokens_total / emitted, 1) if emitted else 0,
        "total_tokens": tokens_total,
        "scenes_seen": scenes,
        "final_scene": bus.scene_profile.name if bus.scene_profile else None,
    }


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--ticks", type=int, default=40)
    parser.add_argument("--tick", type=float, default=0.5)
    parser.add_argument("--json", default=str(OUT / "results.json"))
    args = parser.parse_args()
    cfg = load_config()
    rows = []
    for kind in ("static", "low", "dynamic"):
        for enabled in (False, True):
            rows.append(run_scene(kind, enabled, args.ticks, args.tick, cfg))
            row = rows[-1]
            print(f"{kind:8s} adaptive={str(enabled):5s} emitted={row['emitted']:2d} "
                  f"fps={row['effective_fps']:.2f} tokens={row['total_tokens']} "
                  f"final={row['final_scene']}")
    out = Path(args.json)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps({"rows": rows}, ensure_ascii=False, indent=2), encoding="utf-8")
    print("saved", out)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
