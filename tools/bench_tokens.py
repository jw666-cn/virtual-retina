"""Phase 2 token 基线：全帧 vs 中央凹 vs 区域 vs 静默复用。

对同一个画面用同一句提示词，分别测量 prompt tokens（含图片）与延迟，
输出倍率并落盘 artifacts/phase2_tokens.json。

    py -3.12 tools/bench_tokens.py
"""
from __future__ import annotations

import json
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from vision_retina.config import load_config  # noqa: E402
from vision_retina.service import RetinaService  # noqa: E402

if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")

PROMPT = "用一句话描述当前屏幕上的主要内容。"
OUT = ROOT / "artifacts" / "phase2_tokens.json"


def _measure(svc: RetinaService, label: str, detail: str | None = None) -> dict:
    observed = svc.observe(include_image=True, force_keyframe=True)
    frame = observed.observation.frame
    started = time.time()
    result = svc.ask(PROMPT, include_image=True) if detail is None else svc.ask(PROMPT, include_image=True)
    reply = result.reply
    usage = reply.usage or {}
    record = {
        "label": label,
        "frame_size": [frame.width, frame.height],
        "image_bytes": len(observed.image or b""),
        "prompt_tokens": usage.get("prompt_tokens"),
        "completion_tokens": usage.get("completion_tokens"),
        "total_tokens": usage.get("total_tokens"),
        "latency_ms": reply.latency_ms,
        "wall_ms": round((time.time() - started) * 1000),
        "answer": (reply.text or "")[:120],
        "cached_frame": result.cached_frame,
    }
    print(f"{label:10s} frame={frame.width}x{frame.height} prompt={record['prompt_tokens']} img={record['image_bytes']} ms={record['latency_ms']} cached={record['cached_frame']}")
    return record


def main() -> int:
    cfg = load_config()
    cfg.data.setdefault("bus", {}).setdefault("fovea", {})["enabled"] = False
    cfg.data.setdefault("encode", {})["max_width"] = 2240  # 基线用原生分辨率
    records: list[dict] = []

    with RetinaService(cfg) as svc:
        # 文本-only 基线：用与 ask() 完全相同的 system + prompt，只去掉图片，
        # 这样 prompt 差值就是“单帧图片 token”，不被指令文本污染。
        sys_prompt = svc.brain.system_prompt
        text_messages = ([{"role": "system", "content": sys_prompt}] if sys_prompt else []) + [
            {"role": "user", "content": PROMPT}
        ]
        text_reply = svc.brain.chat(text_messages, max_tokens=8)
        text_only = (text_reply.usage or {}).get("prompt_tokens") or 0
        print(f"text-only  prompt={text_only} (system={'yes' if sys_prompt else 'no'})")

        full = _measure(svc, "full-native")
        records.append(full)

        # 中央凹：不同半径/缩放，找一个既省 token 又保留细节的默认值
        fovea_records = []
        for radius, scale in ((240, 0.66), (240, 0.5), (200, 0.5), (180, 0.5)):
            svc.bus.set_fovea(True, radius=radius, scale=scale)
            record = _measure(svc, f"fovea-{radius}-{scale}")
            record["fovea"] = {"radius": radius, "scale": scale}
            fovea_records.append(record)
            records.append(record)
        svc.bus.set_fovea(False)
        fovea = fovea_records[0]

        # 区域白名单式裁剪
        svc.set_region((0, 0, 800, 600))
        region = _measure(svc, "region")
        records.append(region)
        svc.set_region(None)

        # 静默复用：连续两次 ask，第二次应命中缓存帧
        first = svc.ask(PROMPT, include_image=True)
        time.sleep(0.5)
        second = svc.ask(PROMPT, include_image=True)
        cached = {
            "label": "cached",
            "first_frame_id": first.frame_id,
            "second_frame_id": second.frame_id,
            "second_cached": second.cached_frame,
            "second_prompt_tokens": (second.reply.usage or {}).get("prompt_tokens"),
            "second_latency_ms": second.reply.latency_ms,
        }
        print(f"cached     first_frame={first.frame_id} second_frame={second.frame_id} cached={second.cached_frame}")
        records.append(cached)

    base = (full.get("prompt_tokens") or 0) - text_only
    best = max(fovea_records, key=lambda r: (base / max(1, (r.get("prompt_tokens") or 0) - text_only)))
    best_image_tokens = max(1, (best.get("prompt_tokens") or 0) - text_only)
    best_ratio = round(base / best_image_tokens, 2) if base > 0 else None
    summary = {
        "baseline": full,
        "text_only_tokens": text_only,
        "image_tokens": {"full": base, "fovea": best_image_tokens},
        "records": records,
        "ratios": {
            "full_over_fovea": round((full.get("prompt_tokens") or 0) / (fovea.get("prompt_tokens") or 1), 2),
            "full_over_best_fovea_image_tokens": best_ratio,
            "full_over_region": round((full.get("prompt_tokens") or 0) / (region.get("prompt_tokens") or 1), 2),
        },
        "best_fovea": {"radius": best.get("fovea", {}).get("radius"), "scale": best.get("fovea", {}).get("scale")},
        "verdict": "PASS" if (best_ratio and best_ratio >= 5) else "PARTIAL",
    }
    OUT.write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8")
    print("ratios:", summary["ratios"], "verdict:", summary["verdict"])
    print("report:", OUT)
    return 0 if summary["verdict"] == "PASS" else 1


if __name__ == "__main__":
    raise SystemExit(main())
