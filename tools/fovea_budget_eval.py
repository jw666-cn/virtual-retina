"""V1.2–V1.4 验证：无变焦（整帧）vs 仿生双路 + 指令变焦 + 预算。

用法：
    py -3.12 tools/fovea_budget_eval.py [--budget 1000] [--low-width 640]
"""
from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

import numpy as np
from PIL import Image

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from vision_retina.bionic import BionicConfig, BionicRetina  # noqa: E402
from vision_retina.brain import VisionBrain  # noqa: E402
from vision_retina.config import load_config  # noqa: E402
from vision_retina.frames import RetinaFrame, now_ms  # noqa: E402

SRC = ROOT / "artifacts" / "cad_src"
REDRAW = ROOT / "artifacts" / "cad_redraw"
OUT = ROOT / "artifacts" / "fovea_budget"

CASES = [
    ("减速器装配图_复绘.png", "技术特性表：输入功率是多少 kW？只回答数值。", ["3.5", "3,5"]),
    ("减速器装配图_复绘.png", "技术特性表：高速轴转速是多少 r/min？只回答数值。", ["364"]),
    ("减速器装配图_复绘.png", "技术特性表：效率是多少？只回答数值。", ["0.85"]),
    ("减速器装配图_复绘.png", "技术特性表：传动比是多少？只回答数值。", ["20"]),
    ("减速器装配图_复绘.png", "标题栏里的图纸名称是什么？只回答名称。", ["两级圆柱齿轮减速器"]),
    ("减速器装配图_复绘.png", "标题栏里的比例是多少？只回答比例。", ["1:2", "1：2"]),
    ("教学楼一层平面图_复绘.png", "图纸标题是什么？只回答标题文字。", ["教学楼一层平面图"]),
    ("教学楼一层平面图_复绘.png", "图纸比例是多少？只回答比例。", ["1:200", "1：200"]),
    ("教学楼一层平面图_复绘.png", "轴线 9 与 10 之间的尺寸是多少？只回答数字。", ["1800"]),
]


def load_frame(path: Path) -> RetinaFrame:
    image = Image.open(path).convert("RGB")
    return RetinaFrame(
        frame_id=1, timestamp_ms=now_ms(), source="desktop",
        pixels=np.asarray(image), dirty_rects=[], is_key_frame=True,
    )


def hit(answer: str, expected: list[str]) -> bool:
    text = (answer or "").replace(" ", "")
    return any(item.replace(" ", "") in text for item in expected)


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--budget", type=int, default=1000)
    parser.add_argument("--low-width", type=int, default=640)
    parser.add_argument("--json", default=str(OUT / "results.json"))
    args = parser.parse_args()

    cfg = load_config()
    brain = VisionBrain(cfg)
    missing = brain.missing()
    if missing:
        print("大脑配置不完整：", missing, file=sys.stderr)
        return 2
    bionic_cfg = BionicConfig(
        low_width=args.low_width, low_quality=72,
        focus_max_width=1024, budget_tokens=args.budget,
    )
    engine = BionicRetina(brain, cfg, bionic_cfg)

    rows = []
    for index, (name, prompt, expected) in enumerate(CASES, 1):
        frame = load_frame(REDRAW / name)
        t0 = time.perf_counter()
        base = brain.ask(frame, prompt, max_tokens=512)
        base_ms = int((time.perf_counter() - t0) * 1000)
        base_tokens = int((base.usage or {}).get("prompt_tokens") or 0)

        t0 = time.perf_counter()
        bionic = engine.ask(prompt, frame, budget_tokens=args.budget)
        bionic_ms = int((time.perf_counter() - t0) * 1000)

        row = {
            "case": index,
            "image": name,
            "prompt": prompt,
            "expected": expected,
            "baseline": {
                "answer": base.text.strip(),
                "ok": hit(base.text, expected),
                "prompt_tokens": base_tokens,
                "completion_tokens": int((base.usage or {}).get("completion_tokens") or 0),
                "latency_ms": base_ms,
            },
            "bionic": {
                "answer": bionic.answer,
                "ok": hit(bionic.answer, expected),
                "prompt_tokens": bionic.prompt_tokens,
                "completion_tokens": bionic.completion_tokens,
                "estimated_tokens": bionic.estimated_tokens,
                "budget_tokens": bionic.budget_tokens,
                "budget_ok": bionic.budget_ok,
                "calls": bionic.calls,
                "zoom": list(bionic.zoom) if bionic.zoom else None,
                "zoom_reason": bionic.zoom_reason,
                "denied_reason": bionic.denied_reason,
                "low_size": [bionic.low.width, bionic.low.height],
                "focus_size": [bionic.focus.width, bionic.focus.height] if bionic.focus else None,
                "latency_ms": bionic_ms,
            },
        }
        rows.append(row)
        print(f"[{index}] {name} base={row['baseline']['ok']} {base_tokens}tok "
              f"bionic={row['bionic']['ok']} {bionic.prompt_tokens}tok "
              f"calls={bionic.calls} zoom={row['bionic']['zoom']} "
              f"budget_ok={bionic.budget_ok}")

    summary = {
        "baseline_hits": sum(1 for r in rows if r["baseline"]["ok"]),
        "bionic_hits": sum(1 for r in rows if r["bionic"]["ok"]),
        "cases": len(rows),
        "baseline_tokens": sum(r["baseline"]["prompt_tokens"] for r in rows),
        "bionic_tokens": sum(r["bionic"]["prompt_tokens"] for r in rows),
        "bionic_estimated": sum(r["bionic"]["estimated_tokens"] for r in rows),
        "budget_tokens": args.budget,
        "budget_all_ok": all(r["bionic"]["budget_ok"] for r in rows),
        "zoom_cases": sum(1 for r in rows if r["bionic"]["calls"] == 2),
        "baseline_latency_ms": sum(r["baseline"]["latency_ms"] for r in rows),
        "bionic_latency_ms": sum(r["bionic"]["latency_ms"] for r in rows),
        "low_width": args.low_width,
    }
    payload = {"summary": summary, "rows": rows}
    out = Path(args.json)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps(summary, ensure_ascii=False, indent=2))
    print("saved", out)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
