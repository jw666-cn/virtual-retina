"""Stage B: same retina capture, different brains.

The capture layer is untouched: every image is displayed 1:1 in a borderless
window, captured once through the DXGI retina pipeline, encoded to JPEG, and
that *same byte string* is sent to every model.  Only the brain changes.

Models without a real API key are reported as skipped, never simulated.
"""
from __future__ import annotations

import json
import os
import re
import subprocess
import sys
import time
import winreg
from pathlib import Path

from PIL import Image

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
OUT = ROOT / "artifacts" / "model_compare"
OUT.mkdir(parents=True, exist_ok=True)
FRAMES = OUT / "frames"
FRAMES.mkdir(exist_ok=True)

PROMPT = (
    "请逐行转录这张图中的所有可见文字与数值，保留原始换行。"
    "看不清的字用□代替；不要解释、不要总结、不要补充图中没有的内容。"
)

TARGETS: dict[str, list[str]] = {
    "gearbox": [
        "两级圆柱齿轮减速器", "1:2", "A1", "3.5", "364", "0.85", "20",
        "104", "123", "172.5", "472", "552", "612", "16",
        "199.42", "243", "295", "550",
        "44H7/r6", "45H7/r6", "49H7/r6", "64H7/r6",
        "68H7/js6", "75H7/js6", "95H7/js6",
        "技术要求", "明细表",
    ],
    "building": [
        "教学楼一层平面图", "1:200", "84000", "35400",
        "1800", "5100", "4950", "7200", "5400", "3800",
        "3000", "4200", "3300",
        "教室", "教师办公", "接待室", "保管室", "总务室",
        "门厅", "过厅", "展示厅", "值班室", "配电室",
        "卫生间", "楼梯间", "花池", "入口平台",
    ],
    "real_ui": [
        "虚拟视网膜", "新会话", "插件", "自动化任务", "工作区",
        "接单", "默认工作区", "今日", "余额", "费用明细",
        "识图", "素材", "方法", "对比维度", "输出",
        "PaddleOCR", "Tesseract", "token", "reports/ocr_compare.md", "08:58",
    ],
    "gb_title": ["两级圆柱齿轮减速器", "1:2", "A1"],
    "gb_spec": ["3.5", "364", "0.85", "20"],
    "bd_title": ["教学楼一层平面图", "1:200"],
    "bd_dims": ["1800", "5100", "4950", "7200"],
    "ui_msg": ["素材", "方法", "对比维度", "输出", "PaddleOCR", "Tesseract", "token", "reports/ocr_compare.md"],
}

CASES = [
    ("gearbox", ROOT / "artifacts" / "cad_src" / "gearbox.jpg", 1400, 1000),
    ("building", ROOT / "artifacts" / "cad_src" / "building.jpg", 1440, 1020),
    ("real_ui", ROOT / "artifacts" / "ocr_compare" / "real_ui.png", 1600, 1000),
    ("gb_title", ROOT / "artifacts" / "cad_rec" / "crops2" / "gb_title2.png", 1200, 700),
    ("gb_spec", ROOT / "artifacts" / "cad_rec" / "crops2" / "gb_tech_spec2.png", 1200, 700),
    ("bd_title", ROOT / "artifacts" / "cad_rec" / "crops2" / "bd_title2.png", 1600, 700),
    ("bd_dims", ROOT / "artifacts" / "cad_rec" / "crops2" / "bd_top_left.png", 1600, 700),
    ("ui_msg", ROOT / "artifacts" / "ocr_compare" / "ui_msg.png", 1600, 700),
]

MODELS = [
    {
        "name": "deepseek-flash",
        "base_url": "https://api.deepseek.com",
        "model": "deepseek-flash",
        "key_envs": ["VISION_API_KEY", "DEEPSEEK_API_KEY"],
        "temperature": 0.0,
        "max_tokens_big": 16384,
        "max_tokens_small": 4096,
    },
    {
        "name": "glm-4v-flash",
        "base_url": "https://open.bigmodel.cn/api/paas/v4",
        "model": "glm-4v-flash",
        "key_envs": ["ZHIPU_API_KEY"],
        "temperature": 0.01,
        "max_tokens_big": 1024,
        "max_tokens_small": 1024,
    },
]

SKIPPED = [
    ("glm-4.5v / glm-4v-plus", "同一 Zhipu Key 调用返回 HTTP 429（限流/未开通），本轮跳过"),
    ("qwen-vl-max", "无 DashScope API Key，跳过（待补 Key）"),
    ("gpt-4o", "无 OpenAI API Key，跳过（待补 Key）"),
]


def user_env(name: str) -> str | None:
    try:
        with winreg.OpenKey(winreg.HKEY_CURRENT_USER, r"Environment") as k:
            return str(winreg.QueryValueEx(k, name)[0])
    except Exception:
        return None


def secret(name: str) -> str | None:
    p = ROOT / "config" / "secrets.local.env"
    if not p.exists():
        return None
    for line in p.read_text(encoding="utf-8").splitlines():
        if line.strip().startswith(name + "="):
            return line.split("=", 1)[1].strip()
    return None


def resolve_key(names: list[str]) -> tuple[str | None, str]:
    for n in names:
        v = os.environ.get(n) or secret(n) or user_env(n)
        if v:
            return v, n
    return None, ""


def norm(s: str) -> str:
    for a, b in {"：": ":", "，": ",", "。": ".", "／": "/", "（": "(", "）": ")",
                 "－": "-", "—": "-", "　": "", "\u3000": ""}.items():
        s = s.replace(a, b)
    return re.sub(r"\s+", "", s).lower()


def hit(text: str, target: str) -> bool:
    t = norm(target)
    if re.fullmatch(r"\d+(?:\.\d+)?", t):
        return re.search(rf"(?<![\d.]){re.escape(t)}(?![\d.])", text) is not None
    return t in text


def score(text: str, targets: list[str]) -> tuple[int, list[str], list[str]]:
    n = norm(text or "")
    hits = [t for t in targets if hit(n, t)]
    return len(hits), hits, [t for t in targets if t not in hits]


def capture_frames(only: set[str]) -> dict[str, dict]:
    import ctypes

    try:
        ctypes.windll.shcore.SetProcessDpiAwareness(2)
    except Exception:
        pass
    from vision_retina.bus import RetinaBus
    from vision_retina.capture import create_capture
    from vision_retina.config import load_config

    cfg = load_config()
    frames: dict[str, dict] = {}
    for name, path, max_w, max_h in CASES:
        if only and name not in only:
            continue
        geom_path = OUT / f"_{name}_geom.json"
        if geom_path.exists():
            geom_path.unlink()
        viewer = subprocess.Popen(
            [sys.executable, str(ROOT / "tools" / "image_viewer.py"), str(path),
             str(max_w), str(max_h), str(geom_path), "240"],
            stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL, stdin=subprocess.DEVNULL,
        )
        try:
            deadline = time.time() + 15
            while time.time() < deadline and not geom_path.exists():
                time.sleep(0.1)
            if not geom_path.exists():
                raise RuntimeError("viewer geometry missing")
            geom = json.loads(geom_path.read_text(encoding="utf-8"))
            cap = create_capture("desktop", "dxgi", region=(geom["x"], geom["y"], geom["w"], geom["h"]), config=cfg)
            bus = RetinaBus(cap, cfg)
            bus.start()
            obs = None
            for _ in range(40):
                obs = bus.observe(force_keyframe=True)
                if obs and obs.frame is not None:
                    break
                time.sleep(0.1)
            if obs is None or obs.frame is None:
                raise RuntimeError("no frame")
            data = obs.frame.encode("jpeg", quality=90, max_width=None)
            fpath = FRAMES / f"{name}.jpg"
            fpath.write_bytes(data)
            frames[name] = {"bytes": data, "path": str(fpath), "size": [obs.frame.width, obs.frame.height]}
            print(f"capture {name} {obs.frame.width}x{obs.frame.height} {len(data)}B", flush=True)
            bus.close()
        finally:
            viewer.terminate()
    return frames


def ask_with_retry(brain, data: bytes, prompt: str, max_tokens: int, temperature: float):
    last = None
    for attempt in range(4):
        try:
            t0 = time.time()
            r = brain.ask(data, prompt, max_tokens=max_tokens, temperature=temperature)
            return r, (time.time() - t0) * 1000, attempt
        except Exception as exc:  # noqa: BLE001
            last = exc
            msg = str(exc)
            wait = 4.0 if "429" in msg else 2.0
            time.sleep(wait * (attempt + 1))
    raise last


def main() -> int:
    import argparse

    ap = argparse.ArgumentParser()
    ap.add_argument("--only", default="")
    ap.add_argument("--models", default="")
    ap.add_argument("--no-capture", action="store_true")
    args = ap.parse_args()
    only = set(x.strip() for x in args.only.split(",") if x.strip())
    want = set(x.strip() for x in args.models.split(",") if x.strip())

    if args.no_capture:
        frames = {}
        for name, path, _mw, _mh in CASES:
            if only and name not in only:
                continue
            f = FRAMES / f"{name}.jpg"
            if f.exists():
                im = Image.open(f)
                frames[name] = {"bytes": f.read_bytes(), "path": str(f), "size": list(im.size)}
    else:
        frames = capture_frames(only)
    from vision_retina.brain import VisionBrain

    results = []
    for spec in MODELS:
        if want and spec["name"] not in want:
            continue
        key, kname = resolve_key(spec["key_envs"])
        row = {"model": spec["name"], "base_url": spec["base_url"], "key_env": kname,
               "available": bool(key), "cases": [], "skipped": None}
        if not key:
            row["skipped"] = "无可用 API Key"
            results.append(row)
            print(row["model"], "SKIP no key", flush=True)
            continue
        brain = VisionBrain(base_url=spec["base_url"], api_key=key, model=spec["model"])
        print(f"== {spec['name']} key={kname}", flush=True)
        for name, path, _mw, _mh in CASES:
            if name not in frames:
                continue
            fr = frames[name]
            big = fr["size"][0] * fr["size"][1] > 500_000
            budget = spec["max_tokens_big"] if big else spec["max_tokens_small"]
            try:
                reply, ms, retries = ask_with_retry(brain, fr["bytes"], PROMPT, budget, spec["temperature"])
                text = reply.text or ""
                usage = reply.usage or {}
                err = None
            except Exception as exc:  # noqa: BLE001
                text, usage, err, ms, retries = "", {}, str(exc), 0, 0
            h, hits, misses = score(text, TARGETS[name])
            row["cases"].append({
                "case": name, "hits": h, "targets": len(TARGETS[name]),
                "hit_list": hits, "miss_list": misses,
                "latency_ms": round(ms, 1), "retries": retries,
                "usage": usage, "text": text, "error": err,
            })
            print(f"   {name:9s} {h:2d}/{len(TARGETS[name]):2d}  {ms:7.0f}ms  "
                  f"tok={usage.get('total_tokens')}", flush=True)
            (OUT / f"{spec['name']}_{name}.txt").write_text(text, encoding="utf-8")
            if spec["name"].startswith("glm"):
                time.sleep(2.5)
        results.append(row)

    (OUT / "results.json").write_text(json.dumps({"results": results, "skipped": SKIPPED},
                                                 ensure_ascii=False, indent=2), encoding="utf-8")
    print("saved", OUT / "results.json")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
