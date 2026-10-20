"""Stage A: retina (capture + VLM) vs dedicated OCR (Tesseract).

For each image:
  1. show it in a borderless 1:1 window,
  2. capture that exact rectangle through the retina DXGI pipeline,
  3. ask the configured vision model to transcribe all visible text,
  4. run Tesseract (chi_sim+eng) on the same source image,
  5. score a hand-verified target list against both transcripts.

Outputs artifacts/ocr_compare/results.json and reports/ocr_compare.md.
"""
from __future__ import annotations

import argparse
import json
import re
import shutil
import subprocess
import sys
import time
from pathlib import Path

from PIL import Image, ImageOps

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
OUT = ROOT / "artifacts" / "ocr_compare"
OUT.mkdir(parents=True, exist_ok=True)

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
    ("real_ui", OUT / "real_ui.png", 1600, 1000),
    ("gb_title", ROOT / "artifacts" / "cad_rec" / "crops2" / "gb_title2.png", 1200, 700),
    ("gb_spec", ROOT / "artifacts" / "cad_rec" / "crops2" / "gb_tech_spec2.png", 1200, 700),
    ("bd_title", ROOT / "artifacts" / "cad_rec" / "crops2" / "bd_title2.png", 1600, 700),
    ("bd_dims", ROOT / "artifacts" / "cad_rec" / "crops2" / "bd_top_left.png", 1600, 700),
    ("ui_msg", OUT / "ui_msg.png", 1600, 700),
]


def norm(s: str) -> str:
    table = {
        "：": ":", "，": ",", "。": ".", "／": "/", "（": "(", "）": ")",
        "－": "-", "—": "-", "　": "", "\u3000": "", "⌀": "φ", "Ø": "φ", "ø": "φ",
    }
    for a, b in table.items():
        s = s.replace(a, b)
    s = re.sub(r"\s+", "", s)
    return s.lower()


def hit(text: str, target: str) -> bool:
    t = norm(target)
    if not t:
        return False
    if re.fullmatch(r"\d+(?:\.\d+)?", t):
        return re.search(rf"(?<![\d.]){re.escape(t)}(?![\d.])", text) is not None
    return t in text


def score(text: str, targets: list[str]) -> tuple[int, list[str], list[str]]:
    n = norm(text or "")
    hits = [t for t in targets if hit(n, t)]
    misses = [t for t in targets if t not in hits]
    return len(hits), hits, misses


def run_tesseract(image: Path, name: str) -> tuple[str, float]:
    exe = shutil.which("tesseract") or r"C:\Program Files\Tesseract-OCR\tesseract.exe"
    im = Image.open(image).convert("RGB")
    scale = 2.0 if max(im.size) < 1200 else 1.0
    im = im.resize((int(im.width * scale), int(im.height * scale)), Image.LANCZOS)
    gray = ImageOps.autocontrast(im.convert("L"))
    tmp = OUT / f"_{name}_tess.png"
    gray.save(tmp)
    t0 = time.time()
    proc = subprocess.run(
        [exe, str(tmp), "stdout", "-l", "chi_sim+eng", "--psm", "6"],
        capture_output=True, text=True, encoding="utf-8", errors="replace", timeout=180,
    )
    dt = (time.time() - t0) * 1000
    return proc.stdout, dt


def run_retina(image: Path, geom_path: Path, max_w: int, max_h: int) -> dict:
    import ctypes

    try:
        ctypes.windll.shcore.SetProcessDpiAwareness(2)
    except Exception:
        pass

    for p in (geom_path,):
        if p.exists():
            p.unlink()
    viewer = subprocess.Popen(
        [sys.executable, str(ROOT / "tools" / "image_viewer.py"), str(image),
         str(max_w), str(max_h), str(geom_path), "240"],
        stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL, stdin=subprocess.DEVNULL,
    )
    try:
        deadline = time.time() + 15
        while time.time() < deadline and not geom_path.exists():
            time.sleep(0.1)
        if not geom_path.exists():
            raise RuntimeError("viewer did not report geometry")
        geom = json.loads(geom_path.read_text(encoding="utf-8"))

        from vision_retina.brain import VisionBrain
        from vision_retina.bus import RetinaBus
        from vision_retina.capture import create_capture
        from vision_retina.config import load_config

        cfg = load_config()
        cap = create_capture("desktop", "dxgi", region=(geom["x"], geom["y"], geom["w"], geom["h"]), config=cfg)
        bus = RetinaBus(cap, cfg)
        bus.start()
        t0 = time.time()
        obs = None
        for _ in range(40):
            obs = bus.observe(force_keyframe=True)
            if obs and obs.frame is not None:
                break
            time.sleep(0.1)
        if obs is None or obs.frame is None:
            raise RuntimeError("no frame captured")
        brain = VisionBrain(cfg)
        budget = 16384 if obs.frame.width * obs.frame.height > 500_000 else 4096
        reply = brain.ask(obs.frame, PROMPT, max_tokens=budget, temperature=0.0)
        latency = (time.time() - t0) * 1000
        data = {
            "text": reply.text or "",
            "latency_ms": round(latency, 1),
            "usage": reply.usage,
            "frame": [obs.frame.width, obs.frame.height],
            "geom": geom,
            "model": reply.model,
        }
        bus.close()
        return data
    finally:
        try:
            viewer.terminate()
        except Exception:
            pass


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--only", default="")
    args = ap.parse_args()
    only = set(x.strip() for x in args.only.split(",") if x.strip())

    results = []
    for name, path, max_w, max_h in CASES:
        if only and name not in only:
            continue
        print(f"== {name} {path}", flush=True)
        try:
            ret = run_retina(path, OUT / f"_{name}_geom.json", max_w, max_h)
        except Exception as exc:  # noqa: BLE001
            print("retina failed:", exc, flush=True)
            ret = {"text": "", "latency_ms": 0, "usage": {}, "error": str(exc)}
        (OUT / f"retina_{name}.txt").write_text(ret.get("text", ""), encoding="utf-8")
        tess_text, tess_ms = run_tesseract(path, name)
        (OUT / f"tesseract_{name}.txt").write_text(tess_text, encoding="utf-8")
        targets = TARGETS[name]
        rh, rhit, rmiss = score(ret.get("text", ""), targets)
        th, thit, tmiss = score(tess_text, targets)
        row = {
            "name": name,
            "image": str(path),
            "targets": targets,
            "retina": {**ret, "hits": rh, "hit_list": rhit, "miss_list": rmiss},
            "tesseract": {"text": tess_text, "latency_ms": round(tess_ms, 1), "hits": th,
                          "hit_list": thit, "miss_list": tmiss},
        }
        results.append(row)
        print(f"   retina {rh}/{len(targets)}  {ret.get('latency_ms',0):.0f}ms  "
              f"tokens={ret.get('usage')}  |  tess {th}/{len(targets)}  {tess_ms:.0f}ms", flush=True)

    merged = {}
    if (OUT / "results.json").exists():
        try:
            for r in json.loads((OUT / "results.json").read_text(encoding="utf-8")):
                merged[r["name"]] = r
        except Exception:
            pass
    for r in results:
        merged[r["name"]] = r
    ordered = [merged[k] for k in [c[0] for c in CASES] if k in merged]
    (OUT / "results.json").write_text(json.dumps(ordered, ensure_ascii=False, indent=2), encoding="utf-8")
    print("saved", OUT / "results.json")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
