"""Dynamic-vision test: continuous retina observation + timed questions.

The live window (tools/dynamic_window.py) runs a deterministic 20 s cycle:
  0-10 s MOTION, 10-16 s STATIC, 16-20 s FLASH (wall-clock anchored).

Usage:
  py -3.12 tools/dynamic_test.py --mode full  --json artifacts/dynamic/full.json
  py -3.12 tools/dynamic_test.py --mode delta --json artifacts/dynamic/delta.json
"""
from __future__ import annotations

import argparse
import ctypes
import json
import statistics
import sys
import time
from pathlib import Path

import numpy as np
from PIL import Image, ImageDraw, ImageFont

HERE = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(HERE))

from vision_retina.brain import VisionBrain  # noqa: E402
from vision_retina.config import load_config  # noqa: E402
from vision_retina.frames import Rect
from vision_retina.roi import expand_rect_to_content  # noqa: E402
from vision_retina.service import RetinaService  # noqa: E402

try:
    ctypes.windll.shcore.SetProcessDpiAwareness(2)
except Exception:
    try:
        ctypes.windll.user32.SetProcessDPIAware()
    except Exception:
        pass

CYCLE = 20.0
GEOM = HERE / "artifacts" / "dynamic" / "window_geom.json"
FONT = r"C:\Windows\Fonts\msyh.ttc"


def phase(t: float) -> str:
    t %= CYCLE
    return "MOTION" if t < 10 else ("STATIC" if t < 16 else "FLASH")


def ball_nx(t: float) -> float:
    ph = (t % 4) / 4
    return ph * 2 if ph < 0.5 else (1 - ph) * 2


def wait_window(lo: float, hi: float, min_left: float = 0.8) -> float:
    if lo >= CYCLE:
        lo -= CYCLE
        hi -= CYCLE
    deadline = time.time() + CYCLE + 3.0
    while time.time() < deadline:
        t = time.time() % CYCLE
        if lo <= t <= hi and (hi - t) >= min_left:
            return t
        time.sleep(0.04)
    return -1.0


def union_rect(rects: list[Rect], w: int, h: int, margin: int = 28) -> Rect | None:
    if not rects:
        return None
    x0 = min(r.x for r in rects) - margin
    y0 = min(r.y for r in rects) - margin
    x1 = max(r.x + r.w for r in rects) + margin
    y1 = max(r.y + r.h for r in rects) + margin
    x0, y0 = max(0, x0), max(0, y0)
    x1, y1 = min(w, x1), min(h, y1)
    if x1 - x0 < 80 or y1 - y0 < 60:
        return None
    return Rect(x0, y0, x1 - x0, y1 - y0)


def composite(a, b, gap_s: float) -> bytes:
    """Stack the ball band of two frames with T-1 / NOW labels."""
    h, w = a.shape[:2]
    y0, y1 = int(h * 0.24), int(h * 0.62)
    band_a = a[y0:y1]
    band_b = b[y0:y1]
    im = Image.new("RGB", (w, (y1 - y0) * 2 + 56), "white")
    im.paste(Image.fromarray(band_a), (0, 0))
    im.paste(Image.fromarray(band_b), (0, (y1 - y0) + 56))
    d = ImageDraw.Draw(im)
    f = ImageFont.truetype(FONT, 26, index=0)
    d.text((12, 6), f"BEFORE  (T-{gap_s:.1f}s)", font=f, fill=(20, 20, 20))
    d.text((12, (y1 - y0) + 62), "NOW", font=f, fill=(20, 20, 20))
    from io import BytesIO
    buf = BytesIO()
    im.save(buf, "JPEG", quality=85)
    return buf.getvalue()


def composite_multi(frames, gap_s: float) -> bytes:
    """Stack N frames band with T-/NOW labels (N=2 or 3)."""
    h, w = frames[0].shape[:2]
    y0, y1 = int(h * 0.24), int(h * 0.62)
    band_h = y1 - y0
    im = Image.new("RGB", (w, band_h * len(frames) + 28 * len(frames)), "white")
    d = ImageDraw.Draw(im)
    f = ImageFont.truetype(FONT, 22, index=0)
    for i, fr in enumerate(frames):
        top = i * (band_h + 28)
        im.paste(Image.fromarray(fr[y0:y1]), (0, top + 28))
        label = "NOW" if i == len(frames) - 1 else f"T-{gap_s*(len(frames)-1-i):.1f}s"
        d.text((12, top + 3), label, font=f, fill=(20, 20, 20))
    from io import BytesIO
    buf = BytesIO()
    im.save(buf, "JPEG", quality=85)
    return buf.getvalue()


ROI_BY_KIND = {
    "clock": (0.02, 0.00, 0.98, 0.18),
    "ball": (0.02, 0.28, 0.98, 0.60),
    "direction": (0.02, 0.28, 0.98, 0.60),
    "static": (0.02, 0.28, 0.98, 0.60),
    "static3": (0.02, 0.28, 0.98, 0.60),
    "color": (0.00, 0.00, 1.00, 1.00),
    "bgchange": (0.00, 0.00, 1.00, 1.00),
}


def roi_rect(kind: str, w: int, h: int) -> Rect | None:
    fx = ROI_BY_KIND.get(kind)
    if not fx:
        return None
    return Rect(int(fx[0] * w), int(fx[1] * h), int((fx[2] - fx[0]) * w), int((fx[3] - fx[1]) * h))


def _rect_union(rects):
    out = None
    for r in rects:
        out = r if out is None else Rect(min(out.x, r.x), min(out.y, r.y),
                                         max(out.x+out.w, r.x+r.w)-min(out.x, r.x),
                                         max(out.y+out.h, r.y+r.h)-min(out.y, r.y))
    return out


def composite_panels(frames, ratio: float = 0.60, diff: bool = False) -> bytes:
    """Side-by-side identical-geometry panels (for static/motion judgement)."""
    if diff and len(frames) >= 2:
        import numpy as np
        d = np.abs(frames[0].astype(np.int16) - frames[-1].astype(np.int16)).max(axis=2)
        frames = list(frames) + [np.clip(d * 3, 0, 255).astype(np.uint8)[..., None].repeat(3, axis=2)]
    h, w = frames[0].shape[:2]
    cw, ch = int(w * ratio), int(h * ratio)
    x0, y0 = (w - cw) // 2, (h - ch) // 2
    gap, label_h = 10, 28
    im = Image.new("RGB", (len(frames) * (cw + gap) + gap, ch + label_h + gap), "white")
    d = ImageDraw.Draw(im)
    f = ImageFont.truetype(FONT, 20, index=0)
    for i, fr in enumerate(frames):
        panel = Image.fromarray(fr[y0:y0 + ch, x0:x0 + cw])
        left = gap + i * (cw + gap)
        im.paste(panel, (left, label_h))
        d.rectangle((left - 1, label_h - 1, left + cw, label_h + ch), outline=(120, 120, 120))
        if diff and i == len(frames) - 1 and i >= 3:
            label = "差异图（放大3倍，越黑越静止）"
        elif diff and i == 2:
            label = "帧 3/3（最新）"
        elif diff and i < 3:
            label = f"帧 {i+1}/3"
        else:
            label = f"帧 {i+1}/{len(frames)}" + ("（最新）" if i == len(frames)-1 else "")
        d.text((left + 4, 4), label, font=f, fill=(20, 20, 20))
    from io import BytesIO
    buf = BytesIO()
    im.save(buf, "JPEG", quality=86)
    return buf.getvalue()


def build_delta(frame, obs, kind: str, max_w: int = 900):
    h, w = frame.pixels.shape[:2]
    roi = roi_rect(kind, w, h)
    dirty = list(obs.dirty_rects)
    if roi is not None:
        dirty = [r for r in dirty if not (r.x + r.w <= roi.x or roi.x + roi.w <= r.x or r.y + r.h <= roi.y or roi.y + roi.h <= r.y)]
    r = _rect_union(dirty) or roi
    if r is None:
        return frame, None, True
    if roi is not None:
        r = _rect_union([r, roi])
    safe_rect, ok = expand_rect_to_content(frame.pixels, r)
    if not ok or safe_rect.area > 0.62 * w * h:
        return frame, None, True
    crop = frame.crop((safe_rect.x, safe_rect.y, safe_rect.w, safe_rect.h))
    data = crop.encode("jpeg", quality=82, max_width=max_w)
    return data, safe_rect, True


def parse_clock(text: str):
    import re
    m = re.search(r"(\d{1,2})\s*[:：]\s*(\d{2})\s*[:：]\s*(\d{2})", text)
    if not m:
        return None
    return int(m.group(1)) * 3600 + int(m.group(2)) * 60 + int(m.group(3))


def expected_clock(ts_ms: int) -> int:
    t = time.localtime(ts_ms / 1000)
    return t.tm_hour * 3600 + t.tm_min * 60 + t.tm_sec


def judge(kind: str, ans: str, ts_ms: int, extra: dict):
    low = ans.replace(" ", "")
    if kind == "clock":
        got = parse_clock(ans)
        exp = expected_clock(ts_ms)
        return (got is not None and abs(got - exp) <= 3), f"expected~{exp//3600:02d}:{(exp%3600)//60:02d}:{exp%60:02d}"
    if kind == "ball":
        nx = ball_nx(ts_ms / 1000)
        exp = "左" if nx < 0.34 else ("右" if nx > 0.66 else "中")
        hit = (exp == "左" and "左" in low) or (exp == "右" and "右" in low) or (exp == "中" and ("中" in low or "中间" in low))
        return hit, f"expected {exp} (nx={nx:.2f})"
    if kind == "direction":
        ta, tb = extra["ta"], extra["tb"]
        dx = ball_nx(tb) - ball_nx(ta)
        exp = "右" if dx > 0 else "左"
        hit = (exp == "右" and "右" in low) or (exp == "左" and "左" in low)
        return hit, f"expected {exp} (dx={dx:+.2f})"
    if kind == "static":
        exp = "静止" if phase(ts_ms / 1000) == "STATIC" else "运动"
        return (exp in low), f"expected {exp} (single-frame)"
    if kind == "static3":
        hit = any(k in low for k in ("静止", "没动", "未动", "不变", "位置相同"))
        return hit, "expected 静止 (three static frames)"
    if kind == "static_pair":
        hit = any(k in low for k in ("静止", "没动", "未动", "不变", "相同"))
        return hit, "expected 静止 (two static frames)"
    if kind == "color":
        exp = extra.get("frame_color") or ("红色" if int(ts_ms / 120) % 2 == 0 else "蓝色")
        hit = (exp in ans) and (f"不是{exp}" not in ans) and (f"非{exp}" not in ans)
        return hit, f"expected {exp}"
    if kind == "bgchange":
        ca, cb = extra.get("bg_a"), extra.get("bg_b")
        if ca and cb:
            same = ca == cb
            if same:
                hit = ("相同" in ans or "一样" in ans) and ca in ans
            else:
                hit = ("红" in ans and "蓝" in ans) and ("不同" in ans or "变化" in ans or "不一样" in ans)
            return hit, f"expected {'相同' if same else '不同'} ({ca}->{cb})"
        hit = any(k in ans for k in ("红", "蓝", "相同", "不同", "变"))
        return hit, "expected background color comparison"
    return None, ""


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--mode", choices=["full", "delta"], required=True)
    ap.add_argument("--json", required=True)
    ap.add_argument("--gap", type=float, default=2.0)
    ap.add_argument("--only", default="")
    ap.add_argument("--save-crops", default="")
    args = ap.parse_args()

    geom = json.loads(GEOM.read_text(encoding="utf-8"))
    x, y, w, h = geom["x"], geom["y"], geom["w"], geom["h"]
    print(f"window client {x},{y},{w},{h}  mode={args.mode}", flush=True)

    cfg = load_config()
    cfg.data["capture"]["fps"] = 4.0
    cfg.data["capture"]["ring_size"] = 8
    cfg.data["capture"]["backend"] = "dxgi"
    cfg.data["encode"]["max_width"] = 1280
    cfg.data["encode"]["quality"] = 80
    cfg.data["bus"]["mode"] = "delta" if args.mode == "delta" else "full"
    cfg.data["bus"]["fovea"]["enabled"] = False
    cfg.data["bus"]["silence_timeout_s"] = 999.0
    cfg.data["brain"]["timeout"] = 45
    cfg.data["brain"]["retries"] = 1

    svc = RetinaService(cfg)
    svc.start()
    svc.set_region((x, y, w, h))
    time.sleep(1.2)
    svc.observe()
    brain: VisionBrain = svc.brain
    print("brain", brain.status().get("model"), "ready", flush=True)

    plan = [
        # (id, kind, phase target lo, hi, question, composite?)
        ("q1", "clock", 1.0, 3.0, "画面顶部的大号时钟现在显示几点几分几秒？只回答时间，格式 HH:MM:SS。", False),
        ("q2", "ball", 4.0, 6.0, "红色小球现在位于画面的左侧、中间还是右侧？只回答一个词。", False),
        ("q3", "direction", 6.3, 8.6, "这两帧是同一画面相隔约2秒的截图（上=较早，下=较新）。红色小球是在向左还是向右移动？只回答“向左”或“向右”。", True),
        ("q4", "static3", 10.5, 12.5, "这是同一画面相隔约0.6秒的三帧（上→下为时间顺序）。中央红色小球的位置是否变化？它是静止还是运动？只回答“静止”或“运动”。", True),
        ("q5", "clock", 13.4, 15.4, "画面顶部的大号时钟现在显示几点几分几秒？只回答时间，格式 HH:MM:SS。", False),
        ("q6", "color", 16.30, 17.20, "画面背景此刻主要是什么颜色？只回答“红色”或“蓝色”。", False),
        ("q7", "bgchange", 16.20, 17.20, "这两帧是同一画面相隔约2秒的截图（上=较早，下=较新）。两图的背景颜色相同还是不同？分别是什么颜色？用一句话回答。", True),
        ("q8", "direction", 1.6, 3.4, "这两帧是同一画面相隔约2秒的截图（上=较早，下=较新）。红色小球是在向左还是向右移动？只回答“向左”或“向右”。", True),
        ("q9", "ball", 4.2, 6.0, "红色小球现在位于画面的左侧、中间还是右侧？只回答一个词。", False),
    ]

    only = {x.strip() for x in args.only.split(",") if x.strip()}
    results = []
    for qid, kind, lo, hi, prompt, comp in plan:
        if only and qid not in only:
            continue
        wait_window(lo, hi)
        try:
            svc.bus.capture.drain()
        except Exception:
            pass
        obs_r = svc.observe()
        obs = obs_r.observation
        frame = obs.frame
        if frame is None:
            print("no frame", qid, flush=True)
            continue
        ta = tb = None
        crop_safe = True
        frames_used = 1
        if comp:
            need = 3 if kind == "static3" else 2
            seq = [(frame, obs)]
            gap_s = 1.0 if kind == "static3" else args.gap
            for _ in range(need - 1):
                time.sleep(gap_s)
                try:
                    svc.bus.capture.drain()
                except Exception:
                    pass
                ob = svc.observe().observation
                tries = 0
                while ob.frame is None and tries < 3:
                    time.sleep(0.3)
                    ob = svc.observe().observation
                    tries += 1
                if ob.frame is not None:
                    seq.append((ob.frame, ob))
            ta = seq[0][0].timestamp_ms / 1000.0
            tb = seq[-1][0].timestamp_ms / 1000.0
            frames_used = len(seq)
            crop = None
            if args.mode == "delta" and len(seq) >= 2:
                h0, w0 = seq[0][0].height, seq[0][0].width
                roi = roi_rect(kind, w0, h0)
                rects = []
                for fr, ob in seq:
                    for rr in ob.dirty_rects:
                        if roi is None or not (rr.x + rr.w <= roi.x or roi.x + roi.w <= rr.x or rr.y + rr.h <= roi.y or roi.y + roi.h <= rr.y):
                            rects.append(rr)
                base = _rect_union(rects)
                if base is not None and roi is not None:
                    base = _rect_union([base, roi])
                elif base is None:
                    base = roi
                if base is not None:
                    sr, ok = expand_rect_to_content(seq[0][0].pixels, base)
                    if ok and sr.area <= 0.62 * seq[0][0].width * seq[0][0].height:
                        crop = sr
                        pics = [fr.crop((sr.x, sr.y, sr.w, sr.h)).pixels for fr, _ in seq]
                    else:
                        crop_safe = ok
                        pics = [fr.pixels for fr, _ in seq]
                else:
                    pics = [fr.pixels for fr, _ in seq]
            else:
                pics = [fr.pixels for fr, _ in seq]
            img = composite_panels(pics, 0.24, diff=True) if kind == "static3" else composite_multi(pics, args.gap)
            anchor_ts = seq[-1][0].timestamp_ms
            img_w, img_h = seq[-1][0].width, int(seq[-1][0].height * 0.38) * len(pics) + 28 * len(pics)
            dirty_ratio = 1.0
        else:
            if args.mode == "delta":
                built, crop, crop_safe = build_delta(frame, obs, kind)
                img = built
                img_w, img_h = (crop.w, crop.h) if crop else (frame.width, frame.height)
            else:
                img = frame
                img_w, img_h = frame.width, frame.height
                crop = None
            anchor_ts = frame.timestamp_ms
            dirty_ratio = obs.changed_ratio

        if args.save_crops:
            outdir = Path(args.save_crops); outdir.mkdir(parents=True, exist_ok=True)
            data = img.encode("jpeg", 90, 0) if hasattr(img, "encode") else img
            (outdir / f"{qid}.jpg").write_bytes(data)
        extra = {"ta": ta, "tb": tb}
        if kind == "bgchange":
            def _bg(fr):
                hh, ww = fr.height, fr.width
                patch = fr.pixels[int(hh*0.70):int(hh*0.76), int(ww*0.86):int(ww*0.96)]
                mm = patch.reshape(-1, 3).mean(axis=0)
                return "红色" if mm[0] > mm[2] else "蓝色"
            extra["bg_a"] = _bg(seq[0][0])
            extra["bg_b"] = _bg(seq[-1][0])
        if kind == "color":
            band = frame.pixels[int(frame.height*0.08):int(frame.height*0.20)]
            m = band.reshape(-1, 3).mean(axis=0)
            extra["frame_color"] = "红色" if m[0] > m[2] else "蓝色"
        if crop is not None:
            prompt_sent = (f"这是全屏画面中变化区域的裁剪（原图坐标 x={crop.x}, y={crop.y}, "
                           f"尺寸 {crop.w}x{crop.h}，全屏 {frame.width}x{frame.height}）。" + prompt)
        else:
            prompt_sent = prompt

        t_req = time.time()
        from concurrent.futures import ThreadPoolExecutor, TimeoutError as FutTimeout
        with ThreadPoolExecutor(max_workers=1) as ex:
            fut = ex.submit(brain.ask, img, prompt_sent, max_tokens=2048, temperature=0.0)
            try:
                reply = fut.result(timeout=100)
            except FutTimeout:
                reply = None
                print(f"{qid} TIMEOUT after 100s", flush=True)
        if reply is not None and not (reply.text or "").strip():
            retry = brain.ask(img, prompt_sent, max_tokens=4096, temperature=0.0)
            if retry is not None and (retry.text or "").strip():
                reply = retry
        t_end = time.time()
        if reply is None:
            results.append({"qid": qid, "kind": kind, "mode": args.mode,
                            "error": "timeout", "answer": None, "correct": None,
                            "prompt_tokens": 0, "completion_tokens": 0,
                            "api_latency_ms": None, "e2e_ms": round(t_end*1000-anchor_ts,1),
                            "phase_expected": phase(anchor_ts/1000), "dirty_ratio": 0})
            Path(args.json).write_text(json.dumps({"summary": {"mode": args.mode}, "results": results}, ensure_ascii=False, indent=2), encoding="utf-8")
            continue
        usage = reply.usage or {}
        ok, exp = judge(kind, reply.text or "", anchor_ts, extra)
        rec = {
            "qid": qid, "kind": kind, "mode": args.mode,
            "phase_expected": phase(anchor_ts / 1000),
            "frame_id": frame.frame_id, "anchor_ts": anchor_ts,
            "frame_age_ms": round(t_req * 1000 - anchor_ts, 1),
            "dirty_ratio": round(dirty_ratio, 4),
            "img": [img_w, img_h],
            "prompt_tokens": usage.get("prompt_tokens"),
            "completion_tokens": usage.get("completion_tokens"),
            "api_latency_ms": reply.latency_ms,
            "e2e_ms": round(t_end * 1000 - anchor_ts, 1),
            "answer": reply.text,
            "expected": exp,
            "correct": ok,
            "prompt": prompt_sent,
        }
        results.append(rec)
        Path(args.json).write_text(json.dumps({"summary": {"mode": args.mode, "partial": True}, "results": results}, ensure_ascii=False, indent=2), encoding="utf-8")
        print(f"{qid} {kind:9s} phase={rec['phase_expected']:6s} "
              f"ptok={rec['prompt_tokens']} e2e={rec['e2e_ms']:.0f}ms "
              f"ok={ok} ans={(reply.text or '')[:60]!r}", flush=True)

    single = [r for r in results if not r["qid"] in ("q3", "q4", "q7", "q8")]
    summary = {
        "mode": args.mode,
        "questions": len(results),
        "correct": sum(1 for r in results if r["correct"]),
        "single_image_prompt_tokens": sum((r["prompt_tokens"] or 0) for r in single),
        "single_image_questions": len(single),
        "total_prompt_tokens": sum((r["prompt_tokens"] or 0) for r in results),
        "total_completion_tokens": sum((r["completion_tokens"] or 0) for r in results),
        "e2e_ms": {
            "mean": round(statistics.mean(r["e2e_ms"] for r in results), 1),
            "median": round(statistics.median(r["e2e_ms"] for r in results), 1),
            "min": min(r["e2e_ms"] for r in results),
            "max": max(r["e2e_ms"] for r in results),
        },
        "api_ms": {
            "mean": round(statistics.mean(r["api_latency_ms"] for r in results), 1),
            "median": round(statistics.median(r["api_latency_ms"] for r in results), 1),
        },
    }
    out = {"summary": summary, "results": results}
    Path(args.json).write_text(json.dumps(out, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps(summary, ensure_ascii=False, indent=2), flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
