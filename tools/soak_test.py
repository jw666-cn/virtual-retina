"""Phase 2 挂机稳定性测试：长时间连续采集 + 周期问答，监控内存/队列/错误。

用法：
    py -3.12 tools/soak_test.py --seconds 3600 --fps 1
    py -3.12 tools/soak_test.py --seconds 60 --fps 2 --ask-interval 15   # 快速冒烟

判定（写入 artifacts/phase2_soak.json）：
    - 没有采集错误、没有异常退出；
    - 队列深度不超过环形缓冲容量（不堆积）；
    - 工作集内存增长 < --max-growth-mb；
    - 全程至少完成 1 次模型问答（证明编码 + 网络 + 缓存链路在长跑中可用）。
"""

from __future__ import annotations

import argparse
import ctypes
import json
import sys
import time
from ctypes import wintypes as w
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from vision_retina.config import load_config  # noqa: E402
from vision_retina.service import RetinaService  # noqa: E402


class PROCESS_MEMORY_COUNTERS(ctypes.Structure):
    _fields_ = [
        ("cb", w.DWORD),
        ("PageFaultCount", w.DWORD),
        ("PeakWorkingSetSize", ctypes.c_size_t),
        ("WorkingSetSize", ctypes.c_size_t),
        ("QuotaPeakPagedPoolUsage", ctypes.c_size_t),
        ("QuotaPagedPoolUsage", ctypes.c_size_t),
        ("QuotaPeakNonPagedPoolUsage", ctypes.c_size_t),
        ("QuotaNonPagedPoolUsage", ctypes.c_size_t),
        ("PagefileUsage", ctypes.c_size_t),
        ("PeakPagefileUsage", ctypes.c_size_t),
    ]


def working_set_mb() -> float | None:
    counters = PROCESS_MEMORY_COUNTERS()
    counters.cb = ctypes.sizeof(counters)
    try:
        psapi = ctypes.WinDLL("psapi", use_last_error=True)
        fn = psapi.GetProcessMemoryInfo
        fn.argtypes = [w.HANDLE, ctypes.POINTER(PROCESS_MEMORY_COUNTERS), w.DWORD]
        fn.restype = w.BOOL
        ok = fn(
            w.HANDLE(ctypes.windll.kernel32.GetCurrentProcess()),
            ctypes.byref(counters),
            counters.cb,
        )
    except Exception:  # noqa: BLE001
        return None
    if not ok:
        return None
    return round(counters.WorkingSetSize / (1024 * 1024), 1)


def _write_json(path: Path, payload: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix + ".tmp")
    tmp.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")
    tmp.replace(path)


def main() -> int:
    parser = argparse.ArgumentParser(description="Phase 2 soak test")
    parser.add_argument("--seconds", type=float, default=3600.0)
    parser.add_argument("--fps", type=float, default=1.0)
    parser.add_argument("--sample-interval", type=float, default=30.0)
    parser.add_argument("--ask-interval", type=float, default=60.0)
    parser.add_argument("--max-growth-mb", type=float, default=150.0)
    parser.add_argument("--json", default=str(ROOT / "artifacts" / "phase2_soak.json"))
    args = parser.parse_args()

    cfg = load_config()
    cfg.data.setdefault("capture", {})["fps"] = max(0.2, float(args.fps))
    cfg.data.setdefault("bus", {})["silence_timeout_s"] = 2.0

    out_path = Path(args.json)
    started = time.time()
    samples: list[dict] = []
    errors: list[str] = []
    asks = 0
    tokens = 0
    queue_max = 0

    print(
        f"[soak] start seconds={args.seconds} fps={args.fps} ask_interval={args.ask_interval}",
        flush=True,
    )
    last_sample = 0.0
    last_ask = 0.0
    rss_start = None
    rss_peak = 0.0

    with RetinaService(cfg) as svc:
        svc.observe(include_image=False, force_keyframe=True)
        rss_start = working_set_mb() or 0.0
        rss_peak = rss_start

        while time.time() - started < args.seconds:
            loop_started = time.time()
            try:
                svc.observe(include_image=False)
            except Exception as exc:  # noqa: BLE001
                message = f"{type(exc).__name__}: {exc}"
                errors.append(message)
                print(f"[soak] observe error: {message}", flush=True)

            now = time.time()
            if args.ask_interval > 0 and now - last_ask >= args.ask_interval:
                last_ask = now
                try:
                    reply = svc.ask("用一句话说明当前屏幕上最显眼的内容。")
                    asks += 1
                    usage = reply.reply.usage or {}
                    tokens += int(usage.get("prompt_tokens") or 0) + int(
                        usage.get("completion_tokens") or 0
                    )
                except Exception as exc:  # noqa: BLE001
                    message = f"ask {type(exc).__name__}: {exc}"
                    errors.append(message)
                    print(f"[soak] {message}", flush=True)

            stats = svc.bus.capture.stats()
            queue_max = max(queue_max, int(stats.get("queue_depth") or 0))
            rss = working_set_mb()
            if rss is not None:
                rss_peak = max(rss_peak, rss)

            if now - last_sample >= args.sample_interval:
                last_sample = now
                sample = {
                    "t_s": round(now - started, 1),
                    "rss_mb": rss,
                    "queue_depth": stats.get("queue_depth"),
                    "captured": stats.get("captured"),
                    "dropped": stats.get("dropped"),
                    "capture_errors": stats.get("errors"),
                    "observations": svc.bus.counters.get("observations"),
                    "frames_sent": svc.bus.counters.get("frames"),
                    "silences": svc.bus.counters.get("silences"),
                    "asks": asks,
                    "tokens": tokens,
                }
                samples.append(sample)
                print(f"[soak] {json.dumps(sample, ensure_ascii=False)}", flush=True)
                _write_json(
                    out_path,
                    {
                        "status": "running",
                        "started_at": started,
                        "duration_s": round(now - started, 1),
                        "samples": samples,
                        "errors": errors,
                    },
                )

            # 维持目标帧率，同时给采集线程留出时间
            interval = 1.0 / max(0.2, float(args.fps))
            elapsed = time.time() - loop_started
            if elapsed < interval:
                time.sleep(interval - elapsed)

        final_stats = dict(svc.bus.capture.stats())
        counters = dict(svc.bus.counters)

    rss_end = working_set_mb() or 0.0
    growth = round(rss_end - (rss_start or 0.0), 1)
    verdict = (
        "PASS"
        if (not errors and queue_max <= 8 and growth < args.max_growth_mb and asks >= 1)
        else "FAIL"
    )
    report = {
        "status": "done",
        "verdict": verdict,
        "started_at": started,
        "duration_s": round(time.time() - started, 1),
        "fps": args.fps,
        "rss": {
            "start_mb": rss_start,
            "peak_mb": round(rss_peak, 1),
            "end_mb": rss_end,
            "growth_mb": growth,
            "limit_mb": args.max_growth_mb,
        },
        "queue_max": queue_max,
        "asks": asks,
        "tokens": tokens,
        "capture": final_stats,
        "bus": counters,
        "errors": errors,
        "samples": samples,
    }
    _write_json(out_path, report)
    print(f"[soak] {json.dumps({k: report[k] for k in ('verdict','duration_s','rss','queue_max','asks','tokens','errors')}, ensure_ascii=False)}", flush=True)
    return 0 if verdict == "PASS" else 1


if __name__ == "__main__":
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")
    sys.stderr.reconfigure(encoding="utf-8", errors="replace")
    raise SystemExit(main())

