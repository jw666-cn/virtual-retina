"""采集层冒烟测试：打开指定后端，取 N 帧，打印尺寸/脏区/耗时，并可存图。

用法：
    py -3.12 tools/probe_capture.py --backend dxgi --frames 3
    py -3.12 tools/probe_capture.py --backend gdi  --frames 2 --save artifacts/gdi.png
    py -3.12 tools/probe_capture.py --source camera --frames 2
"""
from __future__ import annotations

import argparse
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from vision_retina.capture import create_capture, list_sources  # noqa: E402


def _utf8_stdout() -> None:
    for stream in (sys.stdout, sys.stderr):
        try:
            stream.reconfigure(encoding="utf-8", errors="replace")
        except Exception:
            pass


def main() -> int:
    _utf8_stdout()
    ap = argparse.ArgumentParser()
    ap.add_argument("--source", default="desktop", choices=["desktop", "camera"])
    ap.add_argument("--backend", default="auto")
    ap.add_argument("--frames", type=int, default=3)
    ap.add_argument("--fps", type=float, default=4.0)
    ap.add_argument("--save", default="")
    ap.add_argument("--region", default="", help="x,y,w,h")
    ap.add_argument("--camera-path", default="", help="摄像头不可用时的视频文件联调源")
    ap.add_argument("--list", action="store_true")
    args = ap.parse_args()

    region = None
    if args.region:
        region = tuple(int(v) for v in args.region.split(","))
        if len(region) != 4:
            ap.error("--region 需要 x,y,w,h")

    if args.list:
        import json
        print(json.dumps(list_sources(), ensure_ascii=False, indent=2))
        return 0

    kwargs = {"fps": args.fps, "ring_size": 8, "region": region}
    if args.camera_path:
        kwargs["camera_path"] = args.camera_path
    t0 = time.perf_counter()
    cap = create_capture(args.source, args.backend, **kwargs)
    print(f"opened in {(time.perf_counter() - t0) * 1000:.0f} ms")
    print("describe:", cap.describe())

    cap.start()
    got = 0
    try:
        deadline = time.perf_counter() + max(5.0, args.frames / max(args.fps, 0.5) + 5)
        while got < args.frames and time.perf_counter() < deadline:
            frame = cap.get_frame(timeout=3.0)
            if frame is None:
                print("no frame within timeout")
                continue
            got += 1
            print(
                f"frame#{got} id={frame.frame_id} {frame.width}x{frame.height} "
                f"source={frame.source} key={frame.is_key_frame} "
                f"dirty={len(frame.dirty_rects)} dirty_area={frame.dirty_area} "
                f"bytes={frame.pixels.nbytes} ts={frame.timestamp_ms}"
            )
            if args.save and got == 1:
                path = Path(args.save)
                path.parent.mkdir(parents=True, exist_ok=True)
                frame.save(path)
                print("saved:", path, path.stat().st_size, "bytes")
    finally:
        cap.close()
    print("stats:", cap.stats())
    return 0 if got == args.frames else 2


if __name__ == "__main__":
    raise SystemExit(main())
