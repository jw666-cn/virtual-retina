"""总线冒烟测试：真实采集 + 差分/静默/中央凹。

用法：
    py -3.12 tools/probe_bus.py --seconds 6 --fps 2
    py -3.12 tools/probe_bus.py --seconds 6 --fovea --save artifacts/bus_fovea.png
"""
from __future__ import annotations

import argparse
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from vision_retina.bus import RetinaBus  # noqa: E402
from vision_retina.config import load_config  # noqa: E402


def main() -> int:
    for stream in (sys.stdout, sys.stderr):
        try:
            stream.reconfigure(encoding="utf-8", errors="replace")
        except Exception:
            pass
    ap = argparse.ArgumentParser()
    ap.add_argument("--seconds", type=float, default=6.0)
    ap.add_argument("--fps", type=float, default=2.0)
    ap.add_argument("--backend", default="auto")
    ap.add_argument("--fovea", action="store_true")
    ap.add_argument("--region", default="")
    ap.add_argument("--save", default="artifacts/bus_probe.png")
    args = ap.parse_args()

    cfg = load_config()
    cfg.data["capture"]["fps"] = args.fps
    cfg.data["capture"]["backend"] = args.backend
    if args.fovea:
        cfg.data["bus"]["fovea"]["enabled"] = True
    if args.region:
        cfg.data["capture"]["region"] = [int(v) for v in args.region.split(",")]

    bus = RetinaBus.create(cfg).start()
    saved = False
    t0 = time.time()
    try:
        while time.time() - t0 < args.seconds:
            obs = bus.observe(timeout=1.5)
            print(
                f"t={time.time() - t0:5.2f}s seq={obs.seq:3d} kind={obs.kind:9s} "
                f"key={int(obs.is_key_frame)} ratio={obs.changed_ratio:7.4f} "
                f"dirty={len(obs.dirty_rects):2d} "
                + (f"img={obs.frame.width}x{obs.frame.height} " if obs.frame else "")
                + f"raw={obs.raw_size[0]}x{obs.raw_size[1]} reason={obs.reason}"
            )
            if obs.frame is not None and not saved and args.save:
                path = Path(args.save)
                path.parent.mkdir(parents=True, exist_ok=True)
                obs.frame.save(path)
                print("saved:", path, path.stat().st_size, "bytes")
                saved = True
    finally:
        bus.close()
    print("status:", bus.status())
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
