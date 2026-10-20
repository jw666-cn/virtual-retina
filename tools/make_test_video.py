"""生成一段可控的“现实场景”测试视频，用于无摄像头时的摄像头链路/识别闭环回归。

用法：
    py -3.12 tools/make_test_video.py --out artifacts/test_reality.avi
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

import cv2
import numpy as np

ROOT = Path(__file__).resolve().parents[1]


def build(out_path: Path, seconds: float = 6.0, fps: int = 10) -> Path:
    width, height = 640, 480
    frames = int(seconds * fps)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    writer = cv2.VideoWriter(str(out_path), cv2.VideoWriter_fourcc(*"MJPG"), fps, (width, height))
    if not writer.isOpened():
        raise SystemExit(f"无法创建视频：{out_path}")

    for index in range(frames):
        frame = np.full((height, width, 3), 245, dtype=np.uint8)
        # 红色圆：从左向右移动
        cx = int(100 + (width - 200) * (index / max(1, frames - 1)))
        cv2.circle(frame, (cx, 220), 45, (0, 0, 255), -1)
        # 蓝色方块（固定在右上）
        cv2.rectangle(frame, (500, 40), (580, 120), (255, 0, 0), -1)
        # 绿色三角形（固定在左下）
        pts = np.array([[70, 400], [130, 400], [100, 340]], dtype=np.int32)
        cv2.fillPoly(frame, [pts], (0, 200, 0))
        # 黑色文字
        cv2.putText(frame, "RETINA 2026", (170, 460), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0, 0, 0), 2, cv2.LINE_AA)
        writer.write(frame)
    writer.release()
    return out_path


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default=str(ROOT / "artifacts" / "test_reality.avi"))
    ap.add_argument("--seconds", type=float, default=6.0)
    ap.add_argument("--fps", type=int, default=10)
    args = ap.parse_args()
    path = build(Path(args.out), seconds=args.seconds, fps=args.fps)
    print(f"written {path} ({path.stat().st_size} bytes)")
    return 0


if __name__ == "__main__":
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")
    raise SystemExit(main())
