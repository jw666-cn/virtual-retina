"""Borderless image viewer used as the "real screen" for the retina pipeline.

Usage:  image_viewer.py <image> <max_w> <max_h> <geom_json> [seconds]

The window is borderless, topmost and shows the image at 1:1 unless it is
larger than (max_w, max_h), in which case it is uniformly scaled down.  The
client rectangle (screen coordinates) is written to <geom_json> so the caller
can capture exactly the displayed image.
"""
from __future__ import annotations

import ctypes
import json
import sys
import tkinter as tk
from pathlib import Path

from PIL import Image, ImageTk


def dpi_aware() -> None:
    try:
        ctypes.windll.shcore.SetProcessDpiAwareness(2)
    except Exception:
        try:
            ctypes.windll.user32.SetProcessDPIAware()
        except Exception:
            pass


def main() -> int:
    if len(sys.argv) < 5:
        print(__doc__)
        return 2
    path = Path(sys.argv[1])
    max_w, max_h = int(sys.argv[2]), int(sys.argv[3])
    geom_path = Path(sys.argv[4])
    seconds = float(sys.argv[5]) if len(sys.argv) > 5 else 180.0

    dpi_aware()
    im = Image.open(path).convert("RGB")
    scale = min(1.0, max_w / im.width, max_h / im.height)
    if scale < 1.0:
        im = im.resize((max(1, int(im.width * scale)), max(1, int(im.height * scale))), Image.LANCZOS)
    w, h = im.size

    root = tk.Tk()
    root.overrideredirect(True)
    root.attributes("-topmost", True)
    x, y = 80, 90
    root.geometry(f"{w}x{h}+{x}+{y}")
    canvas = tk.Canvas(root, width=w, height=h, highlightthickness=0, bd=0)
    canvas.pack()
    photo = ImageTk.PhotoImage(im)
    canvas.create_image(0, 0, anchor="nw", image=photo)
    canvas.image = photo  # type: ignore[attr-defined]

    def write_geom() -> None:
        root.update_idletasks()
        data = {
            "path": str(path),
            "x": int(root.winfo_rootx()),
            "y": int(root.winfo_rooty()),
            "w": int(w),
            "h": int(h),
            "scale": round(scale, 4),
        }
        geom_path.write_text(json.dumps(data, ensure_ascii=False), encoding="utf-8")

    root.after(250, write_geom)
    root.focus_force()
    root.after(int(seconds * 1000), root.destroy)
    root.mainloop()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
