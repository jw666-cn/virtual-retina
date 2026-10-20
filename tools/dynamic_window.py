"""Deterministic live GUI used for the dynamic-vision test.

20 s cycle anchored to the wall clock (Date.now() equivalent = time.time()):
  0-10 s   MOTION : red ball sweeps left<->right (4 s period) with a motion trail
  10-16 s  STATIC : ball frozen at centre, dark green background
  16-20 s  FLASH  : background alternates red/blue every 2.0 s, ball hidden
A 10 s MP4 (motion.mp4) loops in the lower-left corner as a second moving source.
"""
from __future__ import annotations

import ctypes
import json
import time
from pathlib import Path

import cv2
from PIL import Image, ImageTk
import tkinter as tk

HERE = Path(__file__).resolve().parent.parent
OUT = HERE / "artifacts" / "dynamic"
VIDEO = OUT / "motion.mp4"
GEOM = OUT / "window_geom.json"

try:
    ctypes.windll.shcore.SetProcessDpiAwareness(2)
except Exception:
    try:
        ctypes.windll.user32.SetProcessDPIAware()
    except Exception:
        pass

CYCLE = 20.0
W, H = 1200, 800


def phase(t: float) -> str:
    if t < 10:
        return "MOTION"
    if t < 16:
        return "STATIC"
    return "FLASH"


def ball_nx(t: float) -> float:
    ph = (t % 4) / 4
    return ph * 2 if ph < 0.5 else (1 - ph) * 2


def main() -> None:
    root = tk.Tk()
    root.title("Retina Dynamic Test")
    root.geometry(f"{W}x{H}+120+80")
    root.configure(bg="#0b1016")
    canvas = tk.Canvas(root, width=W, height=H, highlightthickness=0, bd=0)
    canvas.pack(fill="both", expand=True)

    cap = cv2.VideoCapture(str(VIDEO))
    cap.set(cv2.CAP_PROP_BUFFERSIZE, 1)
    photo = None
    tick = {"n": 0}

    def draw() -> None:
        nonlocal photo
        now = time.time()
        t = now % CYCLE
        ph = phase(t)

        if ph == "FLASH":
            bg = "#d81b2a" if int((t - 16.0) // 2.0) % 2 == 0 else "#1b3fd8"
        elif ph == "STATIC":
            bg = "#102418"
        else:
            bg = "#0b1016"
        canvas.configure(bg=bg)
        canvas.delete("all")

        lt = time.localtime(now)
        clock = time.strftime("%H:%M:%S", lt) + f".{int((now % 1) * 1000):03d}"
        canvas.create_text(W / 2, 62, text=clock, fill="#f2f6fa",
                           font=("Consolas", 52, "bold"))
        canvas.create_text(W / 2, 118, text="RETINA STREAM / LIVE", fill="#8fa6bd",
                           font=("Consolas", 15))

        ball_y = H * 0.50
        if ph == "MOTION":
            for k in range(4, 0, -1):
                tt = t - k * 0.045 + 40
                nx2 = ball_nx(tt)
                bx2 = W * 0.10 + nx2 * W * 0.80
                r2 = 44 * (1 - k * 0.16)
                canvas.create_oval(bx2 - r2, ball_y - r2 + 8, bx2 + r2, ball_y + r2 + 8,
                                   fill="#4a0d0d", outline="")
            nx = ball_nx(t)
            bx = W * 0.10 + nx * W * 0.80
            canvas.create_oval(bx - 44, ball_y - 44, bx + 44, ball_y + 44,
                               fill="#ff2d2d", outline="")
        elif ph == "STATIC":
            canvas.create_oval(W / 2 - 44, ball_y - 44, W / 2 + 44, ball_y + 44,
                               fill="#ff2d2d", outline="")
        else:
            canvas.create_text(W / 2, H * 0.48,
                               text="A" if int(now / 0.12) % 2 == 0 else "B",
                               fill="#ffffff", font=("Consolas", 96, "bold"))

        # video panel
        ok, frame = cap.read()
        if not ok:
            cap.set(cv2.CAP_PROP_POS_FRAMES, 0)
            ok, frame = cap.read()
        if ok:
            im = Image.fromarray(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)).resize((480, 270))
            photo = ImageTk.PhotoImage(im)
            canvas.create_image(24, H - 24, image=photo, anchor="sw")
            canvas.create_rectangle(24, H - 294, 504, H - 24, outline="#2a3a4a")

        canvas.create_rectangle(0, H - 8, W * (t / CYCLE), H, fill="#7fd1ff", outline="")

        tick["n"] += 1
        if tick["n"] % 25 == 0:
            geom = {"x": root.winfo_rootx(), "y": root.winfo_rooty(),
                    "w": root.winfo_width(), "h": root.winfo_height(),
                    "phase": ph, "t": round(t, 3), "ts_ms": int(now * 1000)}
            GEOM.write_text(json.dumps(geom), encoding="utf-8")
        root.after(40, draw)

    root.after(200, draw)
    root.after(400, lambda: (root.lift(), root.attributes("-topmost", True), root.focus_force()))
    root.mainloop()


if __name__ == "__main__":
    main()
