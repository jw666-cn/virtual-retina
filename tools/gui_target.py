"""Phase 1 闭环测试用的受控 GUI 目标（不是产品的一部分）。

窗口里有一个文本输入框和一个 Submit 按钮；点击 Submit 后把输入内容写到
artifacts/gui_result.json，供闭环脚本做**确定性**验收（不靠模型自述）。

    py -3.12 tools/gui_target.py
"""
from __future__ import annotations

import ctypes
import json
import os
import time
import tkinter as tk
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
OUT = ROOT / "artifacts" / "gui_result.json"
GEOM = ROOT / "artifacts" / "gui_geom.json"

# 让 Tk 坐标 == 屏幕物理像素，避免 150% DPI 缩放带来的坐标歧义。
try:
    ctypes.windll.shcore.SetProcessDpiAwareness(2)
except Exception:  # noqa: BLE001
    try:
        ctypes.windll.user32.SetProcessDPIAware()
    except Exception:  # noqa: BLE001
        pass

TITLE = "Retina Phase1 Target"
EXPECTED_HINT = "在输入框里填入 demo，然后点击 Submit"


def main() -> None:
    OUT.parent.mkdir(parents=True, exist_ok=True)
    if OUT.exists():
        OUT.unlink()

    root = tk.Tk()
    root.title(TITLE)
    root.geometry("920x520+260+180")
    root.configure(bg="#f2f4f8")
    try:
        root.attributes("-topmost", True)
    except tk.TclError:
        pass

    tk.Label(root, text="Phase 1 Target", font=("Microsoft YaHei UI", 30, "bold"), bg="#f2f4f8", fg="#1b2430").pack(pady=(28, 6))
    tk.Label(root, text=EXPECTED_HINT, font=("Microsoft YaHei UI", 16), bg="#f2f4f8", fg="#4a5568").pack(pady=(0, 22))

    card = tk.Frame(root, bg="#ffffff", highlightbackground="#d7dce5", highlightthickness=2)
    card.pack(padx=60, pady=6, fill="x")

    tk.Label(card, text="User name", font=("Microsoft YaHei UI", 17), bg="#ffffff", fg="#1b2430").pack(anchor="w", padx=24, pady=(18, 4))
    entry = tk.Entry(card, font=("Consolas", 22), width=26, relief="solid", bd=2)
    entry.pack(anchor="w", padx=24, ipady=6)

    status = tk.Label(card, text="状态：等待提交", font=("Microsoft YaHei UI", 15), bg="#ffffff", fg="#8a94a6")
    status.pack(anchor="w", padx=24, pady=(10, 2))

    def submit() -> None:
        value = entry.get().strip()
        payload = {
            "title": TITLE,
            "value": value,
            "ok": bool(value),
            "submitted_at": int(time.time() * 1000),
            "pid": os.getpid(),
        }
        tmp = OUT.with_suffix(".tmp")
        tmp.write_text(json.dumps(payload, ensure_ascii=False), encoding="utf-8")
        tmp.replace(OUT)
        status.configure(text=f"状态：提交成功（{value or '空'}）", fg="#1f9d55")
        root.title(f"{TITLE} - DONE")
        print(json.dumps(payload, ensure_ascii=False), flush=True)

    button = tk.Button(
        card,
        text="Submit",
        font=("Microsoft YaHei UI", 20, "bold"),
        bg="#2563eb",
        fg="#ffffff",
        activebackground="#1d4ed8",
        activeforeground="#ffffff",
        relief="flat",
        padx=44,
        pady=10,
        command=submit,
    )
    button.pack(anchor="w", padx=24, pady=(12, 24))

    def dump_geometry() -> None:
        root.update_idletasks()

        def box(widget: tk.Widget) -> dict[str, int]:
            x, y = widget.winfo_rootx(), widget.winfo_rooty()
            w, h = widget.winfo_width(), widget.winfo_height()
            return {"x": x, "y": y, "w": w, "h": h, "cx": x + w // 2, "cy": y + h // 2}

        payload = {"window": box(root), "entry": box(entry), "button": box(button), "dpi_aware": True}
        GEOM.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")
        print(json.dumps(payload, ensure_ascii=False), flush=True)

    root.after(400, dump_geometry)
    root.after(180_000, root.destroy)
    root.mainloop()


if __name__ == "__main__":
    main()
