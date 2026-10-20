"""Launch the dynamic test page in an Edge app window and report its client rect."""
from __future__ import annotations

import ctypes
import ctypes.wintypes as w
import json
import subprocess
import sys
import time
from pathlib import Path

EDGE = r"C:\Program Files (x86)\Microsoft\Edge\Application\msedge.exe"
TITLE = "Retina Dynamic Test"
HERE = Path(__file__).resolve().parent.parent
PAGE = HERE / "artifacts" / "dynamic" / "dynamic_page.html"
PROFILE = HERE / "artifacts" / "dynamic" / "edge-profile"

user32 = ctypes.WinDLL("user32", use_last_error=True)

try:
    ctypes.windll.shcore.SetProcessDpiAwareness(2)
except Exception:
    try:
        ctypes.windll.user32.SetProcessDPIAware()
    except Exception:
        pass

user32.FindWindowW.restype = w.HWND
user32.FindWindowW.argtypes = [w.LPCWSTR, w.LPCWSTR]
user32.EnumWindows.argtypes = [ctypes.WINFUNCTYPE(w.BOOL, w.HWND, w.LPARAM), w.LPARAM]
user32.GetWindowTextW.argtypes = [w.HWND, w.LPWSTR, ctypes.c_int]
user32.IsWindowVisible.argtypes = [w.HWND]
user32.SetWindowPos.argtypes = [w.HWND, w.HWND, ctypes.c_int, ctypes.c_int,
                                ctypes.c_int, ctypes.c_int, ctypes.c_uint]
user32.ShowWindow.argtypes = [w.HWND, ctypes.c_int]
user32.SetForegroundWindow.argtypes = [w.HWND]
user32.GetClientRect.argtypes = [w.HWND, ctypes.POINTER(w.RECT)]
user32.ClientToScreen.argtypes = [w.HWND, ctypes.POINTER(w.POINT)]


def find_window(part: str = TITLE, timeout: float = 20.0):
    deadline = time.time() + timeout
    while time.time() < deadline:
        found = []

        @ctypes.WINFUNCTYPE(w.BOOL, w.HWND, w.LPARAM)
        def cb(hwnd, _):
            if not user32.IsWindowVisible(hwnd):
                return True
            n = user32.GetWindowTextW(hwnd, ctypes.create_unicode_buffer(512), 512)
            if n:
                buf = ctypes.create_unicode_buffer(512)
                user32.GetWindowTextW(hwnd, buf, 512)
                if part in buf.value:
                    found.append((hwnd, buf.value))
            return True

        user32.EnumWindows(cb, 0)
        if found:
            return found[0]
        time.sleep(0.25)
    return None


def client_rect(hwnd):
    r = w.RECT()
    user32.GetClientRect(hwnd, ctypes.byref(r))
    pt = w.POINT(0, 0)
    user32.ClientToScreen(hwnd, ctypes.byref(pt))
    return {"x": pt.x, "y": pt.y, "w": r.right - r.left, "h": r.bottom - r.top}


def main():
    PROFILE.mkdir(parents=True, exist_ok=True)
    url = PAGE.as_uri()
    cmd = [EDGE, f"--app={url}", "--window-size=1200,800", "--window-position=120,80",
           f"--user-data-dir={PROFILE}", "--no-first-run", "--no-default-browser-check"]
    proc = subprocess.Popen(cmd, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
    hit = find_window()
    if not hit:
        print(json.dumps({"ok": False, "error": "window not found", "pid": proc.pid}))
        return 1
    hwnd, title = hit
    user32.ShowWindow(hwnd, 9)  # SW_RESTORE
    user32.SetWindowPos(hwnd, 0, 120, 80, 1200, 800, 0x0040 | 0x0004)  # SHOWWINDOW|NOZORDER
    user32.SetForegroundWindow(hwnd)
    time.sleep(1.2)
    rect = client_rect(hwnd)
    print(json.dumps({"ok": True, "pid": proc.pid, "hwnd": int(hwnd), "title": title,
                      "client": rect, "url": url}, ensure_ascii=False))
    return 0


if __name__ == "__main__":
    sys.exit(main())
