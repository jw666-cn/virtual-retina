"""最小本地“手”执行器（Phase 1 闭环测试替身，不是产品的一部分）。

产品只提供标准接口（MCP / WebSocket JSON），具体用哪只手由客户选择。
本文件只是把模型输出的动作 JSON 落到本机鼠标键盘上，用来验证闭环。

    from tools.hand_local import execute
    execute({"action": "click", "target": {"x": 900, "y": 700}})
"""
from __future__ import annotations

import ctypes
import time
from ctypes import wintypes as wt
from typing import Any

user32 = ctypes.WinDLL("user32", use_last_error=True)
ULONG_PTR = ctypes.c_ulonglong

# 坐标必须是物理像素：若调用方还没声明 DPI 感知，这里补上（重复设置会失败，忽略即可）。
try:
    ctypes.windll.shcore.SetProcessDpiAwareness(2)
except Exception:  # noqa: BLE001
    try:
        ctypes.windll.user32.SetProcessDPIAware()
    except Exception:  # noqa: BLE001
        pass

INPUT_MOUSE = 0
INPUT_KEYBOARD = 1
KEYEVENTF_KEYUP = 0x0002
KEYEVENTF_UNICODE = 0x0004
MOUSEEVENTF_LEFTDOWN = 0x0002
MOUSEEVENTF_LEFTUP = 0x0004
MOUSEEVENTF_RIGHTDOWN = 0x0008
MOUSEEVENTF_RIGHTUP = 0x0010
MOUSEEVENTF_WHEEL = 0x0800


class MOUSEINPUT(ctypes.Structure):
    _fields_ = [
        ("dx", wt.LONG),
        ("dy", wt.LONG),
        ("mouseData", wt.DWORD),
        ("dwFlags", wt.DWORD),
        ("time", wt.DWORD),
        ("dwExtraInfo", ULONG_PTR),
    ]


class KEYBDINPUT(ctypes.Structure):
    _fields_ = [
        ("wVk", wt.WORD),
        ("wScan", wt.WORD),
        ("dwFlags", wt.DWORD),
        ("time", wt.DWORD),
        ("dwExtraInfo", ULONG_PTR),
    ]


class HARDWAREINPUT(ctypes.Structure):
    _fields_ = [("uMsg", wt.DWORD), ("wParamL", wt.WORD), ("wParamH", wt.WORD)]


class _INPUTUNION(ctypes.Union):
    _fields_ = [("mi", MOUSEINPUT), ("ki", KEYBDINPUT), ("hi", HARDWAREINPUT)]


class INPUT(ctypes.Structure):
    _anonymous_ = ("u",)
    _fields_ = [("type", wt.DWORD), ("u", _INPUTUNION)]


user32.SendInput.argtypes = [wt.UINT, ctypes.POINTER(INPUT), ctypes.c_int]
user32.SendInput.restype = wt.UINT
user32.SetCursorPos.argtypes = [ctypes.c_int, ctypes.c_int]
user32.SetCursorPos.restype = wt.BOOL
user32.GetSystemMetrics.argtypes = [ctypes.c_int]
user32.GetSystemMetrics.restype = ctypes.c_int

VK = {
    "ctrl": 0x11,
    "control": 0x11,
    "alt": 0x12,
    "shift": 0x10,
    "win": 0x5B,
    "enter": 0x0D,
    "return": 0x0D,
    "tab": 0x09,
    "esc": 0x1B,
    "escape": 0x1B,
    "space": 0x20,
    "backspace": 0x08,
    "delete": 0x2E,
    "del": 0x2E,
    "home": 0x24,
    "end": 0x23,
    "left": 0x25,
    "up": 0x26,
    "right": 0x27,
    "down": 0x28,
    "pageup": 0x21,
    "pagedown": 0x22,
    "f1": 0x70,
    "f2": 0x71,
    "f3": 0x72,
    "f4": 0x73,
    "f5": 0x74,
}


def _send(*inputs: INPUT) -> None:
    array = (INPUT * len(inputs))(*inputs)
    sent = user32.SendInput(len(inputs), array, ctypes.sizeof(INPUT))
    if sent != len(inputs):
        raise ctypes.WinError(ctypes.get_last_error())


def _mouse(flags: int, data: int = 0) -> INPUT:
    return INPUT(type=INPUT_MOUSE, mi=MOUSEINPUT(0, 0, data, flags, 0, 0))


def _key(vk: int, flags: int = 0) -> INPUT:
    return INPUT(type=INPUT_KEYBOARD, ki=KEYBDINPUT(wVk=vk, wScan=0, dwFlags=flags, time=0, dwExtraInfo=0))


def _unicode(ch: str, flags: int = 0) -> INPUT:
    return INPUT(type=INPUT_KEYBOARD, ki=KEYBDINPUT(wVk=0, wScan=ord(ch), dwFlags=KEYEVENTF_UNICODE | flags, time=0, dwExtraInfo=0))


def move(x: int, y: int) -> None:
    user32.SetCursorPos(int(x), int(y))
    time.sleep(0.03)


def click(x: int | None = None, y: int | None = None, *, button: str = "left", double: bool = False) -> None:
    if x is not None and y is not None:
        move(x, y)
    down, up = (MOUSEEVENTF_RIGHTDOWN, MOUSEEVENTF_RIGHTUP) if button == "right" else (MOUSEEVENTF_LEFTDOWN, MOUSEEVENTF_LEFTUP)
    times = 2 if double else 1
    for _ in range(times):
        _send(_mouse(down), _mouse(up))
        time.sleep(0.08)


def type_text(text: str, interval: float = 0.012) -> None:
    for ch in text:
        if ch == "\n":
            _send(_key(VK["enter"]), _key(VK["enter"], KEYEVENTF_KEYUP))
        elif ch == "\t":
            _send(_key(VK["tab"]), _key(VK["tab"], KEYEVENTF_KEYUP))
        else:
            _send(_unicode(ch), _unicode(ch, KEYEVENTF_KEYUP))
        time.sleep(interval)


def press(key: str) -> None:
    parts = [p.strip().lower() for p in str(key).split("+") if p.strip()]
    if not parts:
        return
    modifiers = [VK[p] for p in parts[:-1] if p in VK]
    main = parts[-1]
    vk = VK.get(main, ord(main.upper()) if len(main) == 1 else 0)
    if not vk:
        raise ValueError(f"未知按键：{key}")
    for mod in modifiers:
        _send(_key(mod))
    _send(_key(vk), _key(vk, KEYEVENTF_KEYUP))
    for mod in reversed(modifiers):
        _send(_key(mod, KEYEVENTF_KEYUP))


def scroll(dx: int = 0, dy: int = -3) -> None:
    steps = int(dy)
    for _ in range(abs(steps)):
        _send(_mouse(MOUSEEVENTF_WHEEL, 120 if steps > 0 else -120))
        time.sleep(0.03)
    if dx:
        # 水平滚动在多数应用里不常用，用 shift+滚轮近似。
        _send(_key(VK["shift"]))
        for _ in range(abs(int(dx))):
            _send(_mouse(MOUSEEVENTF_WHEEL, 120 if dx > 0 else -120))
        _send(_key(VK["shift"], KEYEVENTF_KEYUP))


def drag(x1: int, y1: int, x2: int, y2: int, *, steps: int = 12) -> None:
    move(x1, y1)
    _send(_mouse(MOUSEEVENTF_LEFTDOWN))
    for i in range(1, steps + 1):
        move(int(x1 + (x2 - x1) * i / steps), int(y1 + (y2 - y1) * i / steps))
        time.sleep(0.02)
    _send(_mouse(MOUSEEVENTF_LEFTUP))


def execute(action: dict[str, Any]) -> dict[str, Any]:
    """执行任务书 §4.2 的动作 JSON，返回执行记录。"""
    name = str(action.get("action", "")).strip()
    target = action.get("target") or {}
    params = action.get("params") or {}
    x = target.get("x")
    y = target.get("y")

    if name == "click":
        click(int(x), int(y))
    elif name == "double_click":
        click(int(x), int(y), double=True)
    elif name == "right_click":
        click(int(x), int(y), button="right")
    elif name == "type":
        if x is not None and y is not None:
            click(int(x), int(y))
            time.sleep(0.15)
        type_text(str(params.get("text", "")))
    elif name == "key":
        press(str(params.get("key", "")))
    elif name == "scroll":
        scroll(int(params.get("dx", 0) or 0), int(params.get("dy", -3) or -3))
    elif name == "drag":
        start = target if ("x" in target and "y" in target) else params.get("from", {})
        end = params.get("to", {})
        drag(int(start.get("x", 0)), int(start.get("y", 0)), int(end.get("x", 0)), int(end.get("y", 0)))
    elif name == "wait":
        time.sleep(float(params.get("seconds", 0.5)))
    elif name == "observe":
        pass
    else:
        raise ValueError(f"不支持的 action：{name!r}")

    return {"action": name, "target": target, "params": params, "executed_at": int(time.time() * 1000)}
