"""Phase 1 闭环验证：客户视觉模型 + 一个“手”完成真实多步 GUI 任务。

链路：RetinaService.observe（DXGI 取帧）→ 视觉模型输出动作 JSON →
本地手执行（tools/hand_local.py）→ 回读屏幕 → 受控 GUI 落盘结果做确定性验收。

    py -3.12 tools/closed_loop.py

这不是产品的一部分：产品只暴露 MCP / WebSocket 标准接口，手由客户自选。
"""
from __future__ import annotations

import argparse
import ctypes
import json
import subprocess
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "tools"))

from hand_local import execute  # noqa: E402
from vision_retina.capture import enable_dpi_awareness  # noqa: E402
from vision_retina.config import load_config  # noqa: E402
from vision_retina.service import RetinaService  # noqa: E402

if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")

user32 = ctypes.WinDLL("user32", use_last_error=True)
user32.FindWindowW.argtypes = [ctypes.c_wchar_p, ctypes.c_wchar_p]
user32.FindWindowW.restype = ctypes.c_void_p
user32.SetForegroundWindow.argtypes = [ctypes.c_void_p]
user32.ShowWindow.argtypes = [ctypes.c_void_p, ctypes.c_int]
user32.PostMessageW.argtypes = [ctypes.c_void_p, ctypes.c_uint, ctypes.c_void_p, ctypes.c_void_p]

RESULT = ROOT / "artifacts" / "gui_result.json"
OUT = ROOT / "artifacts" / "phase1_loop.json"
TITLE = "Retina Phase1 Target"


def _wait_window(timeout_s: float = 12.0) -> int:
    deadline = time.time() + timeout_s
    while time.time() < deadline:
        hwnd = user32.FindWindowW(None, TITLE)
        if hwnd:
            return int(hwnd)
        time.sleep(0.2)
    raise SystemExit("受控 GUI 窗口没有出现")


def _read_result() -> dict | None:
    if not RESULT.exists():
        return None
    try:
        return json.loads(RESULT.read_text(encoding="utf-8"))
    except Exception:  # noqa: BLE001
        return None


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--expect", default="demo")
    parser.add_argument("--max-steps", type=int, default=8)
    parser.add_argument("--task", default=None)
    parser.add_argument("--keep-open", action="store_true")
    args = parser.parse_args()

    enable_dpi_awareness()
    RESULT.parent.mkdir(parents=True, exist_ok=True)
    for path in (RESULT, OUT):
        if path.exists():
            path.unlink()

    creation = getattr(subprocess, "DETACHED_PROCESS", 0) | getattr(subprocess, "CREATE_NEW_PROCESS_GROUP", 0)
    app = subprocess.Popen(
        [sys.executable, str(ROOT / "tools" / "gui_target.py")],
        stdin=subprocess.DEVNULL,
        stdout=subprocess.DEVNULL,
        stderr=subprocess.DEVNULL,
        creationflags=creation,
    )
    task = args.task or (
        "屏幕最前方有一个标题为 Retina Phase1 Target 的窗口，里面有一个 User name 输入框和一个 Submit 按钮。"
        f"请在这个输入框里填入 {args.expect}（若输入框里已有内容，先按 Ctrl+A 再输入），"
        "然后点击 Submit 按钮。看到提交成功的提示后，返回 {\"action\":\"observe\"} 表示完成。"
    )

    steps: list[dict] = []
    history: list[str] = []
    verdict = False
    error = None
    t0 = time.time()

    try:
        _wait_window()
        time.sleep(1.0)
        hwnd = user32.FindWindowW(None, TITLE)
        user32.ShowWindow(hwnd, 9)  # SW_RESTORE
        user32.SetForegroundWindow(hwnd)
        time.sleep(0.5)

        cfg = load_config()
        with RetinaService(cfg) as svc:
            for index in range(1, args.max_steps + 1):
                observed = svc.observe(include_image=True, force_keyframe=True)
                frame = observed.observation.frame
                # 原生分辨率，避免模型把“图片坐标”当屏幕坐标
                cfg.data.setdefault("encode", {})["max_width"] = int(frame.width)

                reply = svc.brain.ask_actions(frame, task, history=history[-6:])
                action = reply.action
                record = {
                    "step": index,
                    "frame_id": frame.frame_id,
                    "frame_size": [frame.width, frame.height],
                    "latency_ms": reply.latency_ms,
                    "usage": reply.usage,
                    "finish_reason": reply.finish_reason,
                    "raw": (reply.text or "")[:600],
                    "action": action,
                }
                steps.append(record)
                print(f"step {index}: {json.dumps(action, ensure_ascii=False)} ({reply.latency_ms} ms)")

                if action is None:
                    history.append(f"step {index}: 未返回动作 JSON，请严格只输出一个动作 JSON")
                    if sum(1 for s in steps if s["action"] is None) >= 2:
                        break
                    continue

                name = str(action.get("action", ""))
                if name == "observe":
                    verdict = True
                    break

                executed = execute(action)
                record["executed"] = executed
                history.append(f"step {index}: 已执行 {name} {json.dumps(action.get('target') or {}, ensure_ascii=False)}")
                time.sleep(0.7)

                result = _read_result()
                if result and str(result.get("value", "")).strip() == args.expect:
                    record["gui_result"] = result
                    verdict = True
                    break

            result = _read_result()
            if result is None:
                # 最后再给模型一次机会判断是否完成（不执行动作）
                pass
            else:
                verdict = verdict or str(result.get("value", "")).strip() == args.expect
            summary = {
                "verdict": "PASS" if verdict else "FAIL",
                "expected": args.expect,
                "gui_result": result,
                "steps": steps,
                "total_latency_ms": round((time.time() - t0) * 1000),
                "total_tokens": sum(
                    int((s.get("usage") or {}).get("total_tokens") or 0) for s in steps
                ),
                "service_status": svc.status(),
            }
    except Exception as exc:  # noqa: BLE001
        error = f"{type(exc).__name__}: {exc}"
        summary = {"verdict": "ERROR", "error": error, "steps": steps}
        verdict = False
    finally:
        if not args.keep_open:
            hwnd = user32.FindWindowW(None, TITLE)
            if hwnd:
                user32.PostMessageW(hwnd, 0x0010, 0, 0)  # WM_CLOSE
            time.sleep(0.3)
            if app.poll() is None:
                app.terminate()

    OUT.write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8")
    print("=" * 60)
    print(f"verdict={summary['verdict']} steps={len(steps)} total={summary.get('total_latency_ms')}ms tokens={summary.get('total_tokens')}")
    print(f"gui_result={json.dumps(summary.get('gui_result'), ensure_ascii=False)}")
    print(f"report: {OUT}")
    return 0 if verdict else 1


if __name__ == "__main__":
    raise SystemExit(main())
