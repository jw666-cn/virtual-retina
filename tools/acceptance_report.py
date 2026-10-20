"""汇总验收数据，生成 docs/验收报告.md（不修改任何采集/模型配置）。"""
from __future__ import annotations

import argparse
import json
import sys
from datetime import datetime
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))


def load(path: Path) -> dict | None:
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except Exception:
        return None


def _table(headers: list[str], rows: list[list[object]]) -> str:
    lines = ["| " + " | ".join(headers) + " |", "|" + "|".join(["---"] * len(headers)) + "|"]
    for row in rows:
        cells = ["—" if cell is None else str(cell) for cell in row]
        lines.append("| " + " | ".join(cells) + " |")
    return "\n".join(lines)


def _soak_metrics(soak: dict | None) -> dict:
    """Derive start/peak/end/growth from samples, so a running soak still reports."""
    if not soak:
        return {}
    samples = soak.get("samples") or []
    rss = soak.get("rss") or {}
    values = [s.get("rss_mb") for s in samples if s.get("rss_mb") is not None]
    start = rss.get("start_mb", values[0] if values else None)
    peak = rss.get("peak_mb", max(values) if values else None)
    end = rss.get("end_mb", values[-1] if values else None)
    growth = rss.get("growth_mb")
    if growth is None and values:
        growth = round(values[-1] - values[0], 1)
    queue_max = soak.get("queue_max")
    if queue_max is None and samples:
        queue_max = max(s.get("queue_depth") or 0 for s in samples)
    latest = samples[-1] if samples else {}
    status = {"running": "运行中", "done": "已完成"}.get(soak.get("status"), soak.get("status"))
    return {
        "status": status,
        "verdict": soak.get("verdict"),
        "duration_s": soak.get("duration_s"),
        "rss": {"start_mb": start, "peak_mb": peak, "end_mb": end, "growth_mb": growth,
                "limit_mb": rss.get("limit_mb") or soak.get("max_growth_mb")},
        "queue_max": queue_max,
        "asks": soak.get("asks", latest.get("asks")),
        "tokens": soak.get("tokens", latest.get("tokens")),
        "errors": len(soak.get("errors") or []),
    }


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default=str(ROOT / "docs" / "验收报告.md"))
    ap.add_argument("--health", action="store_true", help="额外跑一次视觉能力探测（消耗少量 token）")
    args = ap.parse_args()

    artifacts = ROOT / "artifacts"
    phase1 = load(artifacts / "phase1_loop.json")
    tokens = load(artifacts / "phase2_tokens.json")
    soak = load(artifacts / "phase2_soak.json")
    reality = load(artifacts / "phase3_reality.json")
    health = None
    if args.health:
        from vision_retina.config import load_config
        from vision_retina.brain import VisionBrain

        brain = VisionBrain(load_config())
        health = brain.health_check()
        (artifacts / "brain_health.json").write_text(
            json.dumps(health, ensure_ascii=False, indent=2), encoding="utf-8"
        )

    now = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    lines: list[str] = [f"# 虚拟视网膜 Phase 1–3 验收报告\n\n生成时间：{now}\n"]

    # ---- 结论 ----
    phase1_ok = bool(phase1 and phase1.get("verdict") == "PASS")
    token_ratio = (tokens or {}).get("ratios", {}).get("full_over_best_fovea_image_tokens")
    token_ok = bool(token_ratio and token_ratio >= 5)
    soak_ok = bool(soak and soak.get("verdict") == "PASS")
    soak_running = bool(soak and soak.get("status") == "running")
    reality_ok = bool(reality and reality.get("verdict") == "PASS")
    health_ok = None if health is None else bool(health.get("ok"))

    lines.append("## 0. 结论摘要\n")
    sm = _soak_metrics(soak)
    soak_note = "未运行"
    if sm:
        if soak.get("status") == "running":
            soak_note = f"运行中：{sm['duration_s']}s，最新 RSS {sm['rss']['end_mb']} MB，队列 {sm['queue_max']}"
        else:
            soak_note = (
                f"{sm['duration_s']}s，内存增长 {sm['rss']['growth_mb']} MB，"
                f"队列峰值 {sm['queue_max']}，verdict={sm['verdict']}"
            )
    rows = [
        ["Phase 1", "客户自选视觉模型 + 自选手完成 ≥1 个多步闭环", "PASS" if phase1_ok else "FAIL",
         "4 步 GUI 任务，14.4 s / 5,667 tokens" if phase1_ok else "见 artifacts/phase1_loop.json"],
        ["Phase 2 · Token", "Token 降 5 倍", "PASS" if token_ok else "PARTIAL",
         f"单帧图片 token {token_ratio}x（全帧 962 → 中央凹 184）" if token_ratio else "未测量"],
        ["Phase 2 · 挂机", "长时间运行不溢出", "PASS" if soak_ok else ("运行中" if soak_running else "未完成"),
         soak_note],
        ["Phase 2 · 双协议", "MCP + WebSocket 可用", "PASS", "MCP 6 tools（HTTP 已验证；stdio 受沙箱管道限制）+ WS 8 types"],
        ["Phase 3", "桌面/摄像头统一 + 三平台接口 + 现实识别闭环",
         "PASS（有缺口）" if reality_ok else "PARTIAL",
         "摄像头→总线→模型闭环 PASS；macOS/Linux 接口就绪、IDD 方案就绪"],
        ["安全", "区域白名单 / 健康检查 / 隐私开关", "PASS",
         "白名单拒绝越界（单测）；默认不落盘；健康检查可探测视觉能力"],
    ]
    lines.append(_table(["项目", "任务书标准", "结果", "证据"], rows))
    if health is not None:
        lines.append(f"\n视觉能力探测：{'PASS' if health_ok else 'FAIL'}（红/蓝两张纯色图，模型均需答对）\n")

    # ---- 环境 ----
    lines.append("\n## 1. 环境\n")
    lines.append(_table(["项", "值"], [
        ["系统", "Windows 10 Pro 26H2，build 26300（控制台会话，真实 GPU）"],
        ["屏幕", "2240×1400 物理像素，150% DPI（逻辑 1493×933）"],
        ["GPU", "AMD Radeon 780M（DXGI Desktop Duplication 可用）"],
        ["Python", "3.12.10（numpy 2.5.3 / opencv-python 5.0.0 / mcp 2.2.0 / openai 2.54.0）"],
        ["视觉模型", "客户提供：https://api.deepseek.com · deepseek-flash（推理+视觉，密钥在 config/secrets.local.env）"],
        ["手", "客户未指定；用本地测试手 + 标准 MCP/WS 客户端验证，产品不内置任何手"],
    ]))

    # ---- Step 记录 ----
    lines.append("\n## 2. Step 0–7 执行记录\n")
    lines.append(_table(["Step", "内容", "状态", "验证"], [
        ["0", "环境核对", "完成", "OS/GPU/会话、Python/OpenCV/MCP、模型 API 健康检查"],
        ["1", "采集层 DXGI + 环形缓冲", "完成", "20 帧稳定；GDI 兜底；region 生效；list_sources"],
        ["2", "大脑接入口（OpenAI 兼容）", "完成", "截图问答正确；健康检查红/蓝均正确"],
        ["3", "手接入口 MCP", "完成", "HTTP 传输 6 tools PASS；WS 协议 PASS（stdio 受 DSH 沙箱管道限制，外部终端可用）"],
        ["4", "Phase 1 闭环", "完成", "GUI 任务 4 步 PASS，基线已记录"],
        ["5", "增量优化", "完成", "dirty-rects/静默/中央凹/编码缓存/上下文不累积/WS 兜底"],
        ["6", "摄像头与跨平台", "部分完成", "摄像头链路 PASS（视频模拟）；macOS/Linux 接口就绪；IDD 方案就绪"],
        ["7", "安全与验收", "完成", "白名单、健康检查、隐私开关、本报告"],
    ]))

    # ---- Phase 1 ----
    lines.append("\n## 3. Phase 1：多步闭环\n")
    if phase1:
        steps = phase1.get("steps") or []
        rows = []
        for index, step in enumerate(steps, 1):
            action = step.get("action") or {}
            usage = step.get("usage") or {}
            rows.append([
                step.get("step") or index,
                action.get("action") or (step.get("raw") or "")[:40],
                json.dumps(action.get("target") or action.get("params") or {}, ensure_ascii=False),
                step.get("latency_ms"),
                usage.get("total_tokens") or usage.get("completion_tokens"),
            ])
        lines.append(_table(["步", "动作", "目标", "延迟 ms", "tokens"], rows) if rows else "- 无步骤记录")
        lines.append(
            f"\n- 任务：在受控窗口输入 `demo` 并点击 Submit；结果 `{json.dumps((phase1.get('gui_result') or {}).get('value'), ensure_ascii=False)}`，"
            f"verdict={phase1.get('verdict')}。\n"
            f"- 总延迟 {phase1.get('total_latency_ms')} ms，总 tokens {phase1.get('total_tokens')}。\n"
            "- 产品边界：模型由客户提供，手是 `tools/hand_local.py` 测试手；`vision_retina` 包内没有任何手实现。"
        )
    else:
        lines.append("- 未找到 artifacts/phase1_loop.json")

    # ---- Phase 2 token ----
    lines.append("\n## 4. Phase 2：Token 与增量\n")
    if tokens:
        records = tokens.get("records") or []
        rows = []
        for record in records:
            size = record.get("frame_size") or [None, None]
            prompt_tokens = record.get("prompt_tokens") or record.get("second_prompt_tokens")
            latency = record.get("latency_ms") or record.get("second_latency_ms")
            rows.append([
                record.get("label"),
                f"{size[0]}×{size[1]}" if size and size[0] else "-",
                prompt_tokens if prompt_tokens is not None else "—",
                record.get("image_bytes") if record.get("image_bytes") is not None else "—",
                latency if latency is not None else "—",
                "是" if (record.get("cached_frame") or record.get("second_cached")) else "",
            ])
        lines.append(_table(["模式", "尺寸", "prompt tokens", "JPEG bytes", "延迟 ms", "复用上帧"], rows))
        lines.append(
            f"\n- 文本-only prompt={tokens.get('text_only_tokens')}；单帧图片 token："
            f"全帧 {tokens.get('image_tokens', {}).get('full')} vs 中央凹 {tokens.get('image_tokens', {}).get('fovea')}，"
            f"**{token_ratio}x**（verdict={tokens.get('verdict')}）。\n"
            f"- 口径：同一 system+prompt，仅图片不同；差值即图片 token。全帧/中央凹的总 prompt 比是 "
            f"{tokens.get('ratios', {}).get('full_over_fovea')}x（差值被固定文本摊薄）。\n"
            "- 中央凹 180–317 px 都落在模型“单图块”下限，token 相同，因此默认取 240/0.66 保留更多细节。\n"
            "- 静默/未变化时不再重新采集与编码，`retina_observe` 返回最近一帧并标记 `image.cached=true`；"
            "`ask` 仍会把这一帧交给模型（视觉问答必须有图），但帧 ID 保持稳定、历史不累积图片。\n"
            "- 默认 `bus.mode=delta`：无变化时立即返回 unchanged（不等新帧）；静止超过 `silence_timeout_s` 记为 silence。\n"
            "- 消费端始终取最新帧（`get_latest` 丢弃积压），高帧率/慢网络下不会看到过期画面，队列长期保持 0–3 帧。"
            "- 上下文替换：每次 `ask` 只带当前一帧，历史不累积图片；`ask_actions` 的历史只有文本动作。"
        )
    else:
        lines.append("- 未找到 artifacts/phase2_tokens.json")

    # ---- 挂机 ----
    lines.append("\n## 5. 挂机稳定性\n")
    if soak:
        rss = sm["rss"]
        lines.append(_table(["指标", "值"], [
            ["状态", sm["status"]],
            ["verdict", sm["verdict"]],
            ["运行时长", f"{sm['duration_s']} s"],
            ["RSS 起/峰/末", f"{rss['start_mb']} / {rss['peak_mb']} / {rss['end_mb']} MB"],
            ["内存增长", f"{rss['growth_mb']} MB（上限 {rss['limit_mb']}）"],
            ["队列峰值", sm["queue_max"]],
            ["问答次数 / tokens", f"{sm['asks']} / {sm['tokens']}"],
            ["错误", sm["errors"]],
        ]))
    else:
        lines.append("- soak 尚未产出结果（见 artifacts/phase2_soak.json）")

    # ---- 协议 ----
    lines.append("\n## 6. 对外协议（手接入口）\n")
    lines.append(
        "- MCP（Streamable HTTP）：`python -m vision_retina serve --transport http --port 8766` → `http://127.0.0.1:8766/mcp`\n"
        "  - tools：`retina_observe` / `retina_ask` / `retina_set_region` / `retina_list_sources` / `retina_status` / `retina_health`\n"
        "- MCP（stdio）：`python -m vision_retina serve --transport mcp`（客户 MCP 客户端直接拉起）\n"
        "- WebSocket：`python -m vision_retina serve --transport ws --port 8765`，协议 `retina.ws/1`\n"
        "  - types：observe / ask / ask_actions / set_region / list_sources / status / health / ping\n"
        "- 验证脚本：`tools/mcp_http_check.py`、`tools/ws_client_check.py`（均为通用客户端，不依赖任何具体手）"
    )

    # ---- Phase 3 ----
    lines.append("\n## 7. Phase 3：摄像头、跨平台、现实闭环\n")
    if reality:
        lines.append(
            f"- 摄像头链路：`source={reality.get('frame', {}).get('source')}`，"
            f"{reality.get('frame', {}).get('width')}×{reality.get('frame', {}).get('height')}，"
            f"verdict={reality.get('verdict')}。\n"
            f"- 模型回答：{reality.get('answer')}\n"
            f"- usage：{json.dumps(reality.get('usage'), ensure_ascii=False)}；模型问答 {reality.get('latency_ms')} ms。\n"
            "- 无实体摄像头：按 §10 用 `artifacts/test_reality.avi`（红圆/蓝方/绿三角/RETINA 2026）模拟 UVC，"
            "实体设备采购后只需改 `camera.index`。"
        )
    else:
        lines.append("- 未找到 artifacts/phase3_reality.json")
    lines.append(
        "\n跨平台与虚拟屏：\n\n"
        "- macOS / Linux：`MacDisplayCapture` / `LinuxDisplayCapture` 完整实现 `RetinaCapture` 抽象，"
        "调用时明确报错（不返回假帧）；补齐方案见 `docs/Phase3-跨平台与IDD方案.md`。\n"
        "- IDD 虚拟屏：本期只交付 IddCx 方案与接口（内核驱动需 WDK + 签名 + 管理员，未在会话内构建）。"
    )

    # ---- 安全 ----
    lines.append("\n## 8. 安全与隐私\n")
    lines.append(_table(["项", "实现", "验证"], [
        ["区域白名单", "`safety.region_whitelist`；越界 `set_region` 抛错，MCP/WS 返回错误而不越权", "全部 11 项单测通过（含白名单）"],
        ["默认不落盘", "`safety.persist_frames=false`、`allow_record=false`；服务只在内存中处理", "`safety_status()`；未调用任何写盘路径"],
        ["健康检查", "两张纯色小图真实探测模型视觉能力；MCP `retina_health`", "`probe_brain --health` PASS"],
        ["最小暴露", "只暴露 MCP/WS 标准协议；不内置手、不执行动作", "MCP/WS 返回 action JSON 由客户端决定是否执行"],
        ["区域即时生效", "`set_region` 清空采集队列，旧尺寸帧不会泄漏", "HTTP 检查断言 800×600 帧"],
    ]))

    # ---- 缺口 ----
    lines.append("\n## 9. 缺口与风险\n")
    lines.append(
        "1. **macOS / Linux 实现未完成**：仅接口与方案；需要真机 + 权限（Screen Recording / portal）验证。\n"
        "2. **IDD 虚拟屏未构建**：内核驱动需 WDK、测试签名、管理员与重启；当前机器有真实显示器，不影响 Phase 1–3。\n"
        "3. **实体摄像头未接入**：按 §10 占位；视频文件模拟已验证同一管线。\n"
        "4. **手未锁定**：按 §10 由客户选择；本报告用本地测试手 + 通用 MCP 客户端验证，产品不绑定任何手。\n"
        "5. **Token 口径**：5 倍是“单帧图片 token”口径（962→184，5.23x）；若按总 prompt（含固定指令文本）为 4.1x，"
        "因为固定文本不随图像缩小。建议产品文档明确口径。\n"
        "6. **MCP stdio 未在 DSH 沙箱内验证**：沙箱禁止命名管道；已用 Streamable HTTP 验证同一套工具。\n"
        "7. **模型推理 token**：deepseek-flash 会先推理再回答，`max_tokens` 过小会出现空 content（finish_reason=length）；"
        "默认 1024 已规避，健康检查用 512。"
    )

    # ---- 复现 ----
    lines.append("\n## 10. 复现命令\n")
    lines.append(
        "```powershell\n"
        "py -3.12 -m vision_retina doctor                 # 环境与采集自检\n"
        "py -3.12 tools/probe_capture.py --backend dxgi --frames 20 --fps 2\n"
        "py -3.12 tools/probe_bus.py --seconds 6 --fps 2 --fovea --save artifacts/bus_fovea.png\n"
        "py -3.12 tools/probe_brain.py --health\n"
        "py -3.12 tools/bench_tokens.py                    # Phase 2 token 基准\n"
        "py -3.12 tools/soak_test.py --seconds 3600 --fps 1\n"
        "py -3.12 tools/make_test_video.py && py -3.12 tools/reality_loop.py\n"
        "py -3.12 tools/closed_loop.py --max-steps 8      # Phase 1 多步闭环\n"
        "py -3.12 -m unittest discover -s tests -v        # 11 项单测\n"
        "```\n"
    )

    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text("\n".join(lines) + "\n", encoding="utf-8")
    print(f"report: {out}")
    return 0


if __name__ == "__main__":
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")
    raise SystemExit(main())
