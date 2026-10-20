from pathlib import Path
import json

base = Path("artifacts/dynamic")
data = json.loads((base/"summary.json").read_text(encoding="utf-8"))
rows = data["rows"]

def table(mode):
    out = ["| 题 | 类型 | 判定 | prompt tokens | 端到端 ms | 帧龄 ms | 脏区占比 | 模型回答 |",
           "|---|---|---|---|---|---|---|---|"]
    for r in rows[mode]:
        ans = (r["answer"] or "").replace("\n", " ").replace("|", "/")
        if len(ans) > 46:
            ans = ans[:46] + "…"
        ok = "✅" if r["correct"] else ("❌" if r["correct"] is False else "—")
        out.append(f"| {r['qid']} | {r['kind']} | {ok} | {r['ptok']} | {r['e2e_ms']} | {r['age_ms']} | {r['dirty']} | {ans} |")
    return "\n".join(out)

F = data["full"]["summary"]; D = data["delta"]["summary"]
single_full = F["single_image_prompt_tokens"] / F["single_image_questions"]
single_delta = D["single_image_prompt_tokens"] / D["single_image_questions"]
save = (1 - D["total_prompt_tokens"] / F["total_prompt_tokens"]) * 100
save_single = (1 - D["single_image_prompt_tokens"] / F["single_image_prompt_tokens"]) * 100

md = f'''# 虚拟视网膜 · 动态画面实时性测试报告

生成时间：2026-10-19 21:35（Asia/Shanghai）
测试对象：`vision_retina`（DXGI 采集 + 总线 + `VisionBrain`）
视觉模型：`deepseek-flash`（客户配置，OpenAI 兼容 `/chat/completions`）
屏幕/窗口：2240×1400 物理像素；受控窗口 1200×800 @ (131,125)，DXGI 区域采集
测试脚本：`tools/dynamic_window.py`（动态源）、`tools/dynamic_test.py`（提问与判定）
原始数据：`artifacts/dynamic/full.json`、`delta.json`、`summary.json`

## 1. 测试方法

- 动态源是一个 20 s 循环的受控窗口，按墙钟严格分相：
  - **MOTION 0–10 s**：红球以 4 s 周期左右扫动（带拖尾），左下角循环播放 10 s MP4（640×360）。
  - **STATIC 10–16 s**：红球冻结在中央，背景深绿。
  - **FLASH 16–20 s**：背景红/蓝每 120 ms 切换，中央字母 A/B 同步切换，球隐藏。
- 窗口内有大号毫秒时钟、进度条；每 25 帧写 `artifacts/dynamic/window_geom.json`。
- 每轮 9 个问题，覆盖运动、静止、两帧方向、背景色、两帧变化、两帧静止判断：
  q1 时钟、q2 球位置、q3 两帧方向、q4 两帧静止、q5 时钟、q6 背景色、q7 两帧背景变化、q8 两帧方向、q9 球位置。
- 两种模式各跑一轮：`full`（整帧 1200×800 JPEG）与 `delta`（仅脏区裁剪，脏区并集 >62% 时回退整帧）。
- 端到端延迟 = 画面时间戳（帧捕获时刻）→ 模型回答返回时刻；另记 API 延迟、帧龄、脏区占比。
- 判定：时钟 ±3 s；球左/中/右；方向由两帧真实位置差判定；静止看两帧；颜色由帧像素实际颜色判定；变化由两帧背景色是否相同判定。

## 2. 结果总览

| 指标 | full（整帧） | delta（增量） |
|---|---|---|
| 答对 | **{F['correct']}/9** | **{D['correct']}/9** |
| 端到端中位数 | {F['e2e_ms']['median']:.0f} ms | {D['e2e_ms']['median']:.0f} ms |
| 端到端均值 / 最大 | {F['e2e_ms']['mean']:.0f} / {F['e2e_ms']['max']:.0f} ms | {D['e2e_ms']['mean']:.0f} / {D['e2e_ms']['max']:.0f} ms |
| API 中位数 | {F['api_ms']['median']:.0f} ms | {D['api_ms']['median']:.0f} ms |
| prompt tokens 合计 | {F['total_prompt_tokens']} | {D['total_prompt_tokens']} |
| 单图 prompt tokens（均） | {single_full:.0f} | {single_delta:.0f} |
| completion tokens 合计 | {F['total_completion_tokens']} | {D['total_completion_tokens']} |

**Token 结论：** 整轮 prompt 只省 **{save:.1f}%**，单图省 **{save_single:.1f}%**（{single_full:.0f} → {single_delta:.0f}）。远低于静态桌面场景的 5.23×，原因是动态画面里“时钟 + 视频缩略图 + 球”同时变化，脏区并集常覆盖大半个窗口；静止时脏区为 0，又回退整帧。

### full 模式逐题

{table("full")}

### delta 模式逐题

{table("delta")}

## 3. 五项结论

### 3.1 能否实时看懂：**部分能**

- **跟得上**：秒级时钟（q1/q5）、球的位置（q2/q9）、两帧方向（q3/q8）、单帧背景色（q6）、慢变化的窗口内容。
- **部分跟上**：两帧静止判断（q4，full 通过、delta 失败；历史上也出现过单帧答“运动”）；两帧背景变化（q7，模型把同色深浅差异判成“不同”）。
- **跟不上**：120 ms 级的“此刻颜色”、快速闪烁中的字母/边框细节、被增量裁剪切掉部分文字后的内容（q5 delta 只剩“53”）。
- 视频缩略图里的细节（球的运动、矩形位置）只能看个大概，模型对缩略图内容的描述明显弱于主画面。

### 3.2 端到端延迟

- full：中位 **{F['e2e_ms']['median']/1000:.2f} s**，均值 {F['e2e_ms']['mean']/1000:.2f} s，范围 {F['e2e_ms']['min']/1000:.2f}–{F['e2e_ms']['max']/1000:.2f} s。
- delta：中位 **{D['e2e_ms']['median']/1000:.2f} s**，均值 {D['e2e_ms']['mean']/1000:.2f} s，最大 {D['e2e_ms']['max']/1000:.2f} s（q5 因裁剪后模型长推理）。
- API 中位 **{F['api_ms']['median']/1000:.2f} s**；帧龄约 30–260 ms，采集不是瓶颈，**模型推理是瓶颈**。
- 同一问题在多轮运行中波动很大（1.2 s–19.5 s），`deepseek-flash` 的 reasoning token 长度不可控；2048 token 预算曾被 reasoning 吃满导致空回答，需 4096 兜底。

### 3.3 增量省了多少 token

- 合计 **{F['total_prompt_tokens']} → {D['total_prompt_tokens']}**（省 {save:.1f}%）；单图 **{single_full:.0f} → {single_delta:.0f}**（省 {save_single:.1f}%）。
- 增量在“局部小变化”才划算；本测试的视频面板持续播放 + 时钟每秒变化，使脏区并集覆盖 70–100% 画面，收益被吃掉。
- 对比：静态桌面（Phase 2）单帧 962→184 token（5.23×）；动态场景只有约 1.1×。**增量不是万能的，需要按内容选择策略。**

### 3.4 有没有“明显变化但模型没察觉/滞后”

有，且分三类：

1. **判定错误**：q4 在 delta 下把静止画面判为“运动”；此前多轮单帧版本也多次答“运动”。
2. **被裁剪误导**：q5 delta 的脏区从 x=292 开始，把时钟左边“21:3”切掉，模型只看到“53”，回答“53”；端到端 19.5 s。这是增量策略的**语义完整性**问题，不是模型能力问题。
3. **快速变化滞后**：FLASH 每 120 ms 换色，而端到端约 1.5–2 s；模型回答的是“它看到的那一帧”，不是“提问时刻”。q6 曾出现期望红、答蓝（帧龄 84 ms，恰逢换色边界）；q7 把同色深浅判成“不同”。
4. **单帧不可判**：单帧既可能是运动中的球，也可能是静止的球；不给时间上下文时，模型倾向猜“运动”。

### 3.5 改进建议

1. **增量裁剪要“语义安全”**：先按固定语义 ROI（时钟、标题、目标控件）扩展脏区，禁止把文本/数字从中间切断；文本类区域永远回退整行/整块。
2. **问题与帧对齐**：缓存最近 N 帧（带时间戳），提问时选与问题时刻最近的帧；对“此刻/是否在动”类问题，强制给 2–3 帧或短 GIF，而不是单帧。
3. **快速变化用状态接口**：>2 Hz 的闪烁不适合视觉问答；能读 UI 状态（DOM/无障碍/窗口标题）就走状态接口，视觉只做兜底。
4. **降低推理延迟**：对实时问答关闭/限制 reasoning（或换低延迟视觉模型），把 max_tokens 与超时做成配置；空回答自动重试一次（已在 `dynamic_test.py` 实现）。
5. **采样与消费解耦**：采集 4–10 fps、问答按需取最新帧；对视频源单独限帧/降分辨率，避免一个动画缩略图把整屏脏区撑大。
6. **可观测性**：记录 frame_id、帧龄、脏区、prompt/completion tokens、API/e2e 延迟，便于定位“答错”是裁剪、延迟还是模型问题。

## 4. 复现

```powershell
# 1) 启动动态源（保持前台置顶）
py -3.12 tools/dynamic_window.py
# 2) 整帧 / 增量各跑一轮
py -3.12 -u tools/dynamic_test.py --mode full  --json artifacts/dynamic/full.json
py -3.12 -u tools/dynamic_test.py --mode delta --json artifacts/dynamic/delta.json
```

证据：`artifacts/dynamic/sample_composite.jpg`（两帧拼接）、`sample_delta_clock.png`（脏区裁剪）、`_flash_1.png`（FLASH 单帧）。

## 5. 局限

- 单轮 9 题、每个模式 1 次，样本小；模型推理长度波动大，结论以中位数与多轮观察为主。
- 动态源为受控 Tk 窗口（本地 MP4），不是真实视频网站；真实网页的视频解码/合成路径可能更慢。
- 判定器是规则式的（关键词/颜色像素），q7 的“同色深浅”属于边界情况。
- `deepseek-flash` 是推理型视觉模型，延迟不代表所有视觉模型；换模型需重跑。
'''
Path("docs/动态实时性测试报告.md").write_text(md, encoding="utf-8")
print("written", len(md))
