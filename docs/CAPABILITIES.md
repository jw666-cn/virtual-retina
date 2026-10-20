# AI 虚拟仿生视网膜 · 能力摘要

**一只可以接任何大脑的“眼睛”**：采集屏幕/摄像头，按仿生方式压缩，用标准接口把画面交给客户自选的视觉模型与执行工具。
不内置模型、不内置“手”、不绑定任何 Agent 框架。

## 它是什么，不是什么

| 是 | 不是 |
|---|---|
| 感知层外设：采集 + 仿生编码 + 标准接口 | 端到端 Agent，不做规划、不执行点击/输入 |
| 模型无关：任意 OpenAI 兼容视觉模型 | 不绑定 UI-TARS / GPT / Claude / DeepSeek 任一模型 |
| 手无关：MCP（stdio / HTTP）+ WebSocket JSON | 不依赖 dsh / openclaw / opencode 的任何内部实现 |
| 本地优先：默认不落盘、不上传；区域白名单 | 不是录屏软件，不保存历史画面 |

## 仿生架构

| 生物机制 | 工程实现 | 作用 |
|---|---|---|
| 视杆细胞（周边、低清） | 全帧低清（默认 768 宽）+ 脏区/差分 | 看懂版面与运动，token 低 |
| 视锥细胞（中央、高清） | 焦点高清（默认 1024 宽）+ 语义安全裁剪 | 看清小字/表格，读得准 |
| 中央凹（可编程注视） | 模型输出 `zoom` 指令 → 按坐标放大返回 | “想看哪→放大哪→读准”闭环 |
| 视网膜并行预处理 | 脏区、语义 ROI、静默、编码缓存 | 不把整张图原样上传 |
| 暗适应 | EMA 变化率 + 滞回三档（static/normal/dynamic） | 静处省 token，动处保采样 |

```
帧 → 差分/脏区 → 关键帧/静默 → 低清全帧 ─┐
                                      ├→ 大脑：回答 或 zoom(x,y,w,h)
        zoom → 坐标换算 → 语义扩展 → 焦点高清 ─┘
```

## 三卖点，带硬数据

### 看得清

- CAD 图纸 9 题（小字表格/标题栏/轴网）：单帧全图 **4/9** → 双路 + 指令变焦 **7/9（78%）**。
- 每命中 token 从 **1,748 → 1,398（−20%）**；单请求图片 token 全部 **≤1,800**。

### 看得省

- 单帧图片 token：全帧 **1,029 → 中央凹 184（5.23×）**。
- 动态序列相对全帧：图片 token **−43.2%**。
- 低变化场景：发送帧 **−27.5%**、token **−32.0%**；动态场景帧数 **不降级**。
- 按需变焦：9 题中 1 题直接低清作答，仅 **434 token**（比全帧省 44%）。

### 不绑定

- 大脑：`/chat/completions` 兼容接口，换模型不改代码。
- 手：MCP tools + WebSocket `retina.ws/1`，客户自选执行器。
- 协议：新字段只增不改；MCP 支持 stdio 与 Streamable HTTP。

## 接口一览

MCP tools：`retina_observe` / `retina_ask` / `retina_ask_fovea` / `retina_set_region` / `retina_list_sources` / `retina_status` / `retina_health`。

```jsonc
// 双路 + 按需变焦
{"type":"ask_fovea","params":{"prompt":"技术特性表的输入功率是多少？","budget_tokens":1800}}
// 返回：answer / zoom / calls / prompt_tokens / budget_ok / crop_safe / low / focus
```

WS：`observe` / `ask` / `ask_fovea` / `ask_actions` / `set_region` / `list_sources` / `status` / `health` / `ping`。

## 快速开始

```powershell
py -3.12 -m pip install -r requirements.txt
# config/secrets.local.env：VISION_API_BASE / VISION_MODEL / VISION_API_KEY
py -3.12 -m vision_retina doctor
py -3.12 tools/probe_brain.py --health
py -3.12 -m vision_retina serve --transport http --port 8766
```

## 边界（先说清楚）

- **不是 Agent**：只输出画面与答案/动作 JSON，不执行、不审批。
- **Windows only（生产）**：DXGI + GDI 已实测；macOS/Linux 采集接口占位，不返回假帧。
- **CAD 识图 78%**：7/9，仍会读错细线数字、也会把变焦框指错区域。
- **120 ms 级闪变会漏**：受采集/推理节拍限制，快速红蓝闪变可能看不到。
- **变焦框依赖模型定位**：模型指错区域时视网膜不纠正（V2 计划加候选区域）。
- **5.23× 是单帧图片 token 口径**：总 prompt 口径约 4.1×，报告同时披露。
- **隐私**：本地采集；默认不上传、不落盘；接云端模型时画面仍会离开本机。

## 验证与文档

| 内容 | 位置 |
|---|---|
| 双路/变焦/预算实测 | [`reports/fovea_budget.md`](../reports/fovea_budget.md) |
| 场景自适应实测 | [`reports/adaptive.md`](../reports/adaptive.md) |
| 仿生设计说明 | [`docs/biomimicry.md`](biomimicry.md) |
| 接口协议 | [`docs/接口协议.md`](接口协议.md) |
| 与 computer-use 方案对比 | [`reports/cu_compare.md`](../reports/cu_compare.md) |
| 单测 | `py -3.12 -m unittest discover -s tests`（27 项） |

## 适用 / 不适用

**适用**：给现有 Agent 加“低带宽视觉”；长时挂机观察；远程桌面/客服/巡检的看屏问答；需要换模型、换执行器的集成。

**不适用**：需要模型自己完成整段 GUI 操作的端到端场景（请直接用 computer-use Agent）；微秒级高速视觉；无模型 Key 的纯离线识图。
