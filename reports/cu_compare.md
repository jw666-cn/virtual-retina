# 视网膜 vs UI-TARS 等 computer-use 方案：方案级横向对比

- 版本：2026-10-20
- 性质：**资料级横向对比**，不安装、不运行 UI-TARS 本体，不做端到端跑分。
- 目的：定位「视网膜」在同类方案中的差异化与明确边界，作为 README 背书。
- 取证方式：只采用可访问的官方/论文原文；本机对 GitHub raw、OpenAI、Anthropic 官方页存在网络限制（超时/403/地区限制），未纳入量化断言。

## 1. 对比对象与依据

| 对象 | 定位 | 本报告依据 |
|---|---|---|
| 视网膜（本项目） | 只做视觉外设：采集 → 总线 → 增量编码 → 标准接口 | 本项目实测：`docs/验收报告.md`、`reports/ocr_compare.md`、`reports/model_compare.md`、`docs/动态实时性测试报告.md` |
| UI-TARS（ByteDance，2025-01） | 原生 GUI Agent 模型：输入截图，输出键鼠动作 | 论文 arXiv:2501.12326（HTML 4.1 节、5 节、附录） |
| UI-TARS-2（2025-09） | 多轮强化学习版原生 Agent，感知/推理/动作/记忆一体化 | 论文 arXiv:2509.02544 摘要与正文 |
| 通用截屏式 computer-use（OpenAI CUA、Anthropic computer use 等） | 每步截屏 → 模型 → 动作的闭环 | 仅作范式说明；官方页本机不可达，**不引用其细节、不参与打分** |

> UI-TARS 论文原文（4.1 节）明确：每个时间步输入任务指令、历史交互 `(o1,a1,...,oi-1,ai-1)` 与当前观测 `oi`，其中 `oi` 是设备截图；摘要写明 “solely perceives the screenshots as input”。这是下面“整屏/步”判断的直接依据。

## 2. 六维横向对比

| 维度 | 视网膜 | UI-TARS（原生 Agent） | 通用截屏式 computer-use |
|---|---|---|---|
| **识图方式** | DXGI 脏区 + 语义 ROI 安全裁剪；差分/静默/中央凹；可只传变化区 | 每步整屏截图 + 历史；论文未描述脏区/增量协议（4.1 节） | 每步整屏截图 |
| **token / 成本效率** | 实测全帧 1,029 prompt tokens → 中央凹 184（**5.23×**）；动态序列图片 token **−43.2%**；本地 OCR 0 token | 本报告检索范围内，论文未给出每步 token 账本；机制上随步数线性传整屏 | 同左，成本由厂商图片计价决定 |
| **模型 / 工具绑定** | **模型无关**（任意 OpenAI 兼容视觉模型）；手走 MCP/WS；不内置模型、不内置手 | 绑定 UI-TARS 模型（原生 Agent，模型内完成感知/推理/动作） | 绑定对应厂商模型/云运行时 |
| **跨平台** | Windows 已实现（DXGI+GDI）；macOS/Linux 采集接口占位（Phase 3） | 论文覆盖 Web、桌面、移动（Android），统一动作空间；OSWorld/AndroidWorld 评测 | 取决于各厂商运行时 |
| **隐私** | 本地采集、默认不落盘/不上传；区域白名单；可接本地模型 | 画面必须进入模型（云端或自托管） | 截图进入厂商云 |
| **适用场景** | 给现有 agent/“手”提供低带宽视觉；实时观察、长时挂机、成本敏感 | 端到端 GUI 自动化（自己看、自己想、自己做） | 端到端 GUI 自动化 |
| **上手成本** | `pip` + 配置 + MCP/WS 客户端；无需部署模型 | 部署/调用模型 + runtime + 账号 | 厂商账号 + 虚拟机/运行时 |

## 3. 结论

### 3.1 差异化优势（可验证）

1. **感知层解耦**：视网膜不训练模型、不内置“手”，只输出 `RetinaFrame` / `Observation` 与标准动作 JSON；大脑和手由客户自选。UI-TARS 是“模型 + 运行时”的端到端方案，二者不在同一层。
2. **增量协议是结构性差异**：脏区 + 语义 ROI + 静默 + 中央凹，使单帧图片 token 实测降到 1/5.23，动态序列再省 43.2% 图片 token；UI-TARS 类方案每步传整屏，成本随步数线性增长。
3. **接口标准化**：MCP（stdio / Streamable HTTP）+ WebSocket `retina.ws/1` 双协议，任何“手”都能接；不要求客户换模型或换运行时。
4. **隐私与成本可控**：本地采集、默认不落盘、区域白名单；可接本地视觉模型，做到画面不出机器。注意：接云端大脑时画面仍会离开本机，这一点与截屏式方案相同。

### 3.2 明确边界（不回避）

- **不是 Agent**：视网膜没有规划、反思、执行能力；离开大脑与手，它只是“会看的摄像头”。UI-TARS 端到端完成“看懂→决策→执行”。
- **平台**：目前只有 Windows 生产可用；macOS/Linux 是接口占位，未实现。
- **识图精度**：CAD 类图纸实测 44–53/95（46%–56%），显著高于 Tesseract 的 23/95，但技术特性表/轴网尺寸仍是 0 分；不能替代人工读图。
- **实时性**：动态画面 120 ms 级闪变会漏；端到端中位约 1.5–1.7 s，慢变化可跟、快切换跟不上。
- **ROI 依赖先验**：语义安全裁剪需要 `ROI_BY_KIND`（时钟/球/背景等）；未知界面需先配置区域，否则退化为整帧或裁错。
- **成本口径**：5.23× 是“单帧图片 token”口径；按总 prompt 计约 4.1×，报告同时披露，避免误导。

### 3.3 一句话定位

> 视网膜不是“另一个 UI-TARS”，而是给任何 UI-TARS 式（或自研）大脑/手用的**低带宽视觉外设**：把“每步整屏截图”换成“只传变化、只传该看的地方”，并用 MCP/WS 把视觉能力标准化。

## 4. 引用

- UI-TARS: *Pioneering Automated GUI Interaction with Native Agents*, arXiv:2501.12326 — https://arxiv.org/abs/2501.12326
- UI-TARS-2: *Advancing GUI Agent with Multi-Turn Reinforcement Learning*, arXiv:2509.02544 — https://arxiv.org/abs/2509.02544
- UI-TARS 开源仓库（论文给出的地址）：https://github.com/bytedance/UI-TARS
- 本项目实测：
  - `docs/验收报告.md`（Phase 1–3）
  - `reports/ocr_compare.md`（视网膜 vs Tesseract）
  - `reports/model_compare.md`（DeepSeek vs GLM）
  - `docs/动态实时性测试报告.md`、`docs/动态测试修复报告.md`
  - `docs/接口协议.md`

## 5. 方法与限制

- 未安装、未运行 UI-TARS，未做同机同题跑分；“每步整屏”来自论文机制描述，不是实测其 token 数。
- OpenAI CUA / Anthropic computer use 官方页在本机返回 403/地区限制，故只作为“截屏式范式”提及，不引用具体价格、平台或能力数字。
- 视网膜侧数字均来自本项目可复现脚本与落盘 JSON；复现命令见各报告。
