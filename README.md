# 虚拟视网膜（Vision Retina）

纯视觉输入外设：**只做视网膜**——采集、统一总线、增量编码，以标准接口把画面交给“大脑”，把标准动作 JSON 交给“手”。
不内置模型、不内置执行工具、不依赖 dsh/openclaw/opencode 任何内部实现。

- 平台：Windows（DXGI + GDI 兜底，已实测）；macOS/Linux 采集接口预留（Phase 3，见 `docs/Phase3-跨平台与IDD方案.md`）。
- 大脑：任意 OpenAI 兼容视觉模型（`/chat/completions`），配置驱动。
- 手：MCP Server（首选，stdio + Streamable HTTP）+ WebSocket JSON（兜底）。
- 接口契约：[`docs/接口协议.md`](docs/接口协议.md)（RetinaFrame / Observation / 动作 JSON / MCP tools / WS `retina.ws/1`）。
- 对外能力摘要（发布页用）：[`docs/CAPABILITIES.md`](docs/CAPABILITIES.md)。

## 快速开始

```powershell
# 0) 依赖（Python 3.12）：numpy opencv-python pillow openai httpx websockets mcp PyYAML
py -3.12 -m pip install -r requirements.txt

# 1) 配置：在 config/secrets.local.env 写入（不写进 yaml、不入库）
#    VISION_API_BASE=https://api.deepseek.com
#    VISION_MODEL=deepseek-flash
#    VISION_API_KEY=...
#    也可用同名环境变量覆盖；config/retina.yaml 只放非敏感参数。

# 2) 环境 + 采集自检（不消耗 token）
py -3.12 -m vision_retina doctor

# 3) 视觉能力健康检查（两张纯色图，真实探测模型是否能看图）
py -3.12 tools/probe_brain.py --health

# 4) 看一帧 + 提问
py -3.12 -m vision_retina observe --out artifacts/observe.jpg
py -3.12 -m vision_retina ask "屏幕上是什么？"

# 5) 启动对外协议
py -3.12 -m vision_retina serve --transport http --port 8766   # MCP over Streamable HTTP → http://127.0.0.1:8766/mcp
py -3.12 -m vision_retina serve --transport mcp               # MCP over stdio（客户 MCP 客户端拉起）
py -3.12 -m vision_retina serve --transport ws  --port 8765    # WebSocket 兜底，协议 retina.ws/1
```

MCP 客户端配置示例（stdio）：

```json
{
  "mcpServers": {
    "virtual-retina": {
      "command": "py",
      "args": ["-3.12", "-m", "vision_retina", "serve", "--transport", "mcp"],
      "cwd": "<path-to-repo>"
    }
  }
}
```

工具：`retina_observe` / `retina_ask` / `retina_ask_fovea` / `retina_perceive` / `retina_set_region` / `retina_list_sources` / `retina_status` / `retina_health`。
产品只返回标准动作 JSON，**不执行**任何动作；审批与危险操作由客户选择的“手”负责。

## 架构

```
视觉源 → ①采集层 → ②统一输入总线 → ③增量编码 → ③'视神经转化（OpticNerve）→ ④对外标准接口
                                                          ├─► 大脑 API 接入口（OpenAI 兼容）
                                                          └─► 手接入协议（MCP + WebSocket JSON）
```

总线四机制：帧间差分（dirty-rects）、静默、中央凹、特征缓存；`ask` 每次只带当前一帧，历史不累积图片。
V1 新增：双路喂图（低清全帧 + 按需焦点高清）、指令式变焦、token 预算联动、场景自适应（static/normal/dynamic）。
V2 新增：视神经转化引擎 OpticNerve——画面→增强像素/稀疏编码（马赛克帧 + 坐标映射），文档→markdown，表格/代码→结构化；引擎按优先级注册转换器，新内容类型只需加一个 converter。

## 目录

| 路径 | 内容 |
|---|---|
| `vision_retina/capture/dxgi.py` | DXGI Desktop Duplication（ctypes 自研，含 dirty-rects / 旋转 / 区域裁剪） |
| `vision_retina/capture/gdi.py` | GDI 兜底（含光标），DXGI 不可用时自动降级 |
| `vision_retina/capture/camera.py` | OpenCV 摄像头；`camera.path` 可指向视频文件做无硬件联调 |
| `vision_retina/capture/macos.py` / `linux.py` | 三平台接口占位：明确报错，不返回假帧 |
| `vision_retina/bus.py` | 统一输入总线（关键帧 / 差分 / 静默 / 中央凹 / 自适应节流） |
| `vision_retina/bionic.py` | 双路喂图 + 指令式变焦 + token 预算 |
| `vision_retina/optic_nerve.py` | 视神经转化引擎（V2）：画面稀疏编码 / 文档 markdown / 结构化 / 视频关键帧 |
| `vision_retina/adaptive.py` | 场景自适应（暗适应）：EMA + 滞回三档 |
| `vision_retina/roi.py` | 语义安全裁剪：文字行/连通块扩展，避免切字 |
| `vision_retina/service.py` | 采集 + 大脑门面（MCP/WS 共用、编码缓存、区域白名单） |
| `vision_retina/brain.py` | OpenAI 兼容大脑接入口 + 视觉能力健康检查 + 动作 JSON 解析 |
| `vision_retina/mcp_server.py` | MCP Server（stdio / Streamable HTTP） |
| `vision_retina/ws_server.py` | WebSocket 兜底协议 `retina.ws/1` |
| `tools/` | 自检/基准/闭环/验收脚本（不属于产品运行时） |
| `docs/` | 接口协议、Phase 3 方案与验收报告 |
| `reports/` | 对比报告（OCR / 模型 / computer-use） |
| `artifacts/` | 自检样张与运行产物（不进版本库） |

## 验证与验收

```powershell
py -3.12 -m unittest discover -s tests -v          # 98 项单测：帧/差分/动作解析/编码缓存/白名单/静默/双路预算/自适应/OpticNerve
py -3.12 tools/probe_capture.py --backend dxgi --frames 20 --fps 2
py -3.12 tools/probe_bus.py --seconds 6 --fps 2 --fovea --save artifacts/bus_fovea.png
py -3.12 tools/bench_tokens.py                      # Phase 2 token 基准（5.23x）
py -3.12 tools/soak_test.py --seconds 3600 --fps 1  # 1 小时挂机
py -3.12 tools/make_test_video.py; py -3.12 tools/reality_loop.py   # 摄像头现实闭环
py -3.12 tools/closed_loop.py --max-steps 8         # Phase 1 多步闭环（受控 GUI + 本地测试手）
py -3.12 tools/acceptance_report.py --health         # 生成 docs/验收报告.md
py -3.12 tools/fovea_budget_eval.py --budget 1800 --low-width 640   # 双路/变焦/预算前后对比
py -3.12 tools/adaptive_eval.py                     # 场景自适应帧率/token 对比
```

## 与 computer-use 方案的定位差异

视网膜不是“另一个 UI-TARS”，而是给任何 UI-TARS 式（或自研）大脑/手用的**低带宽视觉外设**：

- UI-TARS 这类 native-agent 方案是“模型内完成感知/推理/动作”的端到端闭环，论文中每步输入整屏截图；
- 视网膜只做感知层：脏区 + 语义 ROI + 静默 + 中央凹；单帧图片 token 实测降到 1/5.23，动态序列相对全帧省 43.2% 图片 token；
- 不绑定模型与手，用 MCP / WebSocket 标准接口接入，客户自选大脑与手。

完整六维对比、证据与边界见 [`reports/cu_compare.md`](reports/cu_compare.md)。

## V1 仿生能力（双路 / 变焦 / 预算 / 自适应）

- **双路喂图**（视杆 + 视锥）：低清全帧看版面，焦点高清看小字；两路同一次请求交给模型。
- **指令式变焦**（可编程中央凹）：模型输出 `zoom` 指令，视网膜换算坐标、语义安全裁剪、按预算放大返回。
- **token 预算**：按需触发、只裁必要区域、双路图片 token 设上限；超预算先缩小焦点，仍超则拒绝变焦。
- **场景自适应**（暗适应）：静态/低变化自动降帧与压缩，动态自动提高采样与质量，带滞回避免抖动。
- **实测**：CAD 9 题命中 4/9 → 7/9，每命中 token −20%，单请求全部 ≤1,800；低变化场景帧数 −27.5%、token −32%，动态场景不降级。
- 设计说明：[`docs/biomimicry.md`](docs/biomimicry.md) · 变焦预算：[`reports/fovea_budget.md`](reports/fovea_budget.md) · 自适应：[`reports/adaptive.md`](reports/adaptive.md)
- 完整项目开发报告（V1 + A/B/C）：[`docs/PROJECT_REPORT.md`](docs/PROJECT_REPORT.md)

## V2 视神经转化引擎（OpticNerve）

视网膜负责"看见"，大脑负责"理解"；OpticNerve 是两者之间的实时转化层：**任何输入 → 对当前大脑最优的形态**。

- **画面不转文本**：输出增强像素 / 稀疏编码马赛克帧（`image_url`），markdown 只用于文档与文本类。
- **省 token 不靠转文字**：收墨迹（丢掉背景留白/分隔线）→ 分块 → 小字超分 → 增强 → 拼图 → 预算控制；超预算先降超分、再降分辨率、最后才丢冗余块。
- **坐标可还原**：`result.mapper.to_native_point(x, y)` 把马赛克坐标换回原生像素，跨块矩形取并集。
- **四类结果**：`markdown` / `image`（增强编码帧）/ `structured`（行列、schema、代码大纲）/ `frame`（视频关键帧）。
- **保留 MarkItDown 骨架**：`DocumentConverter` 抽象基类、`register_converter` 插件注册、优先级排序、magika 类型识别、`StreamInfo` 全部保留；新增内容类型 = 加一个 converter，引擎本体不动。
- **实测**：1280×800 桌面（对话框+图标）576→353 token（0.61，8 块全保留、原分辨率）；密排整页 1.0（自动整帧，不硬切）；动态场景不降级；小字低对比帧可读性 0.029→0.224（×7.7）。
- 设计说明与配置：[`docs/optic_nerve.md`](docs/optic_nerve.md) · 配置段：`config/retina.yaml` 的 `optic_nerve`

```python
from vision_retina.optic_nerve import default_engine

engine = default_engine()
result = engine.convert_observation(service.bus.observe())   # unchanged → None（零 token）
if result is not None:
    brain.ask_images(result.image_bytes(), "屏幕上的报错是什么？")
```

## 边界（重要）

- 我们只发指令，不执行、不审批；危险操作由客户所选的“手”负责。
- 默认不存储、不上传画面；仅在内存中实时处理，`safety.persist_frames=true` 才落盘。
- `safety.region_whitelist` 可限定只采集指定区域，越界请求会被拒绝。
- 客户接入的模型必须支持视觉输入，健康检查会做真实图片探测并明确报错。
- 摄像头实体型号未定（任务书 §10 占位），已用视频文件模拟同一管线验证。
