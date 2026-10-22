# 视神经转化引擎 OpticNerve

> 任何输入 → 对当前大脑最优的形态。
> 视网膜负责"看见"，大脑负责"理解"；OpticNerve 是两者之间的实时转化层。

版本 2.0 · 源码 `vision_retina/optic_nerve.py` · 测试 `tests/test_optic_nerve.py`

---

## 1. 为什么需要它

虚拟视网膜早期把帧编码成 JPEG 就直接喂给大脑，问题出在两头：

- **整帧喂图贵**。1920×1080 一帧约 962 token（估算口径），而屏幕上真正有信息的地方常常只占 1/10。
- **画面不能转文本**。"图 → markdown" 对文档是对的，对画面是灾难：OCR 一错，空间关系、颜色、图标、布局全丢，VLM 反而更看不清。

OpticNerve 的结论很直接：

| 输入 | 输出 | 理由 |
| --- | --- | --- |
| 画面（截图、照片、视频帧） | **增强像素 / 稀疏编码帧**（JPEG 马赛克 + 坐标映射） | VLM 吃像素；省 token 靠"少传冗余像素"，不靠"转文字" |
| 文档（PDF/Word/PPT/Excel/邮件/EPUB/网页） | markdown | 文本本来就该是文本，复用 MarkItDown 的成熟转换器 |
| 表格 / CSV / JSON | 结构化载荷（不丢行列）+ 可选 markdown 表 | 行列关系是语义的一部分 |
| 代码 | 结构化（语言、行号、符号大纲）+ 原文 | 高亮/大纲比裸文本更好读 |
| 视频 | 关键帧 + 转写挂载点 | 一帧一帧地看，别把视频当文档 |

同时**不改大脑接口**：输出仍然是 OpenAI 兼容的 `image_url` / `text` 内容块，任何现有 VLM 都能直接吃。

---

## 2. 架构

```
                     ┌──────────────────────────── OpticNerve ────────────────────────────┐
  RetinaFrame ──────▶│ sniff（magika / 魔数 / 扩展名 / ZIP 容器 / 文本判定）                │
  Observation ──────▶│        │                                                            │
  文件 / URL / bytes ▶│        ▼  按 priority 排序的转换器注册表（同类只留最新）             │
  data URI / 流 ────▶│  vision 0 · video 1 · structured 10 · code 12 · markdown 40 · ...   │
                     │        │                     │                     │                │
                     │        ▼                     ▼                     ▼                │
                     │  VisionConverter     StructuredConverter     CodeConverter          │
                     │  检测→超分→增强       行列/记录/schema         语言/大纲              │
                     │  →拼图→预算→映射       └──────────┬──────────┘                      │
                     │        │                        ▼                                 │
                     │        │              docx/html/xlsx/pdf/pptx/epub/zip/plain        │
                     │        ▼                        ▼                                 │
                     │  Result(kind=image)      Result(kind=markdown/structured/frame)     │
                     └────────┬─────────────────────────────────────────┬──────────────────┘
                              ▼                                         ▼
                  image_url 内容块（增强像素）                    text 内容块（markdown/表）
                              └──────────────► 大脑（OpenAI 兼容）◄─────┘
```

保留的 MarkItDown 骨架（未改语义）：

- `DocumentConverter` 抽象基类：`accepts(file_stream, stream_info, **kwargs)` / `convert(...)`
- `StreamInfo`（frozen dataclass + `copy_and_update`）
- `register_converter(converter, priority=...)` 插件式注册，**同类转换器只保留最新一次**（插件可热替换）
- 优先级排序 + `magika` 类型识别（缺失时回退到魔数/扩展名推断，行为一致）

---

## 3. 快速开始

### 3.1 画面：一帧进，一帧便宜的增强帧出

```python
from vision_retina.optic_nerve import OpticNerve
from vision_retina.frames import Rect

engine = OpticNerve()
result = engine.convert(frame)          # RetinaFrame / ndarray / PNG 路径都行

result.kind                             # ResultKind.IMAGE
result.image.tokens                     # 输出 token（估算）
result.tokens.original, result.tokens.output   # 基线 token → 输出 token
result.tokens.ratio                     # 输出/基线；例如 0.45 = 省了 55%（视画面而定）

result.image.save("artifacts/view.jpg") # 人眼可看的增强画布
brain.ask_images(result.image_bytes(), "屏幕上第三个按钮写的是什么？")
```

### 3.2 坐标还原（**最容易踩的坑**）

稀疏编码后，模型看到的是"马赛克画布"，坐标不是原帧坐标。要让模型指位置、或者把它的坐标用回真实屏幕，必须过映射器：

```python
mapper = result.mapper                     # result.meta["mapper"] 同义
mapper.to_native_point(120, 40)            # 马赛克坐标 → 原帧坐标 (x, y)
mapper.to_native(Rect(0, 0, 200, 60))      # 矩形 → 原帧矩形（跨块取并集）
mapper.to_mosaic(Rect(900, 300, 120, 40))  # 原帧矩形 → 马赛克矩形
```

视觉流水线里所有裁剪/放大/拼图都会写入 `placements`，映射不是线性缩放。

### 3.3 文档：什么都不用改

```python
result = engine.convert("report.docx")     # 也可传 PDF/PPTX/XLSX/HTML/EPUB/URL
print(result.markdown)                     # MarkItDown 语义的 markdown
print(result.title, result.kind)           # ResultKind.MARKDOWN
```

### 3.4 实时流

```python
from vision_retina.optic_nerve import OpticNerveStream

stream = OpticNerveStream(engine, dedupe=True)
for frame in camera_frames():              # 或 bus 的观察流
    result = stream.push(frame, change_ratio=ratio, dirty_rects=dirty)
    if result is None:
        continue                           # 与上一帧同构：零 token
    send_to_brain(result)

stream.stats()   # {'pushed':…, 'emitted':…, 'deduped':…, 'tokens_in':…, 'token_ratio':…}
```

已有帧序列时也可以直接要一个生成器（内部同样是 `OpticNerveStream`）：

```python
for result in engine.stream(frames, dedupe=True):
    if result is not None:
        send_to_brain(result)
```

---

## 4. 结果类型

`DocumentConverterResult` 在保留 markitdown 字段（`markdown` / `title` / `text_content` / `__str__`）的基础上扩展为四类：

| `kind` | 载荷字段 | 典型来源 | 给大脑的形式 |
| --- | --- | --- | --- |
| `MARKDOWN` | `markdown` | 文档、网页、纯文本 | `text` |
| `IMAGE` | `image: EncodedFrame`（+`images`） | 截图、照片、图片文件 | `image_url` |
| `FRAME` | `frames: list[EncodedFrame]` | 视频关键帧 | 多张 `image_url` |
| `STRUCTURED` | `structured: StructuredPayload` | 表格/CSV/JSON/代码 | `text`（紧凑表或记录） |

关键不变量：

- **视觉结果的 `markdown` 恒为空字符串**——不是说"忘了填"，而是从类型上禁止"把画面转成文字"这条退化路径。
- `result.is_visual` 区分视觉/文本；`result.image_bytes()` 直接给出可喂大脑的字节。
- `result.to_openai_content(prompt)` 生成 `[{"type":"text"}, {"type":"image_url"}]` 列表，可直接拼进 `messages`。
- `result.tokens: TokenReport`（`original` / `output` / `ratio` / `saved`），预算与省了多少一目了然。

`EncodedFrame` 自带 `to_data_uri()` / `to_openai_part()` / `save()` / `to_dict()`；`StructuredPayload` 自带 `to_markdown()` / `to_compact()` / `to_records()` / `shape`。

---

## 5. 转换器家族

内置 15 个，按 priority 从小到大尝试（越小越先）：

| priority | 转换器 | 接受 | 输出 |
| --- | --- | --- | --- |
| 0 | `VisionConverter` | `RetinaFrame`/`Observation`/ndarray/图片字节/图片路径 | `IMAGE`（稀疏编码/整帧/双路） |
| 1 | `VideoConverter` | 视频字节/路径（需 OpenCV） | `FRAME`（关键帧） |
| 10 | `StructuredConverter` | CSV/TSV/JSON/JSONL/值表 | `STRUCTURED` |
| 12 | `CodeConverter` | 源码扩展名/语言标记文本 | `STRUCTURED(format=code)` |
| 40 | `MarkdownConverter` | `.md`/`.markdown` | `MARKDOWN` |
| 50 | `DocxConverter` | `.docx`（zipfile + XML，零依赖） | `MARKDOWN` |
| 50 | `HtmlConverter` | `.html`/`text/html` | `MARKDOWN` |
| 51 | `EpubConverter` | `.epub`（container→opf→spine） | `MARKDOWN` |
| 52 | `PptxConverter` | `.pptx`（需 python-pptx） | `MARKDOWN` |
| 53 | `PdfConverter` | `.pdf`（需 pypdf） | `MARKDOWN` |
| 54 | `XlsxConverter` | `.xlsx`（需 openpyxl） | `STRUCTURED(format=sheets)` |
| 60 | `ImageDocumentConverter` | 图片 + `as_document=True` | `MARKDOWN`（图片链接） |
| 70 | `ZipConverter` | `.zip` | `MARKDOWN`（清单） |
| 300 | `PlainTextConverter` | 可打印文本 | `MARKDOWN` |
| 1000 | `FallbackConverter` | 最后兜底 | `MARKDOWN` 或 `UnsupportedContentError` |

两条容易忽略的规则：

1. **专用优先于兜底**：`_convert_stream` 先跑所有 `generic=False` 的转换器（对全部候选 StreamInfo），再跑 `generic=True` 的兜底。否则 docx 的二进制会被当成"可打印文本"输出垃圾 markdown。
2. **缺依赖要明说**：`PptxConverter` / `PdfConverter` 缺库时抛 `MissingDependencyException`（带 `pip install …` 提示），不静默降级。

---

## 6. 视觉流水线（OpticNerve 的核心）

`VisionConverter.convert_pixels` 的每一步都是可关的：

```
原帧
 ├─ 1. 基线 token：按 baseline_width 缩放整帧估算（"不改造时大脑要付多少"）
 ├─ 2. 分块 detect_blocks：前景掩码 → XY-Cut 递归切分 → 邻近合并 → 显著度
 │      显著度 = 0.6·边缘密度 + 0.4·对比度（脏矩形内 +0.35 加权）
 ├─ 3. 收墨迹 _ink_rect：每块收紧到真正有像素的范围，丢掉分隔线/边条
 │      —— 背景留白不该付 token（1280×9 的横条会把整张画布撑宽）
 ├─ 4. 拼图 _pack：shelf 布局（按高度降序），块间距 gap，底色取四边中位数
 ├─ 5. 超分 _text_upscale：块内估计字高 < min_text_px 时放大（≤2×）
 ├─ 6. 增强 enhance_pixels：分位数拉伸 → 局部均值/方差增益（clip 0.55~2.1）→ unsharp
 └─ 7. 预算 budget_area：超预算时按下面的顺序逐级让步，先省冗余再省内容
```

预算超了怎么办（**顺序即优先级**）：

1. **收超分倍数**：先按 `sqrt(预算/实际)` 把"小字超分"降下来，内容一个不少；
2. **整体降采样**：仍超就缩画布，但守住"最小可辨认字高 ≈12px"（`12/max_text_px` 作为下限），字不会被缩到看不清；
3. **丢块**：只有当画布比"直接发一整帧"还贵（`hard_area` = 整帧基线）时，才按显著度从低到高丢冗余块，并保底 `keep_floor`。

> 一句话：**先丢分辨率，再丢冗余，最后才丢内容**。默认参数下"删除用户正在看的文字"永远不会为了省 token 而发生。

三个模式（`options.mode`）：

- `sparse`（默认，静态/普通场景）：上面的流水线，省 token 的主力。
- `full`：整帧缩放 + 增强。用于**动态场景**（`scene="dynamic"` 强制）和"整个画面都是内容"的整页扫描件——这类图分块收益小，硬切反而割裂版面。
- `dual`：视杆低清全帧 + 视锥焦点高清（脏矩形并集或最显著 3 块）。给需要"全局上下文 + 局部细节"的场景，**token 会变多**（两帧），是 opt-in。

### 实测（合成帧，估算口径）

| 场景 | 基线 | 输出 | 比例 | 说明 |
| --- | --- | --- | --- | --- |
| 1280×800 桌面（对话框 + 6 图标 + 任务栏） | 576 | 353 | **0.61** | 8 块全保留、**scale=1.0**；省掉的是壁纸/背景 |
| 800×600 文档式（卡片 + 段落 + 小字段） | 394 | 286 | **0.73** | 合成字高仅 5–6px，触发了超分，比例偏高 |
| 800×600 密排整页文字 | 394 | 394 | 1.00 | 内容≈整帧 → 自动 `full`，不硬切 |
| 同上，`scene="dynamic"` | 394 | 394 | 1.00 | **不降级**：动态场景强制整帧 |
| 同上，`mode="dual"` | 394 | 556 | >1 | 双路本就更贵，按需开 |
| 460×300 纯对话框（整屏都是内容） | 211 | 211 | 1.00 | 只有内容没有背景时，老实不省 |
| 指令变焦 440×120 区域 | 394 | 214 | 0.54 | 局部高清，绝对值远低于整帧 |
| 小字低对比模糊帧（420×120） | — | — | — | 可读性 0.029 → **0.224（×7.7）** |

> 结论口径：**空得越多省得越多，内容越满越不省**。token ≈ 570·√(像素面积)，
> 所以省一半 token 需要面积降到 1/4——这正是"丢背景、不丢内容"能拿到的上限。
> 桌面截图这类"内容只占 30%"的画面落在 0.6 附近；纯对话框落在 1.0（不降级）。
> 小字场景（合成字高 5–6px）为了可读性会主动超分，此时比例偏高是**有意的**。

### token 估算

`estimate_image_tokens(w, h) ≈ max(48, 570·√(MP))`，与既有 `bionic.estimate_image_tokens` 同一标定口径（317²→184、800×600→346、2240×1400→962）。基线按 `baseline_width` 缩放后的面积算，所以"省了多少"是相对"原来的喂法"而言，不是相对某个绝对标准。

---

## 7. 与现有能力兼容

| 现有能力 | 接入方式 | 行为 |
| --- | --- | --- |
| 增量编码（`bus.grid_diff` / `Observation.dirty_rects`） | `convert_observation(obs)` | 脏矩形块显著度 +0.35，优先保留；`kind=unchanged/silence` 直接返回 `None`（零 token） |
| 双路喂图（`bionic.BionicRetina`） | `mode="dual"` 或继续用 bionic | 两条链路独立；OpticNerve 的 dual 是"低清全帧+焦点高清"的像素级实现 |
| 指令变焦（`ZOOM_PROTOCOL`） | `engine.zoom(frame, rect)` | 返回单块高清结果（默认 min_width=720，≤2× 放大），**变焦是为了看清，不承诺省 token** |
| 场景自适应（`adaptive.SceneProfile`） | `convert(frame, scene=profile)` | `static` 目标比例 0.48/预算 680；`dynamic` 比例 0.95/预算 1100 且强制 full |
| 隐私区域（`safety.region_whitelist`） | 不变 | 裁剪在采集层完成，OpticNerve 只处理已经合法的像素 |

`scene` 接受 `"static"/"normal"/"dynamic"` 字符串、`SceneProfile` 实例或含 `name` 的 dict。

---

## 8. 接入视网膜（两种粒度）

**逐帧**（最小改动，适合现有 `service.ask`）：

```python
from vision_retina.optic_nerve import default_engine

engine = default_engine()                     # 读 config/retina.yaml 的 optic_nerve 段
obs = service.bus.observe()
result = engine.convert_observation(obs)      # unchanged → None；frame → IMAGE
if result is not None:
    brain.ask_images(result.image_bytes(), prompt)
```

**流式**（推荐，自动去重 + 统计）：

```python
from vision_retina.optic_nerve import OpticNerveStream

stream = OpticNerveStream(engine, scene="normal")
while True:
    obs = service.bus.observe_latest()
    result = stream.push(
        obs.frame, change_ratio=obs.changed_ratio,
        dirty_rects=obs.dirty_rects, force=obs.is_key_frame,
    )
```

`OpticNerveStream` 用 32×18 灰度指纹去重：画面没变、且变化率低于 `change_threshold`（默认 0.0015）时返回 `None`，一个 token 都不花。

**服务层直连**（`retina_perceive` 工具就是这个）：

```python
service.perceive()                        # {"ok": True, "kind": "image", "tokens": {...}, "image": {...}}
service.perceive(include_image=True)      # 额外带 image.base64，可直接转给 OpenAI 兼容接口
```

`perceive()` 内部就是 `observe()` → `OpticNerve.convert_observation()`，静默帧返回
`{"ok": True, "skipped": True, "reason": "unchanged"}`。MCP 侧暴露为 `retina_perceive`，
WebSocket 侧为 `{"type": "perceive"}`。

---

## 9. 配置（`config/retina.yaml` → `optic_nerve`）

全部字段都有默认值（`vision_retina/config.py:DEFAULTS`），不配也能跑。

```yaml
optic_nerve:
  mode: auto             # auto | sparse | full | dual
  max_tokens: 900        # 单帧视觉预算
  target_ratio: 0.62     # 稀疏编码目标面积占比
  baseline_width: 1280   # token 基线：不改造时大脑收到的宽度
  enhance: true
  sharpen: 0.62
  min_text_px: 16        # 小于该字高触发超分
  max_scale_up: 2.0
  sparse: true
  gap: 6
  max_blocks: 48
  min_block_area: 256
  keep_floor: 0.2
  dynamic_keep: 0.85
  dirty_bonus: 0.35
  jpeg_quality: 82
  png: false
  min_output_width: 360
  max_output_width: 1280
  low_width: 640         # dual 低清全帧
  low_quality: 72
  focus_max_width: 960   # dual 焦点高清
  focus_quality: 88
```

代码里也可以直接覆盖：`OpticNerve(options=VisionOptions(max_tokens=600, mode="sparse"))`。

---

## 10. 扩展：新增一种内容类型

引擎本体不动，只加一个 converter：

```python
from vision_retina.optic_nerve import (
    DocumentConverter, DocumentConverterResult, StreamInfo, PRIORITY_STRUCTURED,
)

class SrtConverter(DocumentConverter):
    name = "srt"

    def accepts(self, file_stream, stream_info: StreamInfo, **kwargs) -> bool:
        return (stream_info.extension or "").lower() in {".srt", ".vtt"}

    def convert(self, file_stream, stream_info: StreamInfo, **kwargs) -> DocumentConverterResult:
        text = file_stream.read().decode("utf-8", "replace")
        return DocumentConverterResult(
            kind="structured",
            structured=StructuredPayload(format="subtitle", rows=...),
            title="字幕",
        )

engine.register_converter(SrtConverter(), priority=PRIORITY_STRUCTURED)
```

约定：

- `accepts` **不能改变流位置**（引擎会 `seek(0)`，但仍要求契约干净）；
- 优先级用模块常量，别写魔法数字；
- 视觉类结果把 `markdown` 留空；
- 缺依赖抛 `MissingDependencyException`，不吞异常。

---

## 11. 分阶段状态

| 家族 | 状态 | 说明 |
| --- | --- | --- |
| 视觉（Vision） | **完整** | 分块/超分/增强/拼图/预算/映射/双路/变焦/流式去重 |
| 文档（PDF/Word/PPT/Excel/HTML/EPUB/ZIP/纯文本） | **可用** | docx 零依赖自实现；pptx/pdf 需装库 |
| 结构化（CSV/JSON/代码） | **可用** | 行列不丢、类型推断、代码大纲 |
| 视频 | **骨架** | 关键帧采样已实现（需 OpenCV）；`transcribe=` 回调留给 Phase 3 接 ASR |

视频的下一步：`VideoConverter(transcribe=asr_fn)` 已有挂载点，接上 whisper 系模型即可"关键帧 + 时间轴转写"。

---

## 12. 测试

```bash
python -m unittest discover -s tests -t tests -p "test_*.py"
# Ran 98 tests … OK   （原有 27 + OpticNerve 71）
```

`tests/test_optic_nerve.py` 覆盖：

- 四类结果的字段契约（视觉 `markdown` 必须为空、`to_openai_content` 结构）
- 稀疏编码：token 下降、坐标双向映射、脏矩形加权、`keep_floor` 不被突破
- 动态场景不降级、`dual` 两帧、预算不超、`zoom` 放大
- 可读性：小字低对比增强后 `score` 显著上升
- 文档：txt/md/html/csv/json/jsonl/py/docx/xlsx/zip/png/data URI/bytes/StreamInfo dict
- 引擎：优先级顺序、`generic` 两轮分发、缺依赖异常、未知二进制报错、`status()` 计数
- 服务接入：`service.perceive()` / 静默帧跳过 / MCP `retina_perceive` 已注册
- 流：去重、`force`、统计口径、`convert_observation` 对 `unchanged` 返回 `None`
