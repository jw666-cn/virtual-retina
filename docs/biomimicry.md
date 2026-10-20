# 仿生设计说明（V1）

> 目标：把“眼睛”做成外设——采集、仿生编码、标准接口；理解与行动交给客户的大脑与手。
> 本文件对应 V1.2–V1.6，实现见 `vision_retina/bionic.py`、`adaptive.py`、`bus.py`、`roi.py`。

## 1. 生物机制 → 数字实现

| 生物机制 | 数字实现 | 代码 / 接口 | 状态 |
|---|---|---|---|
| 视杆细胞：周边、低分辨、看全局与运动 | 全帧低清（默认 768 宽）+ 脏区/差分 | `BionicRetina.low_view()`、`bus.grid_diff` | ✅ V1 |
| 视锥细胞：中央、高分辨、看细节 | 焦点高清（默认 1024 宽）+ 语义安全裁剪 | `BionicRetina.focus_view()`、`roi.expand_rect_to_content` | ✅ V1 |
| 中央凹：可编程注视点 | 模型输出 `zoom` 指令 → 指定区域放大返回 | `parse_zoom()`、`BionicRetina.ask()` | ✅ V1 |
| 视网膜并行预处理：不把整张图原样上传 | 脏区 + 语义 ROI + 静默 + 编码缓存 | `bus.py`、`service.py` | ✅ V1 |
| 暗适应：暗/静时提高灵敏度、降低采样 | EMA 变化率 + 滞回三档（static/normal/dynamic） | `adaptive.AdaptiveController` | ✅ V1 |
| 眼跳：微动采样、大脑拼接 | 多帧采样 / 注视点序列 | — | V2 |
| 盲点：无感光、大脑补全 | 区域白名单 / 关键区冗余 | `safety.region_whitelist`（白名单已做） | V2 扩展 |

## 2. 双路喂图与指令式变焦

```
帧 ──► 低清全帧（视杆） ──┐
                          ├─► 大脑第 1 次：回答 or 请求 zoom
                          │
        ┌─────────────────┘
        ▼
  zoom(x,y,w,h) ──► 坐标换算（低清→原图）──► 语义安全扩展 ──► 焦点高清（视锥）
                                                              │
                          大脑第 2 次：低清 + 焦点 ────────────┘
```

第一次请求要求模型只输出二选一 JSON：

```json
{"action":"answer"}
{"action":"zoom","region":{"x":120,"y":80,"w":300,"h":140},"reason":"表格数字太小"}
```

- `region` 用**低清图坐标**；视网膜按 `原图宽/低清宽` 等比换算。
- 裁剪后调用 `expand_rect_to_content()`：检测与裁剪框相交的文字行/连通块，扩到整行，避免“切字”。
- 若语义扩展后面积超过 `max_area_ratio`（默认 0.45），退回“请求框 + 边距”并标记 `crop_safe=false`，不返回整帧。

## 3. token 预算联动（三条）

| 原则 | 实现 |
|---|---|
| ① 按需触发 | 只有第 1 次回答是 `zoom` 才发第 2 次；`answer` 直接结束（实测 9 题中 1 题未变焦，仅 434 token） |
| ② 裁剪到位 | 只裁模型指定矩形并做语义扩展；实测 8 次变焦平均覆盖约 5% 画面 |
| ③ 双路受控 | `budget_tokens` 约束“低清 + 焦点”的图片 token 估计；超预算先缩小焦点宽度，仍超则拒绝变焦、返回第 1 次回答 |

token 估计：`570 × sqrt(width×height/1e6)`，按 DeepSeek 图像计费校准；不同厂商需在 `bus.fovea.budget_tokens` 侧重新标定。

预算参数（`config/retina.yaml` → `bus.fovea`）：

```yaml
fovea:
  enabled: true
  low_width: 768          # 视杆：全帧低清宽度
  low_quality: 72
  focus_max_width: 1024   # 视锥：焦点高清上限
  focus_quality: 88
  max_scale: 2.0          # 小区域最大上采样倍数
  budget_tokens: 1200     # 单请求双路图片 token 上限
  text_reserve: 120       # 给提示词/系统消息预留
  max_area_ratio: 0.45    # 语义扩展后的面积上限
```

## 4. 场景自适应（暗适应）

- 每帧 `changed_ratio` → EMA（α=0.3）。
- `ema < 0.006` 持续 3 s → `static`：fps 1.0、JPEG 60、宽 1000、静默 6 s。
- `ema ≥ 0.02` 持续 0.4 s → `dynamic`：fps 6.0、JPEG 86、宽 1600、静默 2 s。
- 中间为 `normal`：fps 2.0、JPEG 80、宽 1280。
- 滞回：进 static 后需 `ema > 0.010` 才退出；进 dynamic 后需 `< 0.008` 才退出。
- 发送节流：`static` 档下即使检测到小变化，也最多 1 fps 发帧，其余返回 `unchanged`（无图）。
- `service.observe()` 按当前档位编码，帧 `meta.scene` 与 `status().adaptive` 可观测。

## 5. 增量层统一

所有入口共用同一条预处理链：

```
采集帧 → grid_diff（脏区 + 变化率）→ 关键帧/静默 → 语义 ROI（可选）
      → 编码（按档位/预算）→ 单帧缓存（同像素复用）→ Observation / RetinaFrame
```

- `bus.evaluate()` 输出 `Observation(kind=frame|unchanged|silence, dirty_rects, changed_ratio, frame)`。
- `service.observe()` 对静默/无变化返回最近一帧的缓存编码（`image.cached=true`）。
- `retina_ask` 每次只带当前一帧，不累积历史图片；`retina_ask_fovea` 在一次请求内最多两张（低清 + 焦点）。

## 6. 对外接口

- `retina_ask(prompt)`：单帧问答（全帧或按档位低清）。
- `retina_ask_fovea(prompt, budget_tokens=1200)`：双路 + 按需变焦，返回 `answer / zoom / calls / prompt_tokens / budget_ok / crop_safe`。
- WebSocket：`{"type":"ask_fovea","params":{"prompt":"...","budget_tokens":1200}}`。
- 协议详见 `docs/接口协议.md`。

## 7. 验证结论

| 能力 | 结果 | 报告 |
|---|---|---|
| 双路 + 变焦 + 预算 | 9 题命中 4→7（+75%），每命中 token −20%，单请求全部 ≤1,800 | `reports/fovea_budget.md` |
| 场景自适应 | 低变化场景帧数 −27.5%、token −32%；动态场景不降级 | `reports/adaptive.md` |
| 增量层 | dirty-rects / 语义裁剪 / 静默 / 缓存统一进总线，27 项单测 | `tests/` |

## 8. 边界

- 视网膜不是 Agent：不规划、不执行动作，只给画面和答案。
- 变焦区域由模型给出；模型指错区域时视网膜不纠正（V2 可加候选区域列表）。
- 焦点清晰度受原图分辨率约束；3–5 px 字高的扫描件无法靠放大补救。
- 自适应阈值按 1200×800、2 Hz 桌面场景标定；4K/高帧率需重新标定。
- 客户模型必须支持多图输入（双路）；单图模型会退化为“先低清、再单焦点”。
