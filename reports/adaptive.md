# 场景自适应编码（仿暗适应）：验证报告

> 对应 V1.5。实现：`vision_retina/adaptive.py`，接入 `vision_retina/bus.py` 与 `vision_retina/service.py`。
> 验证脚本：`tools/adaptive_eval.py`；原始数据：`artifacts/adaptive/results.json`。
> 日期：2026-10-20。

## 1. 仿生机制

暗适应：环境暗、变化少时提高感光灵敏度、降低采样；出现运动/亮暗突变时迅速恢复。

数字实现用一个带滞回的 EMA 控制器：

- 每帧计算 `changed_ratio`（`grid_diff` 的变化块占比），更新 `ema = α·ratio + (1−α)·ema`（α=0.3）。
- **动态**：`ema ≥ dynamic_ratio(0.02)` 持续 `enter_dynamic_s(0.4)` → `dynamic` 档（高采样/高质量）。
- **静态/低变化**：`ema < static_ratio(0.006)` 持续 `enter_static_s(3.0)` → `static` 档（降帧、压缩）。
- 滞回：进入 `dynamic` 后需 `ema < dynamic_exit_ratio(0.008)` 才退出；进入 `static` 后需 `ema > static_exit_ratio(0.010)` 才退出，避免边界抖动。
- 档位不仅影响编码质量，还影响**发送节流**：`static` 档下即使检测到小变化，也最多按 `fps=1.0` 发送，其余返回 `unchanged`（无图）。

档位参数（`config/retina.yaml` → `adaptive`）：

| 档位 | fps | JPEG 质量 | 最大宽度 | 静默阈值 |
|---|---|---|---|---|
| static | 1.0 | 60 | 1000 | 6.0 s |
| normal | 2.0 | 80 | 1280 | 3.0 s |
| dynamic | 6.0 | 86 | 1600 | 2.0 s |

`service.observe()` 使用当前档位的 `quality/max_width` 编码，并在 `status()` 的 `adaptive` 与每帧 `meta.scene` 中暴露档位，便于客户观测。

## 2. 验证设置

合成 1200×800 画面，2 Hz 驱动 20 s（40 tick），三组场景：

- **static**：画面完全不变（只有一条静态标题条）。
- **low**：每个 tick 只有 10×40 px 的小块在明暗间切换（低变化，但超过 `change_min_ratio`）。
- **dynamic**：160×160 红方块每个 tick 移动（约 2.7% 面积变化）。

每组分别关闭/开启 `adaptive.enabled`，其余配置相同；统计实际发送帧数、有效 fps、单帧字节与图片 token 估算。

## 3. 结果

| 场景 | adaptive | 发送帧 | 有效 fps | 单帧字节 | token 合计 | 结束时档位 |
|---|---|---|---|---|---|---|
| static | off | 1 | 0.05 | 8,092 | 558 | normal |
| static | on | 1 | 0.05 | 8,092 | 558 | **static** |
| low | off | 40 | 2.00 | 8,148 | 22,320 | normal |
| low | on | **29** | **1.45** | **7,434** | **15,170** | **static** |
| dynamic | off | 39 | 1.95 | 9,305 | 21,762 | normal |
| dynamic | on | 39 | 1.95 | 9,469 | 21,762 | **dynamic** |

相对变化：

- **低变化场景**：发送帧 −27.5%（40→29），token −32.0%（22,320→15,170），单帧字节 −8.8%。
- **动态场景**：帧数与 token 不变（39 帧），档位稳定在 `dynamic`，质量略升（照片类画面字节 +1.8%），采样灵敏度没有下降。
- **完全静止场景**：两种配置都只发首帧——静默机制（无变化不发送）先于自适应生效，因此这里没有额外收益；自适应负责的是"**有少量变化但没有实质内容变化**"的灰区。

## 4. 结论

1. 自适应在"低变化但仍在发帧"的场景里确实省：帧率降到 1.45 fps、token 降三成，同时保留了变化可见性。
2. 动态场景不被误降级：39 帧全发，档位为 `dynamic`，没有为了省 token 丢运动信息。
3. 完全静止时收益来自"静默"，不是自适应；两者是互补关系，报告中分开计。
4. 代价：低变化场景的响应延迟从 0.5 s 级升到 1 s 级；对需要"看到每一个小变化"的客户，可把 `static_ratio` 调低或关闭 `adaptive`。

## 5. 复现

```powershell
py -3.12 -u tools/adaptive_eval.py
py -3.12 -m unittest discover -s tests
```

## 6. 限制

- 阈值是按 1200×800、2 Hz 桌面场景调的经验值；4K 屏或高帧率采集需要重新标定 `change_min_ratio`。
- `static` 档的 1 fps 是"发送上限"，不是采集上限；采集线程仍按配置 fps 抓帧，抓帧开销不变，省的是编码与上传。
- 合成场景覆盖"无变化/小变化/明显运动"，未覆盖"大面积滚动"和"视频播放"；后者会进入 `dynamic`，不会省 token（这是设计选择：宁可多花，不漏运动）。
