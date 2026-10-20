# Phase 3：摄像头、跨平台与虚拟屏方案

> 状态：摄像头链路已在 Windows 用 OpenCV 打通（无实体摄像头，按 §10 用视频文件模拟 UVC 验证）；
> macOS / Linux 为**接口就绪、实现待补**；IDD 虚拟屏为**驱动级预留**，本期未构建。

## 1. 摄像头（Windows，已完成）

- 实现：`vision_retina/capture/camera.py` → `CameraCapture`，走 `cv2.VideoCapture`（Windows 用 `CAP_DSHOW`）。
- 统一：输出同样的 `RetinaFrame(source="camera")`，进入 `RetinaBus`，享受差分/静默/中央凹/编码缓存。
- 无硬件时的联调：`camera.path` 可指向视频文件，播完自动循环（`tools/make_test_video.py` 生成受控场景）。
- 实测：`tools/reality_loop.py` 用 640x480 测试视频，模型正确答出“红色圆形、绿色三角形、蓝色正方形、RETINA 2026”，
  prompt_tokens=282，模型问答耗时 1,333 ms（不含采集/启动）。
- 实体摄像头：按任务书 §10 留占位，采购后把 `camera.index` 改成设备序号即可，无需改上层。

## 2. macOS（接口就绪，未实现）

`vision_retina/capture/macos.py::MacDisplayCapture` 已实现 `RetinaCapture` 全部抽象方法，调用时明确抛
`CaptureUnavailable`，不会返回假帧。补齐计划：

1. **桌面**：ScreenCaptureKit（macOS 12.3+）`SCStream` + `SCStreamOutput`，或旧系统 CGDisplayStream。
   - 权限：首次调用触发“屏幕录制”授权；未授权时返回明确错误。
   - 脏区：`SCStreamFrameInfoDirtyRects` / `CGDisplayStreamUpdateGetRects`，映射为 `Rect`。
   - 输出 BGRA → 本项目的 RGB `uint8` 数组（与 DXGI 后端一致）。
2. **摄像头**：直接复用 `CameraCapture`（OpenCV/AVFoundation），无需改总线。
3. **虚拟屏**：macOS 无官方 IDD 等价物，第三方方案（如 CGVirtualDisplay 私有 API）风险高，暂不做。

## 3. Linux（接口就绪，未实现）

`vision_retina/capture/linux.py::LinuxDisplayCapture` 同上。补齐计划：

| 会话 | 方案 | 脏区 |
|---|---|---|
| X11 | `XGetImage` / `XShmGetImage`（ctypes 或 python-xlib） | 无，靠总线 `grid_diff` |
| Wayland | `xdg-desktop-portal` ScreenCast + PipeWire（`gst`/`pipewire` 绑定） | 无，靠 `grid_diff` |

Wayland 需要用户在 portal 弹窗授权，且不同合成器行为有差异；实现时把“授权失败/用户拒绝”作为明确错误返回。
摄像头同样复用 `CameraCapture`（V4L2）。

## 4. IDD 虚拟屏（Windows，驱动级，未实现）

用途：无物理显示器 / 远程会话 / CI 机器上提供一块“可采集的虚拟桌面”，让 DXGI 有输出可抓。

方案（IddCx，UMDF2 驱动）：

1. 用 WDK（本机已装 10.0.26100.0）+ Visual Studio 建 IddCx 间接显示驱动工程；
2. 实现 `EvtIddCxAdapterInitFinished` / `EvtIddCxMonitorArrival`，创建虚拟显示器；
3. 通过 `IDARG_OUT_*` 暴露帧缓冲，或用 `IddCxSwapChain` 把桌面合成结果交给用户态采集进程；
4. 安装需要管理员权限 + 驱动签名（开发期用测试签名模式 `bcdedit /set testsigning on`，需重启）。

本期不构建的原因：

- 内核驱动需要 WDK 工程、签名与重启，无法在当前会话内安全完成与验证；
- 当前机器有真实 GPU/显示器，DXGI 采集已满足 Phase 1–3 的全部验收；
- 任务书 §8 将其标为“架构预留”，本期只要求接口/方案就绪。

交付物：本节方案 + `RetinaCapture` 抽象（虚拟屏将来作为 `DxgiCapture(output_index=N)` 的一个输出，不需要改总线）。

## 5. 三平台接口一致性

| 能力 | Windows | macOS | Linux |
|---|---|---|---|
| 桌面采集 | ✅ DXGI（脏区）+ GDI 兜底 | 🟡 接口就绪 | 🟡 接口就绪 |
| 摄像头 | ✅ OpenCV（含视频文件回退） | ✅ 复用 OpenCV（未实测） | ✅ 复用 OpenCV（未实测） |
| 虚拟屏 | 🟡 方案就绪（IddCx） | — | 🟡 X dummy / portal 方案 |
| 总线/编码/大脑/手 | ✅ 跨平台纯 Python | ✅ | ✅ |

图例：✅ 已实现并验证；🟡 接口/方案就绪、实现待补；— 不适用。
