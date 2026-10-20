"""DXGI Desktop Duplication 采集实现（Windows）。

设计要点（任务书 §2 / §3.1）：
- 直接从显卡复制桌面帧，不经过 GDI；
- 读取 `DXGI_OUTDUPL_FRAME_INFO`，拿 `AccumulatedFrames` 与 dirty-rects；
- 通过 staging texture + Map 取像素，转成 RGB；
- 不依赖任何第三方抓屏库（ctypes 直接调 d3d11.dll / dxgi.dll）；
- vtable 索引由 Windows SDK 头文件解析确认（见 tools/parse_com_vtables.py）。

vtable 索引（解析自 SDK 10.0.26100.0）：
  IDXGIFactory1.EnumAdapters1            = 12
  IDXGIAdapter1.EnumOutputs              = 7
  IDXGIOutput.GetDesc                    = 7
  IDXGIOutput1.DuplicateOutput           = 22
  IDXGIOutputDuplication.GetDesc         = 7
  IDXGIOutputDuplication.AcquireNextFrame= 8
  IDXGIOutputDuplication.GetFrameDirtyRects = 9
  IDXGIOutputDuplication.ReleaseFrame    = 14
  ID3D11Device.CreateTexture2D           = 5
  ID3D11DeviceContext.CopyResource       = 47
  ID3D11DeviceContext.Map                = 14
  ID3D11DeviceContext.Unmap              = 15
"""
from __future__ import annotations

import ctypes
import sys
import time
from ctypes import wintypes as w
from typing import Any

import numpy as np

from ..frames import Rect, RetinaFrame, merge_rects, now_ms
from .base import AccessLost, CaptureError, CaptureUnavailable, RetinaCapture, enable_dpi_awareness

HRESULT = ctypes.c_long

# ---------------------------------------------------------------- 常量 / GUID

DXGI_FORMAT_B8G8R8A8_UNORM = 87
D3D11_USAGE_STAGING = 3
D3D11_CPU_ACCESS_READ = 0x20000
D3D11_MAP_READ = 1
D3D11_SDK_VERSION = 7
D3D_DRIVER_TYPE_UNKNOWN = 0

DXGI_ERROR_INVALID_CALL = 0x887A0001
DXGI_ERROR_NOT_FOUND = 0x887A0002
DXGI_ERROR_MORE_DATA = 0x887A0003
DXGI_ERROR_UNSUPPORTED = 0x887A0004
DXGI_ERROR_DEVICE_REMOVED = 0x887A0005
DXGI_ERROR_NOT_CURRENTLY_AVAILABLE = 0x887A0022
DXGI_ERROR_ACCESS_LOST = 0x887A0026
DXGI_ERROR_WAIT_TIMEOUT = 0x887A0027
DXGI_ERROR_SESSION_DISCONNECTED = 0x887A0028

_ERROR_TEXT = {
    DXGI_ERROR_INVALID_CALL: "参数/状态无效（DXGI_ERROR_INVALID_CALL）",
    DXGI_ERROR_NOT_FOUND: "未找到资源（DXGI_ERROR_NOT_FOUND）",
    DXGI_ERROR_UNSUPPORTED: "当前会话不支持桌面复制（DXGI_ERROR_UNSUPPORTED）",
    DXGI_ERROR_DEVICE_REMOVED: "显卡设备被移除（DXGI_ERROR_DEVICE_REMOVED）",
    DXGI_ERROR_NOT_CURRENTLY_AVAILABLE: "桌面复制暂不可用（DXGI_ERROR_NOT_CURRENTLY_AVAILABLE）",
    DXGI_ERROR_ACCESS_LOST: "桌面复制被系统收回（DXGI_ERROR_ACCESS_LOST）",
    DXGI_ERROR_SESSION_DISCONNECTED: "会话已断开（DXGI_ERROR_SESSION_DISCONNECTED）",
    0x80070005: "拒绝访问（E_ACCESSDENIED）：请在交互式桌面会话中运行",
    0x80070057: "参数错误（E_INVALIDARG）",
}


class GUID(ctypes.Structure):
    _fields_ = [
        ("Data1", w.DWORD),
        ("Data2", w.WORD),
        ("Data3", w.WORD),
        ("Data4", ctypes.c_ubyte * 8),
    ]


def _guid(text: str) -> GUID:
    parts = text.split("-")
    if len(parts) != 5:
        raise ValueError(f"非法 GUID: {text}")
    # GUID 形式：Data1-Data2-Data3-Data4[0:2]-Data4[2:8]，最后两段合起来才是 8 字节 Data4
    data4 = (ctypes.c_ubyte * 8)(*bytes.fromhex(parts[3] + parts[4]))
    return GUID(int(parts[0], 16), int(parts[1], 16), int(parts[2], 16), data4)


IID_IDXGIFactory1 = _guid("770aae78-f26f-4dba-a829-253c83d1b387")
IID_IDXGIAdapter1 = _guid("29038f61-3839-4626-91fd-086879011a05")
IID_IDXGIOutput1 = _guid("00cddea8-939b-4b83-a340-a685226666cc")
IID_IDXGIOutputDuplication = _guid("191cfac3-a341-470d-b26e-a864f428319c")
IID_ID3D11Device = _guid("db6f6ddb-ac77-4e88-8253-819df9bbf140")
IID_ID3D11Texture2D = _guid("6f15aaf2-d208-4e89-9ab4-489535d34f9c")
IID_IDXGIResource = _guid("035f3ab4-482e-4e50-b41f-8a7f8bd8960b")

# ---------------------------------------------------------------- 结构体


class DXGI_MODE_DESC(ctypes.Structure):
    # DXGI_RATIONAL 拍平成两个 UINT，内存布局一致
    _fields_ = [
        ("Width", w.UINT),
        ("Height", w.UINT),
        ("RefreshRateNumerator", w.UINT),
        ("RefreshRateDenominator", w.UINT),
        ("Format", w.UINT),
        ("ScanlineOrdering", w.UINT),
        ("Scaling", w.UINT),
    ]


class DXGI_OUTDUPL_DESC(ctypes.Structure):
    _fields_ = [
        ("ModeDesc", DXGI_MODE_DESC),
        ("Rotation", w.UINT),
        ("DesktopImageInSystemMemory", w.BOOL),
    ]


class POINT(ctypes.Structure):
    _fields_ = [("x", w.LONG), ("y", w.LONG)]


class DXGI_OUTDUPL_POINTER_POSITION(ctypes.Structure):
    _fields_ = [("Position", POINT), ("Visible", w.BOOL)]


class DXGI_OUTDUPL_FRAME_INFO(ctypes.Structure):
    _fields_ = [
        ("LastPresentTime", ctypes.c_longlong),
        ("LastMouseUpdateTime", ctypes.c_longlong),
        ("AccumulatedFrames", w.UINT),
        ("RectsCoalesced", w.BOOL),
        ("ProtectedContentMaskedOut", w.BOOL),
        ("PointerPosition", DXGI_OUTDUPL_POINTER_POSITION),
        ("TotalMetadataBufferSize", w.UINT),
        ("PointerShapeBufferSize", w.UINT),
    ]


class DXGI_OUTPUT_DESC(ctypes.Structure):
    _fields_ = [
        ("DeviceName", w.WCHAR * 32),
        ("DesktopCoordinates", w.RECT),
        ("AttachedToDesktop", w.BOOL),
        ("Rotation", w.UINT),
        ("Monitor", w.HANDLE),
    ]


class LUID(ctypes.Structure):
    _fields_ = [("LowPart", w.DWORD), ("HighPart", w.LONG)]


class DXGI_ADAPTER_DESC1(ctypes.Structure):
    _fields_ = [
        ("Description", w.WCHAR * 128),
        ("VendorId", w.UINT),
        ("DeviceId", w.UINT),
        ("SubSysId", w.UINT),
        ("Revision", w.UINT),
        ("DedicatedVideoMemory", ctypes.c_size_t),
        ("DedicatedSystemMemory", ctypes.c_size_t),
        ("SharedSystemMemory", ctypes.c_size_t),
        ("AdapterLuid", LUID),
        ("Flags", w.UINT),
    ]


class D3D11_TEXTURE2D_DESC(ctypes.Structure):
    # DXGI_SAMPLE_DESC 拍平成两个 UINT
    _fields_ = [
        ("Width", w.UINT),
        ("Height", w.UINT),
        ("MipLevels", w.UINT),
        ("ArraySize", w.UINT),
        ("Format", w.UINT),
        ("SampleCount", w.UINT),
        ("SampleQuality", w.UINT),
        ("Usage", w.UINT),
        ("BindFlags", w.UINT),
        ("CPUAccessFlags", w.UINT),
        ("MiscFlags", w.UINT),
    ]


class D3D11_MAPPED_SUBRESOURCE(ctypes.Structure):
    _fields_ = [("pData", ctypes.c_void_p), ("RowPitch", w.UINT), ("DepthPitch", w.UINT)]


# ---------------------------------------------------------------- COM 调用


def _com(this: Any, index: int, restype: Any, *argtypes: Any):
    """按 vtable 索引取方法并构造可调用函数。"""
    ptr = this if isinstance(this, int) else ctypes.cast(this, ctypes.c_void_p).value
    vtable = ctypes.cast(ptr, ctypes.POINTER(ctypes.c_void_p))[0]
    address = ctypes.cast(vtable, ctypes.POINTER(ctypes.c_void_p))[index]
    return ctypes.WINFUNCTYPE(restype, ctypes.c_void_p, *argtypes)(address)


def _u32(hr: int) -> int:
    return hr & 0xFFFFFFFF


def _hr_text(hr: int) -> str:
    code = _u32(hr)
    return _ERROR_TEXT.get(code, f"HRESULT 0x{code:08X}")


def _check(hr: int, where: str) -> None:
    if hr < 0:
        code = _u32(hr)
        if code in (DXGI_ERROR_ACCESS_LOST, DXGI_ERROR_DEVICE_REMOVED, DXGI_ERROR_SESSION_DISCONNECTED):
            raise AccessLost(f"{where} 失败：{_hr_text(hr)}")
        if code == DXGI_ERROR_UNSUPPORTED:
            raise CaptureUnavailable(
                f"{where} 失败：{_hr_text(hr)}。常见原因：远程桌面会话、锁屏、无桌面会话、显卡驱动过旧"
            )
        raise CaptureError(f"{where} 失败：{_hr_text(hr)}")


def _release(ptr: Any) -> None:
    if not ptr:
        return
    try:
        _com(ptr, 2, w.ULONG)(ptr)
    except Exception:
        pass


def _qi(ptr: Any, iid: GUID) -> ctypes.c_void_p:
    out = ctypes.c_void_p()
    hr = _com(ptr, 0, HRESULT, ctypes.POINTER(GUID), ctypes.POINTER(ctypes.c_void_p))(
        ptr, ctypes.byref(iid), ctypes.byref(out)
    )
    _check(hr, "QueryInterface")
    return out


def _clip_rects(rects: list[Rect], x0: int, y0: int, x1: int, y1: int) -> list[Rect]:
    """把脏区裁剪到 [x0,x1)×[y0,y1)，并平移到区域原点。"""
    out: list[Rect] = []
    for r in rects:
        nx0, ny0 = max(r.x, x0), max(r.y, y0)
        nx1, ny1 = min(r.x + r.w, x1), min(r.y + r.h, y1)
        if nx1 > nx0 and ny1 > ny0:
            out.append(Rect(nx0 - x0, ny0 - y0, nx1 - nx0, ny1 - ny0))
    return out


# ---------------------------------------------------------------- 采集器


class DxgiCapture(RetinaCapture):
    """Windows DXGI Desktop Duplication 采集器。"""

    source = "desktop"

    def __init__(self, output_index: int = 0, fps: float = 2.0, ring_size: int = 8,
                 region: tuple[int, int, int, int] | None = None, **_ignored: Any):
        super().__init__(fps=fps, ring_size=ring_size, region=region)
        self.output_index = int(output_index)
        self._d3d11 = None
        self._dxgi = None
        self._factory = ctypes.c_void_p()
        self._adapter = ctypes.c_void_p()
        self._output = ctypes.c_void_p()
        self._device = ctypes.c_void_p()
        self._context = ctypes.c_void_p()
        self._dupl = ctypes.c_void_p()
        self._staging = ctypes.c_void_p()
        self._width = 0
        self._height = 0
        self._rotation = 0
        self._surface_width = 0
        self._surface_height = 0
        self._adapter_name = ""
        self._output_name = ""
        self._dpi_mode = "unaware"
        self._last_present = 0
        self._dirty_buffer = ctypes.create_string_buffer(64 * 16)  # 最多 16 个矩形起步，按需扩容
        self._dirty_capacity = 16
        self._com_ready = False

    # ---- 生命周期 ----

    def open(self) -> "DxgiCapture":
        if self._opened:
            return self
        if sys.platform != "win32":
            raise CaptureUnavailable("DXGI 采集仅支持 Windows")
        self._dpi_mode = enable_dpi_awareness()
        self._open_backend()
        self._opened = True
        return self

    def _open_backend(self) -> None:
        if sys.platform != "win32":
            raise CaptureUnavailable("DXGI 采集仅支持 Windows")
        self._load_dlls()
        self._create_factory_and_output()
        self._create_device()
        self._create_duplication()

    def _close_backend(self) -> None:
        for name in ("_staging", "_dupl", "_context", "_device", "_output", "_adapter", "_factory"):
            _release(getattr(self, name))
            setattr(self, name, ctypes.c_void_p())
        self._com_ready = False

    # ---- 初始化步骤 ----

    def _load_dlls(self) -> None:
        if self._d3d11 is not None:
            return
        self._d3d11 = ctypes.WinDLL("d3d11")
        self._dxgi = ctypes.WinDLL("dxgi")
        self._dxgi.CreateDXGIFactory1.argtypes = [ctypes.POINTER(GUID), ctypes.POINTER(ctypes.c_void_p)]
        self._dxgi.CreateDXGIFactory1.restype = HRESULT
        self._d3d11.D3D11CreateDevice.argtypes = [
            ctypes.c_void_p, ctypes.c_int, ctypes.c_void_p, w.UINT,
            ctypes.POINTER(w.UINT), w.UINT, w.UINT,
            ctypes.POINTER(ctypes.c_void_p), ctypes.POINTER(w.UINT), ctypes.POINTER(ctypes.c_void_p),
        ]
        self._d3d11.D3D11CreateDevice.restype = HRESULT

    def _create_factory_and_output(self) -> None:
        factory = ctypes.c_void_p()
        hr = self._dxgi.CreateDXGIFactory1(ctypes.byref(IID_IDXGIFactory1), ctypes.byref(factory))
        _check(hr, "CreateDXGIFactory1")
        self._factory = factory

        adapters: list[ctypes.c_void_p] = []
        outputs: list[tuple[int, ctypes.c_void_p]] = []  # (adapter 下标, output)
        adapter_index = 0
        while True:
            adapter = ctypes.c_void_p()
            hr = _com(self._factory, 12, HRESULT, w.UINT, ctypes.POINTER(ctypes.c_void_p))(
                self._factory, adapter_index, ctypes.byref(adapter)
            )
            if _u32(hr) == DXGI_ERROR_NOT_FOUND or hr < 0:
                break
            adapters.append(adapter)
            output_index = 0
            while True:
                output = ctypes.c_void_p()
                hr_out = _com(adapter, 7, HRESULT, w.UINT, ctypes.POINTER(ctypes.c_void_p))(
                    adapter, output_index, ctypes.byref(output)
                )
                if _u32(hr_out) == DXGI_ERROR_NOT_FOUND or hr_out < 0:
                    break
                desc = DXGI_OUTPUT_DESC()
                hr_desc = _com(output, 7, HRESULT, ctypes.POINTER(DXGI_OUTPUT_DESC))(output, ctypes.byref(desc))
                if hr_desc >= 0 and desc.AttachedToDesktop:
                    outputs.append((len(adapters) - 1, output))
                else:
                    _release(output)
                output_index += 1
            adapter_index += 1

        if not outputs:
            for adapter in adapters:
                _release(adapter)
            raise CaptureUnavailable("未找到可复制的显示输出（没有已连接的显示器）")
        if self.output_index >= len(outputs):
            for _, output in outputs:
                _release(output)
            for adapter in adapters:
                _release(adapter)
            raise CaptureUnavailable(
                f"output_index={self.output_index} 超出范围，当前可用输出 {len(outputs)} 个"
            )

        selected_adapter_index, selected_output = outputs[self.output_index]
        self._adapter = adapters[selected_adapter_index]
        self._output = selected_output
        # 释放未选中的 output/adapters，避免占用桌面复制名额
        for _, other_output in outputs:
            if other_output.value != selected_output.value:
                _release(other_output)
        for i, adapter in enumerate(adapters):
            if i != selected_adapter_index:
                _release(adapter)

        desc = DXGI_OUTPUT_DESC()
        _check(_com(self._output, 7, HRESULT, ctypes.POINTER(DXGI_OUTPUT_DESC))(self._output, ctypes.byref(desc)),
               "IDXGIOutput::GetDesc")
        self._output_name = ctypes.wstring_at(desc.DeviceName)
        self._rotation = int(desc.Rotation)
        self._output_rect = (desc.DesktopCoordinates.left, desc.DesktopCoordinates.top,
                             desc.DesktopCoordinates.right, desc.DesktopCoordinates.bottom)

        # 适配器名称
        try:
            adapter1 = _qi(self._adapter, IID_IDXGIAdapter1)
            try:
                desc1 = DXGI_ADAPTER_DESC1()
                hr = _com(adapter1, 10, HRESULT, ctypes.POINTER(DXGI_ADAPTER_DESC1))(adapter1, ctypes.byref(desc1))
                if hr >= 0:
                    self._adapter_name = ctypes.wstring_at(desc1.Description)
                    self._adapter_vendor = int(desc1.VendorId)
                    self._adapter_device_id = int(desc1.DeviceId)
                    self._adapter_dedicated_vram = int(desc1.DedicatedVideoMemory)
            finally:
                _release(adapter1)
        except Exception:
            self._adapter_name = ""

    def _create_device(self) -> None:
        device = ctypes.c_void_p()
        feature = w.UINT(0)
        context = ctypes.c_void_p()
        levels = (w.UINT * 4)(0xB100, 0xB000, 0xA100, 0xA000)
        hr = self._d3d11.D3D11CreateDevice(
            self._adapter, D3D_DRIVER_TYPE_UNKNOWN, None, 0,
            levels, 4, D3D11_SDK_VERSION,
            ctypes.byref(device), ctypes.byref(feature), ctypes.byref(context),
        )
        if hr < 0 and _u32(hr) == 0x80070057:  # Win7 平台不支持 11_1，去掉后重试
            levels = (w.UINT * 3)(0xB000, 0xA100, 0xA000)
            hr = self._d3d11.D3D11CreateDevice(
                self._adapter, D3D_DRIVER_TYPE_UNKNOWN, None, 0,
                levels, 3, D3D11_SDK_VERSION,
                ctypes.byref(device), ctypes.byref(feature), ctypes.byref(context),
            )
        _check(hr, "D3D11CreateDevice")
        self._device = device
        self._context = context
        self._feature_level = int(feature.value)

    def _create_duplication(self) -> None:
        dupl = ctypes.c_void_p()
        output1 = _qi(self._output, IID_IDXGIOutput1)
        try:
            hr = _com(output1, 22, HRESULT, ctypes.c_void_p, ctypes.POINTER(ctypes.c_void_p))(
                output1, self._device, ctypes.byref(dupl)
            )
        finally:
            _release(output1)
        _check(hr, "IDXGIOutput1::DuplicateOutput")
        self._dupl = dupl

        desc = DXGI_OUTDUPL_DESC()
        _check(_com(self._dupl, 7, HRESULT, ctypes.POINTER(DXGI_OUTDUPL_DESC))(self._dupl, ctypes.byref(desc)),
               "IDXGIOutputDuplication::GetDesc")
        self._surface_width = int(desc.ModeDesc.Width)
        self._surface_height = int(desc.ModeDesc.Height)
        self._width, self._height = (
            (self._surface_height, self._surface_width) if self._rotation in (2, 4)  # ROTATE90 / ROTATE270
            else (self._surface_width, self._surface_height)
        )
        self._desktop_in_sysmem = bool(desc.DesktopImageInSystemMemory)
        self._last_present = 0
        # staging texture：CPU 可读，用于把 GPU 上的桌面面拷贝到内存
        tex_desc = D3D11_TEXTURE2D_DESC()
        tex_desc.Width = self._surface_width
        tex_desc.Height = self._surface_height
        tex_desc.MipLevels = 1
        tex_desc.ArraySize = 1
        tex_desc.Format = DXGI_FORMAT_B8G8R8A8_UNORM
        tex_desc.SampleCount = 1
        tex_desc.SampleQuality = 0
        tex_desc.Usage = D3D11_USAGE_STAGING
        tex_desc.BindFlags = 0
        tex_desc.CPUAccessFlags = D3D11_CPU_ACCESS_READ
        tex_desc.MiscFlags = 0
        staging = ctypes.c_void_p()
        hr = _com(self._device, 5, HRESULT, ctypes.POINTER(D3D11_TEXTURE2D_DESC), ctypes.c_void_p,
                  ctypes.POINTER(ctypes.c_void_p))(self._device, ctypes.byref(tex_desc), None, ctypes.byref(staging))
        _check(hr, "ID3D11Device::CreateTexture2D")
        self._staging = staging
        self._com_ready = True

    # ---- 取帧 ----

    def _read(self, timeout_ms: int) -> RetinaFrame | None:
        if not self._com_ready:
            raise AccessLost("桌面复制尚未就绪")
        info = DXGI_OUTDUPL_FRAME_INFO()
        resource = ctypes.c_void_p()
        hr = _com(self._dupl, 8, HRESULT, w.UINT, ctypes.POINTER(DXGI_OUTDUPL_FRAME_INFO),
                  ctypes.POINTER(ctypes.c_void_p))(
            self._dupl, timeout_ms, ctypes.byref(info), ctypes.byref(resource)
        )
        code = _u32(hr)
        if code == DXGI_ERROR_WAIT_TIMEOUT:
            return None
        if hr < 0:
            _check(hr, "AcquireNextFrame")
        try:
            if info.AccumulatedFrames == 0:
                return None  # 只有鼠标移动，没有新的桌面像素
            dirty = self._read_dirty_rects()
            texture = _qi(resource, IID_ID3D11Texture2D)
            try:
                # CopyResource 返回 void，不要拿返回值当 HRESULT 比较
                _com(self._context, 47, None, ctypes.c_void_p, ctypes.c_void_p)(
                    self._context, self._staging, texture
                )
                pixels = self._map_staging()
            finally:
                _release(texture)
        finally:
            _release(resource)
            _com(self._dupl, 14, HRESULT)(self._dupl)  # ReleaseFrame

        pixels = self._rotate(pixels)
        dirty = self._rotate_rects(dirty)
        height, width = pixels.shape[:2]
        x0, y0, x1, y1 = self._region_bounds(width, height)
        if (x0, y0, x1, y1) != (0, 0, width, height):
            dirty = _clip_rects(dirty, x0, y0, x1, y1)
            pixels = pixels[y0:y1, x0:x1].copy()
        self._last_present = int(info.LastPresentTime)
        return RetinaFrame(
            frame_id=0,
            timestamp_ms=now_ms(),
            source="desktop",
            pixels=pixels,
            dirty_rects=dirty,
            is_key_frame=not dirty,
            meta={
                "backend": "dxgi",
                "accumulated_frames": int(info.AccumulatedFrames),
                "last_present_time": int(info.LastPresentTime),
                "protected_content": bool(info.ProtectedContentMaskedOut),
                "rotation": self._rotation,
                "output_index": self.output_index,
                "adapter": self._adapter_name,
                "device": self._output_name,
                "feature_level": getattr(self, "_feature_level", None),
            },
        )

    def _read_dirty_rects(self) -> list[Rect]:
        """读取本帧 dirty-rects；缓冲区不足时按返回的 required 扩容重试。"""
        for _ in range(3):
            buffer = ctypes.create_string_buffer(self._dirty_capacity * ctypes.sizeof(w.RECT))
            required = w.UINT(0)
            hr = _com(self._dupl, 9, HRESULT, w.UINT, ctypes.c_void_p, ctypes.POINTER(w.UINT))(
                self._dupl, self._dirty_capacity * ctypes.sizeof(w.RECT),
                ctypes.cast(buffer, ctypes.c_void_p), ctypes.byref(required)
            )
            code = _u32(hr)
            if hr >= 0:
                count = required.value // ctypes.sizeof(w.RECT)
                if count == 0:
                    return []
                rects = ctypes.cast(buffer, ctypes.POINTER(w.RECT * count)).contents
                result: list[Rect] = []
                for i in range(count):
                    r = rects[i]
                    if r.right <= r.left or r.bottom <= r.top:
                        continue
                    result.append(Rect.from_ltrb(r.left, r.top, r.right, r.bottom))
                return merge_rects(result, gap=8)
            if code == DXGI_ERROR_MORE_DATA and required.value:
                self._dirty_capacity = max(1, required.value // ctypes.sizeof(w.RECT) + 2)
                continue
            # 某些驱动在无 metadata 时返回 INVALID_CALL：当作整帧处理
            if code == DXGI_ERROR_INVALID_CALL:
                return []
            _check(hr, "GetFrameDirtyRects")
        return []

    def _map_staging(self) -> np.ndarray:
        mapped = D3D11_MAPPED_SUBRESOURCE()
        hr = _com(self._context, 14, HRESULT, ctypes.c_void_p, w.UINT, w.UINT, w.UINT,
                  ctypes.POINTER(D3D11_MAPPED_SUBRESOURCE))(
            self._context, self._staging, 0, D3D11_MAP_READ, 0, ctypes.byref(mapped)
        )
        _check(hr, "ID3D11DeviceContext::Map")
        try:
            row_pitch = int(mapped.RowPitch)
            total = row_pitch * self._surface_height
            raw = (ctypes.c_ubyte * total).from_address(mapped.pData)
            rows = np.frombuffer(raw, dtype=np.uint8).reshape(self._surface_height, row_pitch)
            bgra = rows[:, : self._surface_width * 4].reshape(self._surface_height, self._surface_width, 4)
            rgb = np.ascontiguousarray(bgra[:, :, [2, 1, 0]])
            return rgb
        finally:
            _com(self._context, 15, None, ctypes.c_void_p, w.UINT)(self._context, self._staging, 0)

    def _rotate(self, pixels: np.ndarray) -> np.ndarray:
        # DXGI_MODE_ROTATION: 0=UNSPECIFIED, 1=IDENTITY, 2=ROTATE90, 3=ROTATE180, 4=ROTATE270
        if self._rotation == 2:      # 顺时针 90°
            return np.ascontiguousarray(np.rot90(pixels, k=-1))
        if self._rotation == 3:      # 180°
            return np.ascontiguousarray(np.rot90(pixels, k=2))
        if self._rotation == 4:      # 顺时针 270°
            return np.ascontiguousarray(np.rot90(pixels, k=1))
        return pixels

    def _rotate_rects(self, rects: list[Rect]) -> list[Rect]:
        """把未旋转表面坐标下的 dirty-rects 映射到旋转后的屏幕坐标。"""
        if not rects or self._rotation not in (2, 3, 4):
            return rects
        sw, sh = self._surface_width, self._surface_height
        out: list[Rect] = []
        for r in rects:
            if self._rotation == 2:      # 顺时针 90°：(x,y) -> (sh-(y+h), x)
                out.append(Rect(sh - (r.y + r.h), r.x, r.h, r.w))
            elif self._rotation == 3:    # 180°
                out.append(Rect(sw - (r.x + r.w), sh - (r.y + r.h), r.w, r.h))
            else:                        # 顺时针 270°：(x,y) -> (y, sw-(x+w))
                out.append(Rect(r.y, sw - (r.x + r.w), r.h, r.w))
        return out

    # ---- 状态 ----

    def describe(self) -> dict[str, Any]:
        info = super().describe()
        info.update({
            "backend": "dxgi",
            "output_index": self.output_index,
            "adapter": self._adapter_name,
            "device": self._output_name,
            "width": self._width,
            "height": self._height,
            "rotation": self._rotation,
            "desktop_image_in_system_memory": getattr(self, "_desktop_in_sysmem", None),
            "dpi_mode": self._dpi_mode,
        })
        return info

    @classmethod
    def list_sources(cls) -> list[dict[str, Any]]:
        """枚举当前可复制的显示输出（不建立复制，避免占用名额）。"""
        if sys.platform != "win32":
            return []
        enable_dpi_awareness()  # 否则输出坐标会被 DPI 虚拟化成逻辑像素
        try:
            dxgi = ctypes.WinDLL("dxgi")
            factory = ctypes.c_void_p()
            hr = dxgi.CreateDXGIFactory1(ctypes.byref(IID_IDXGIFactory1), ctypes.byref(factory))
            if hr < 0:
                return []
            results: list[dict[str, Any]] = []
            adapter_index = 0
            while True:
                adapter = ctypes.c_void_p()
                hr_a = _com(factory, 12, HRESULT, w.UINT, ctypes.POINTER(ctypes.c_void_p))(
                    factory, adapter_index, ctypes.byref(adapter))
                if hr_a < 0:
                    break
                output_index = 0
                while True:
                    output = ctypes.c_void_p()
                    hr_o = _com(adapter, 7, HRESULT, w.UINT, ctypes.POINTER(ctypes.c_void_p))(
                        adapter, output_index, ctypes.byref(output))
                    if hr_o < 0:
                        break
                    desc = DXGI_OUTPUT_DESC()
                    if _com(output, 7, HRESULT, ctypes.POINTER(DXGI_OUTPUT_DESC))(output, ctypes.byref(desc)) >= 0 \
                            and desc.AttachedToDesktop:
                        r = desc.DesktopCoordinates
                        results.append({
                            "name": f"desktop:{len(results)}",
                            "source": "desktop",
                            "backend": "dxgi",
                            "device": ctypes.wstring_at(desc.DeviceName),
                            "x": r.left, "y": r.top, "width": r.right - r.left, "height": r.bottom - r.top,
                            "primary": len(results) == 0,
                            "available": True,
                        })
                    _release(output)
                    output_index += 1
                _release(adapter)
                adapter_index += 1
            _release(factory)
            return results
        except Exception:
            return []


__all__ = ["DxgiCapture"]
