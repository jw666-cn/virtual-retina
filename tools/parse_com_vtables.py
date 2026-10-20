"""Parse COM interface method order from the Windows SDK headers.

Used to derive exact vtable slot indices for the ctypes DXGI/D3D11 bindings.
Run:  py -3.12 tools/parse_com_vtables.py
"""
from __future__ import annotations

import re
import sys
from pathlib import Path

SDK_INCLUDE = Path(r"C:\Program Files (x86)\Windows Kits\10\Include\10.0.26100.0")
HEADERS = [
    SDK_INCLUDE / "shared" / "dxgi.h",
    SDK_INCLUDE / "shared" / "dxgi1_2.h",
    SDK_INCLUDE / "um" / "d3d11.h",
]

IFACE_RE = re.compile(
    r'MIDL_INTERFACE\(\s*"(?P<iid>[0-9a-fA-F-]+)"\s*\)\s*\n\s*(?P<name>\w+)\s*:\s*public\s+(?P<parent>\w+)\s*\n\s*\{'
)
METHOD_RE = re.compile(r"virtual\s+[\w\s\*&]+?STDMETHODCALLTYPE\s+(?P<name>\w+)\s*\(")


def load_interfaces() -> dict[str, dict]:
    interfaces: dict[str, dict] = {}
    for header in HEADERS:
        if not header.exists():
            continue
        text = header.read_text(encoding="utf-8", errors="ignore")
        for match in IFACE_RE.finditer(text):
            start = match.end()
            depth = 1
            i = start
            while i < len(text) and depth:
                ch = text[i]
                if ch == "{":
                    depth += 1
                elif ch == "}":
                    depth -= 1
                i += 1
            body = text[start:i]
            methods = [m.group("name") for m in METHOD_RE.finditer(body)]
            interfaces[match.group("name")] = {
                "iid": match.group("iid"),
                "parent": match.group("parent"),
                "methods": methods,
            }
    return interfaces


def base_index(interfaces: dict[str, dict], name: str, _seen: set[str] | None = None) -> int:
    """Number of vtable slots occupied by an interface and all its ancestors."""
    if name == "IUnknown":
        return 3
    if name not in interfaces:
        raise KeyError(f"interface {name} not found in headers")
    seen = _seen or set()
    if name in seen:
        raise RuntimeError(f"inheritance cycle at {name}")
    seen.add(name)
    iface = interfaces[name]
    return base_index(interfaces, iface["parent"], seen) + len(iface["methods"])


def table(interfaces: dict[str, dict], name: str) -> list[tuple[int, str]]:
    chain: list[dict] = []
    cur = name
    while cur != "IUnknown":
        if cur not in interfaces:
            raise KeyError(cur)
        chain.append(interfaces[cur])
        cur = interfaces[cur]["parent"]
    chain.reverse()
    index = 3
    rows: list[tuple[int, str]] = []
    for iface in chain:
        for method in iface["methods"]:
            rows.append((index, method))
            index += 1
    return rows


if __name__ == "__main__":
    ifaces = load_interfaces()
    wanted = [
        "IDXGIFactory1",
        "IDXGIAdapter1",
        "IDXGIOutput",
        "IDXGIOutput1",
        "IDXGIOutputDuplication",
        "IDXGIResource",
        "ID3D11Device",
        "ID3D11DeviceContext",
        "IDXGISwapChain1",
    ]
    for name in wanted:
        if name not in ifaces:
            print(f"--- {name}: NOT FOUND ---")
            continue
        print(f"--- {name} (iid {ifaces[name]['iid']}) ---")
        for idx, method in table(ifaces, name):
            print(f"{idx:3d}  {method}")
        print()
    sys.exit(0)
