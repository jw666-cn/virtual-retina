"""Lightweight CAD drafting core: draw in paper millimetres, output PNG + DXF.

Both drawings are authored in paper coordinates (mm) so the sheet, title block,
tables and views share one coordinate system.  The DXF is written in model
units = paper millimetres (1 unit = 1 mm), which is the normal convention for a
plotted sheet.
"""
from __future__ import annotations

import math
from dataclasses import dataclass, field
from pathlib import Path

from PIL import Image, ImageDraw, ImageFont

try:  # ezdxf is available in this environment; keep the PNG path independent.
    import ezdxf
except Exception:  # pragma: no cover
    ezdxf = None

FONT_CANDIDATES = [
    r"C:\Windows\Fonts\msyh.ttc",
    r"C:\Windows\Fonts\simhei.ttf",
    r"C:\Windows\Fonts\simsun.ttc",
]

LAYERS = {
    "OUTLINE": dict(color=7, width=0.50, rgb=(20, 20, 20)),
    "THIN": dict(color=8, width=0.25, rgb=(90, 90, 90)),
    "CENTER": dict(color=1, width=0.20, rgb=(200, 40, 40)),
    "DIM": dict(color=3, width=0.18, rgb=(30, 90, 160)),
    "TEXT": dict(color=7, width=0.18, rgb=(10, 10, 10)),
    "HATCH": dict(color=8, width=0.13, rgb=(120, 120, 120)),
    "FRAME": dict(color=7, width=0.70, rgb=(0, 0, 0)),
    "TABLE": dict(color=7, width=0.25, rgb=(40, 40, 40)),
    "FILL": dict(color=8, width=0.10, rgb=(225, 225, 225)),
}


def _font(size_px: int, bold: bool = False):
    path = FONT_CANDIDATES[0]
    if bold:
        for p in (r"C:\Windows\Fonts\msyhbd.ttc", r"C:\Windows\Fonts\simhei.ttf"):
            if Path(p).exists():
                path = p
                break
    return ImageFont.truetype(path, max(6, int(size_px)))


def _dash(p1, p2, dash=6.0, gap=2.0):
    """Yield dashed sub-segments between p1 and p2 (paper mm)."""
    x1, y1 = p1
    x2, y2 = p2
    length = math.hypot(x2 - x1, y2 - y1)
    if length == 0:
        return
    ux, uy = (x2 - x1) / length, (y2 - y1) / length
    t = 0.0
    while t < length:
        t2 = min(t + dash, length)
        yield ((x1 + ux * t, y1 + uy * t), (x1 + ux * t2, y1 + uy * t2))
        t = t2 + gap


def _hatch_lines(poly, angle=45.0, spacing=3.0):
    """Return clipped hatch segments for a polygon (even-odd rule)."""
    if len(poly) < 3:
        return []
    a = math.radians(angle)
    dx, dy = math.cos(a), math.sin(a)
    nx, ny = -dy, dx
    pts = list(poly)
    # Project polygon on normal axis.
    vals = [x * nx + y * ny for x, y in pts]
    lo, hi = min(vals), max(vals)
    segs = []
    v = math.floor(lo / spacing) * spacing
    while v <= hi:
        # Line: P = v*N + t*D.  Intersect with every polygon edge.
        hits = []
        n = len(pts)
        for i in range(n):
            x1, y1 = pts[i]
            x2, y2 = pts[(i + 1) % n]
            # edge in (normal, direction) coordinates
            p1n, p1d = x1 * nx + y1 * ny, x1 * dx + y1 * dy
            p2n, p2d = x2 * nx + y2 * ny, x2 * dx + y2 * dy
            if (p1n - v) * (p2n - v) <= 0 and p1n != p2n:
                t = (v - p1n) / (p2n - p1n)
                hits.append(p1d + t * (p2d - p1d))
        hits.sort()
        for i in range(0, len(hits) - 1, 2):
            t1, t2 = hits[i], hits[i + 1]
            segs.append(
                (
                    (v * nx + t1 * dx, v * ny + t1 * dy),
                    (v * nx + t2 * dx, v * ny + t2 * dy),
                )
            )
        v += spacing
    return segs


@dataclass
class Draft:
    sheet_w: float = 841.0
    sheet_h: float = 594.0
    items: list = field(default_factory=list)

    # ---------------------------------------------------------------- geometry
    def line(self, p1, p2, layer="OUTLINE", width=None):
        self.items.append(("line", p1, p2, layer, width))

    def dashed(self, p1, p2, layer="THIN", dash=6.0, gap=2.0, width=None):
        for a, b in _dash(p1, p2, dash, gap):
            self.line(a, b, layer, width)

    def poly(self, pts, closed=False, layer="OUTLINE", width=None):
        pts = list(pts)
        if closed and pts:
            pts = pts + [pts[0]]
        for a, b in zip(pts, pts[1:]):
            self.line(a, b, layer, width)

    def rect(self, x0, y0, x1, y1, layer="OUTLINE", width=None):
        self.poly([(x0, y0), (x1, y0), (x1, y1), (x0, y1)], True, layer, width)

    def circle(self, c, r, layer="OUTLINE", width=None):
        self.items.append(("circle", c, r, layer, width))

    def arc(self, c, r, a0, a1, layer="OUTLINE", width=None):
        self.items.append(("arc", c, r, a0, a1, layer, width))

    def center_h(self, x1, x2, y, layer="CENTER"):
        self.dashed((x1, y), (x2, y), layer, dash=8, gap=2)

    def center_v(self, x, y1, y2, layer="CENTER"):
        self.dashed((x, y1), (x, y2), layer, dash=8, gap=2)

    def hatch(self, pts, angle=45.0, spacing=3.0, layer="HATCH"):
        self.items.append(("hatch", list(pts), angle, spacing, layer))

    def fill(self, pts, rgb=(232, 232, 232), layer="FILL"):
        self.items.append(("fill", list(pts), rgb, layer))

    # ------------------------------------------------------------------- text
    def text(self, p, s, h=3.0, layer="TEXT", align="left", rot=0, bold=False):
        self.items.append(("text", p, str(s), h, layer, align, rot, bold))

    # -------------------------------------------------------------- dimensions
    def _arrow(self, tip, direction, size=2.2, layer="DIM"):
        dx, dy = direction
        _len = math.hypot(dx, dy) or 1.0
        dx, dy = dx / _len, dy / _len
        # small filled triangle
        px, py = -dy, dx
        p1 = (tip[0] - dx * size + px * size * 0.35, tip[1] - dy * size + py * size * 0.35)
        p2 = (tip[0] - dx * size - px * size * 0.35, tip[1] - dy * size - py * size * 0.35)
        self.fill([tip, p1, p2], rgb=(30, 90, 160), layer=layer)

    def dim_h(self, x1, x2, y, y_ref=None, text=None, layer="DIM", h=2.6, outside=0.0):
        y_ref = y if y_ref is None else y_ref
        self.line((x1, y_ref), (x1, y), layer, 0.15)
        self.line((x2, y_ref), (x2, y), layer, 0.15)
        self.line((x1, y), (x2, y), layer, 0.18)
        self._arrow((x1, y), (1, 0), layer=layer)
        self._arrow((x2, y), (-1, 0), layer=layer)
        label = text if text is not None else f"{abs(x2 - x1):g}"
        self.text(((x1 + x2) / 2, y + 0.8 + outside), label, h, layer, "center")

    def dim_v(self, y1, y2, x, x_ref=None, text=None, layer="DIM", h=2.6, outside=0.0):
        x_ref = x if x_ref is None else x_ref
        self.line((x_ref, y1), (x, y1), layer, 0.15)
        self.line((x_ref, y2), (x, y2), layer, 0.15)
        self.line((x, y1), (x, y2), layer, 0.18)
        self._arrow((x, y1), (0, 1), layer=layer)
        self._arrow((x, y2), (0, -1), layer=layer)
        label = text if text is not None else f"{abs(y2 - y1):g}"
        self.text((x - 0.8 - outside, (y1 + y2) / 2), label, h, layer, "right", rot=90)

    def leader(self, pts, s, h=2.4, layer="DIM", align="left"):
        pts = list(pts)
        self.poly(pts, False, layer, 0.15)
        self._arrow(pts[0], (pts[0][0] - pts[1][0], pts[0][1] - pts[1][1]), 2.0, layer)
        self.text(pts[-1], s, h, layer, align)

    # ------------------------------------------------------------------ tables
    def table(self, x, y_top, col_w, row_h, data, header=None, h=2.3, layer="TABLE"):
        """Simple table, top-left anchor, rows downwards."""
        ncol = len(col_w)
        nrow = len(data) + (1 if header else 0)
        width = sum(col_w)
        height = row_h * nrow
        y_bottom = y_top - height
        self.rect(x, y_bottom, x + width, y_top, layer)
        # horizontal lines
        for i in range(1, nrow):
            yy = y_top - i * row_h
            self.line((x, yy), (x + width, yy), layer, 0.2)
        # vertical lines
        xx = x
        for w in col_w[:-1]:
            xx += w
            self.line((xx, y_bottom), (xx, y_top), layer, 0.2)
        rows = ([header] if header else []) + list(data)
        for r, row in enumerate(rows):
            yy = y_top - r * row_h
            bold = bool(header) and r == 0
            xx = x
            for c, cell in enumerate(row):
                self.text((xx + 1.0, yy - row_h + (row_h - h) / 2), cell, h, layer, "left", bold=bold)
                xx += col_w[c]

    # ------------------------------------------------------------------ output
    def render(self, path, ppm=4.0, bg=(255, 255, 255)):
        w = int(self.sheet_w * ppm)
        h = int(self.sheet_h * ppm)
        img = Image.new("RGB", (w, h), bg)
        d = ImageDraw.Draw(img)

        def P(p):
            return (p[0] * ppm, (self.sheet_h - p[1]) * ppm)

        def W(layer, width=None):
            base = width if width is not None else LAYERS.get(layer, LAYERS["THIN"])["width"]
            return max(1, int(round(base * ppm)))

        def C(layer):
            return LAYERS.get(layer, LAYERS["THIN"])["rgb"]

        for item in self.items:
            kind = item[0]
            if kind == "line":
                _, p1, p2, layer, width = item
                d.line([P(p1), P(p2)], fill=C(layer), width=W(layer, width))
            elif kind == "circle":
                _, c, r, layer, width = item
                cx, cy = P(c)
                rr = r * ppm
                d.ellipse([cx - rr, cy - rr, cx + rr, cy + rr], outline=C(layer), width=W(layer, width))
            elif kind == "arc":
                _, c, r, a0, a1, layer, width = item
                # PIL arc angles: clockwise from 3 o'clock in screen coords.
                box = [P((c[0] - r, c[1] + r)), P((c[0] + r, c[1] - r))]
                d.arc([box[0][0], box[0][1], box[1][0], box[1][1]], start=-a1, end=-a0,
                      fill=C(layer), width=W(layer, width))
            elif kind == "text":
                _, p, s, hh, layer, align, rot, bold = item
                size = max(6, int(hh * ppm * 1.15))
                font = _font(size, bold)
                if rot:
                    tmp = Image.new("RGBA", (int(len(s) * size * 1.2) + 8, size + 8), (255, 255, 255, 0))
                    td = ImageDraw.Draw(tmp)
                    td.text((2, 2), s, font=font, fill=C(layer))
                    tmp = tmp.rotate(rot, expand=True)
                    x, y = P(p)
                    if align == "center":
                        img.paste(tmp, (int(x - tmp.width / 2), int(y - tmp.height / 2)), tmp)
                    elif align == "right":
                        img.paste(tmp, (int(x - tmp.width), int(y - tmp.height / 2)), tmp)
                    else:
                        img.paste(tmp, (int(x), int(y - tmp.height / 2)), tmp)
                else:
                    x, y = P(p)
                    anchor = {"left": "ls", "center": "ms", "right": "rs"}.get(align, "ls")
                    d.text((x, y), s, font=font, fill=C(layer), anchor=anchor)
            elif kind == "hatch":
                _, pts, angle, spacing, layer = item
                for a, b in _hatch_lines(pts, angle, spacing):
                    d.line([P(a), P(b)], fill=C(layer), width=W(layer))
            elif kind == "fill":
                _, pts, rgb, layer = item
                d.polygon([P(p) for p in pts], fill=rgb)

        Path(path).parent.mkdir(parents=True, exist_ok=True)
        img.save(path)
        return path

    def save_dxf(self, path, layers=None):
        if ezdxf is None:
            raise RuntimeError("ezdxf is not available")
        doc = ezdxf.new("R2010", setup=True)
        doc.header["$INSUNITS"] = 4  # millimetres
        doc.header["$MEASUREMENT"] = 1
        def _layer_of(item):
            kind = item[0]
            if kind in ("line", "circle"):
                return item[3]
            if kind == "arc":
                return item[5]
            if kind == "text":
                return item[4]
            if kind == "hatch":
                return item[4]
            if kind == "fill":
                return item[3]
            return "THIN"

        used = set(layers or [_layer_of(i) for i in self.items])
        for name in used:
            if name not in doc.layers:
                meta = LAYERS.get(name, LAYERS["THIN"])
                doc.layers.add(name, color=meta["color"])
        msp = doc.modelspace()

        for item in self.items:
            kind = item[0]
            if kind == "line":
                _, p1, p2, layer, width = item
                e = msp.add_line(p1, p2, dxfattribs={"layer": layer})
                if width:
                    e.dxf.lineweight = int(width * 100)
            elif kind == "circle":
                _, c, r, layer, width = item
                e = msp.add_circle(c, r, dxfattribs={"layer": layer})
                if width:
                    e.dxf.lineweight = int(width * 100)
            elif kind == "arc":
                _, c, r, a0, a1, layer, width = item
                e = msp.add_arc(c, r, a0, a1, dxfattribs={"layer": layer})
                if width:
                    e.dxf.lineweight = int(width * 100)
            elif kind == "text":
                _, p, s, hh, layer, align, rot, bold = item
                e = msp.add_text(
                    s,
                    height=hh,
                    rotation=rot,
                    dxfattribs={"layer": layer, "style": "Standard"},
                )
                e.set_placement(p, align={
                    "left": ezdxf.enums.TextEntityAlignment.LEFT,
                    "center": ezdxf.enums.TextEntityAlignment.CENTER,
                    "right": ezdxf.enums.TextEntityAlignment.RIGHT,
                }.get(align, ezdxf.enums.TextEntityAlignment.LEFT))
            elif kind == "hatch":
                _, pts, angle, spacing, layer = item
                hatch = msp.add_hatch(color=8, dxfattribs={"layer": layer})
                hatch.set_pattern_fill("ANSI31", scale=max(0.2, spacing / 3.0), angle=angle)
                hatch.paths.add_polyline_path(pts, is_closed=True)
            elif kind == "fill":
                _, pts, rgb, layer = item
                solid = msp.add_solid([pts[0], pts[1], pts[2] if len(pts) > 2 else pts[1]])
                solid.dxf.layer = layer
                solid.dxf.color = 8

        Path(path).parent.mkdir(parents=True, exist_ok=True)
        doc.saveas(path)
        return path

