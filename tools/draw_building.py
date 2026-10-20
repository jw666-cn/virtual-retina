"""Rebuild the teaching-building ground-floor plan (A2, 1:200).

The grid uses the sum of the printed bay dimensions (84000 x 36300 overall,
including the 9000 entrance platform).  The printed overall figures 85300 /
35400 disagree with their own bay chains; that discrepancy is recorded in the
sheet notes instead of being silently "corrected".
"""
from __future__ import annotations

import math
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent))
from cadgen import Draft  # noqa: E402

OUT = Path("artifacts/cad_redraw")
S = 1 / 200.0  # 1:200

# Model grid (mm)
AX = [0, 1800, 6900, 11850, 16800, 21750, 26700, 33900,
      41100, 42900, 50100, 57300, 62250, 67200, 72150, 77100, 82200, 84000]
AY = {"A": 9000, "B": 12800, "C": 15800, "D": 20000,
      "E": 23300, "F": 27100, "G": 30900, "H": 36300}

T = 240  # wall thickness


def build() -> Draft:
    d = Draft(594, 420)  # A2 landscape
    d.rect(10, 10, 584, 410, "FRAME", 0.7)
    d.rect(25, 10, 584, 410, "THIN", 0.3)

    OX, OY = 80.0, 118.0

    def P(x, y):
        return (OX + x * S, OY + y * S)

    def ln(x1, y1, x2, y2, layer="OUTLINE", w=None):
        d.line(P(x1, y1), P(x2, y2), layer, w)

    def txt(x, y, s, h=2.8, layer="TEXT", align="center", rot=0, bold=False):
        d.text(P(x, y), s, h, layer, align, rot, bold)

    def rectm(x0, y0, x1, y1, layer="OUTLINE", w=None):
        d.rect(*P(x0, y0), *P(x1, y1), layer, w)

    # --------------------------------------------------------------- wall helper
    def wall_h(x1, x2, y, openings=(), windows=(), t=T):
        """Horizontal wall; openings/windows are (a, b) in model x."""
        cuts = sorted([(a, b, "door") for a, b in openings] + [(a, b, "win") for a, b in windows])
        cur = x1
        for a, b, kind in cuts:
            a, b = max(a, x1), min(b, x2)
            if a > cur:
                ln(cur, y - t / 2, a, y - t / 2)
                ln(cur, y + t / 2, a, y + t / 2)
            # jambs
            ln(a, y - t / 2, a, y + t / 2, "THIN", 0.15)
            ln(b, y - t / 2, b, y + t / 2, "THIN", 0.15)
            if kind == "win":
                ln(a, y, b, y, "THIN", 0.15)
                ln(a, y - t / 2, b, y - t / 2, "OUTLINE", 0.3)
                ln(a, y + t / 2, b, y + t / 2, "OUTLINE", 0.3)
            cur = b
        if cur < x2:
            ln(cur, y - t / 2, x2, y - t / 2)
            ln(cur, y + t / 2, x2, y + t / 2)

    def wall_v(y1, y2, x, openings=(), windows=(), t=T):
        cuts = sorted([(a, b, "door") for a, b in openings] + [(a, b, "win") for a, b in windows])
        cur = y1
        for a, b, kind in cuts:
            a, b = max(a, y1), min(b, y2)
            if a > cur:
                ln(x - t / 2, cur, x - t / 2, a)
                ln(x + t / 2, cur, x + t / 2, a)
            ln(x - t / 2, a, x + t / 2, a, "THIN", 0.15)
            ln(x - t / 2, b, x + t / 2, b, "THIN", 0.15)
            if kind == "win":
                ln(x, a, x, b, "THIN", 0.15)
                ln(x - t / 2, a, x - t / 2, b, "OUTLINE", 0.3)
                ln(x + t / 2, a, x + t / 2, b, "OUTLINE", 0.3)
            cur = b
        if cur < y2:
            ln(x - t / 2, cur, x - t / 2, y2)
            ln(x + t / 2, cur, x + t / 2, y2)

    def door(x, y, w=900, direction="up", horizontal=True):
        """Door leaf + quarter arc at a wall opening."""
        if horizontal:
            sgn = 1 if direction == "up" else -1
            ln(x, y, x, y + sgn * w, "THIN", 0.2)
            d.arc(P(x, y), w * S, 0 if sgn > 0 else 270, 90 if sgn > 0 else 360, "THIN", 0.2)
        else:
            sgn = 1 if direction == "right" else -1
            ln(x, y, x + sgn * w, y, "THIN", 0.2)
            d.arc(P(x, y), w * S, 0 if sgn > 0 else 90, 90 if sgn > 0 else 180, "THIN", 0.2)

    # ================================================================== envelope
    # south (stepped): wings at B, centre at A
    wall_h(0, 33900, AY["B"], windows=[(9900, 12900), (18800, 21800), (8700, 11100), (17700, 20100)])
    wall_h(50100, 84000, AY["B"], windows=[(59100, 62100), (68100, 71100), (60300, 62700), (69300, 71700)])
    wall_h(33900, 50100, AY["A"], openings=[(41100, 42900)])
    # north
    wall_h(0, 33900, AY["G"], windows=[(2400, 5400), (9300, 12300), (18300, 21300), (30300, 32700)])
    wall_h(50100, 84000, AY["G"], windows=[(53100, 55500), (61500, 64500), (70500, 73500), (78600, 81600)])
    wall_h(33900, 50100, AY["G"], windows=[(34500, 36900), (39900, 42300), (44700, 47100)])
    # ends
    wall_v(AY["A"], AY["G"], 0, windows=[(14500, 17000)])
    wall_v(AY["A"], AY["G"], 84000, windows=[(14500, 17000)])

    # ---------------------------------------------------------------- veranda
    d.dashed(P(0, AY["A"]), P(33900, AY["A"]), "THIN", 3, 1.5, 0.15)
    d.dashed(P(50100, AY["A"]), P(84000, AY["A"]), "THIN", 3, 1.5, 0.15)
    for cx in AX[1:-1]:
        if cx < 33900 or cx > 50100:
            rectm(cx - 200, AY["A"] - 200, cx + 200, AY["A"] + 200, "THIN", 0.2)

    # ------------------------------------------------------------- main interior
    # corridor walls
    wall_h(0, 33900, AY["D"], openings=[(9900, 11400), (19400, 20900), (29800, 30700), (600, 1500)])
    wall_h(50100, 84000, AY["D"], openings=[(60900, 62400), (69400, 70900), (53600, 54500), (82500, 83400)])
    wall_h(0, 33900, AY["E"], openings=[(9900, 11400), (19400, 20900), (30400, 31300), (600, 1500)])
    wall_h(50100, 84000, AY["E"], openings=[(60900, 62400), (69400, 70900), (55000, 55900), (82500, 83400)])
    wall_h(33900, 50100, AY["B"], openings=[(34500, 35700), (44700, 45900)])

    # vertical partitions
    wall_v(AY["A"], AY["B"], 6900)
    wall_v(AY["B"], AY["D"], 6900)
    wall_v(AY["D"], AY["E"], 6900, openings=[(20500, 21400)])
    wall_v(AY["E"], AY["G"], 6900)
    wall_v(AY["A"], AY["G"], 26700, openings=[(15000, 15900), (26000, 26900)])
    wall_v(AY["A"], AY["G"], 33900, openings=[(16500, 17700), (28500, 29400)])
    wall_v(AY["A"], AY["G"], 50100, openings=[(16500, 17700), (28500, 29400)])
    wall_v(AY["A"], AY["G"], 57300, openings=[(15000, 15900), (26000, 26900)])
    wall_v(AY["A"], AY["G"], 77100)
    # centre north rooms
    wall_v(AY["E"], AY["G"], 39000, openings=[(29500, 30400)])
    wall_v(AY["E"], AY["G"], 44100, openings=[(29500, 30400)])
    # centre south: exhibition halls vs entrance
    wall_v(AY["A"], AY["B"], 41100, openings=[(10500, 11700)])
    wall_v(AY["A"], AY["B"], 42900, openings=[(10500, 11700)])

    # doors (leaf + arc)
    for x in (9900, 19400):
        door(x, AY["D"], 1500, "up", True)
        door(x, AY["E"], 1500, "up", True)
    for x in (60900, 69400):
        door(x, AY["D"], 1500, "up", True)
        door(x, AY["E"], 1500, "up", True)
    door(30400, AY["E"], 900, "up", True)
    door(55000, AY["E"], 900, "up", True)
    door(600, AY["D"], 900, "up", True)
    door(600, AY["E"], 900, "up", True)
    door(82500, AY["D"], 900, "up", True)
    door(82500, AY["E"], 900, "up", True)
    door(34500, AY["B"], 1200, "up", True)
    door(44700, AY["B"], 1200, "up", True)
    # entrance double door
    for i in range(2):
        door(41100 + i * 900, AY["A"], 900, "up", True)

    # ------------------------------------------------------------- rooms / fills
    rooms = [
        ("教室", 6900, AY["B"], 16800, AY["D"]), ("教室", 16800, AY["B"], 26700, AY["D"]),
        ("教室", 6900, AY["E"], 16800, AY["G"]), ("教室", 16800, AY["E"], 26700, AY["G"]),
        ("教室", 57300, AY["B"], 67200, AY["D"]), ("教室", 67200, AY["B"], 77100, AY["D"]),
        ("教室", 57300, AY["E"], 67200, AY["G"]), ("教室", 67200, AY["E"], 77100, AY["G"]),
        ("教师办公", 26700, AY["E"], 33900, AY["G"]),
        ("教师办公", 50100, AY["E"], 57300, AY["G"]),
        ("值班室", 26700, AY["A"], 33900, AY["D"]),
        ("配电室", 50100, AY["A"], 57300, AY["D"]),
        ("接待室", 33900, AY["E"], 39000, AY["G"]),
        ("保管室", 39000, AY["E"], 44100, AY["G"]),
        ("总务室", 44100, AY["E"], 50100, AY["G"]),
        ("门厅", 33900, AY["B"], 50100, AY["D"]),
        ("过厅", 33900, AY["D"], 50100, AY["E"]),
        ("展示厅", 33900, AY["A"], 41100, AY["B"]),
        ("展示厅", 42900, AY["A"], 50100, AY["B"]),
        ("卫生间", 0, AY["E"], 3000, AY["G"]),
        ("卫生间", 81000, AY["E"], 84000, AY["G"]),
        ("楼梯间", 0, AY["B"], 3000, AY["D"]),
        ("楼梯间", 81000, AY["B"], 84000, AY["D"]),
    ]
    for name, x0, y0, x1, y1 in rooms:
        if name in ("教室", "教师办公", "接待室", "保管室", "总务室", "门厅", "过厅", "展示厅", "卫生间", "楼梯间"):
            ty = y1 - 1100
        else:
            ty = (y0 + y1) / 2
        txt((x0 + x1) / 2, ty, name, 3.2)

    # ------------------------------------------------------------- desks etc.
    def classroom_furniture(x0, y0, x1, y1):
        cols, rows = 3, 3
        w, h = 600, 400
        cw = (x1 - x0 - 1600) / cols
        ch = (y1 - y0 - 1600) / rows
        for r in range(rows):
            for c in range(cols):
                cx = x0 + 800 + cw * (c + 0.5) - w / 2
                cy = y0 + 800 + ch * (r + 0.5) - h / 2
                rectm(cx, cy, cx + w, cy + h, "THIN", 0.15)
                rectm(cx + 60, cy - 320, cx + w - 60, cy - 60, "THIN", 0.12)

    for name, x0, y0, x1, y1 in rooms:
        if name == "教室":
            classroom_furniture(x0, y0, x1, y1)
        elif name in ("教师办公",):
            cx, cy = (x0 + x1) / 2, (y0 + y1) / 2
            rectm(cx - 1600, cy - 350, cx + 1600, cy + 350, "THIN", 0.2)
            for k in (-900, 0, 900):
                rectm(cx + k - 200, cy - 800, cx + k + 200, cy - 450, "THIN", 0.12)
        elif name == "门厅":
            cx = (x0 + x1) / 2
            rectm(cx - 1800, y0 + 2500, cx + 1800, y0 + 3800, "THIN", 0.2)
        elif name == "卫生间":
            n = 3
            for k in range(n):
                yy = y0 + 800 + k * 1100
                rectm(x0 + 300, yy, x0 + 1300, yy + 900, "THIN", 0.12)
                d.circle(P(x0 + 800, yy + 450), 180 * S, "THIN", 0.12)
            rectm(x1 - 900, y0 + 800, x1 - 300, y0 + 1400, "THIN", 0.12)

    # stairs: treads + direction arrow
    def stair(x0, y0, x1, y1, vertical=True, up=True):
        n = 10
        if vertical:
            for k in range(1, n):
                yy = y0 + (y1 - y0) * k / n
                ln(x0, yy, x1, yy, "THIN", 0.15)
            ln((x0 + x1) / 2, y0 + 400, (x0 + x1) / 2, y1 - 400, "THIN", 0.2)
            d._arrow(P((x0 + x1) / 2, y1 - 700 if up else y0 + 700), (0, 1 if up else -1), 1.6, "THIN")
            txt((x0 + x1) / 2 + 300, (y0 + y1) / 2, "上" if up else "下", 2.4)
        else:
            for k in range(1, n):
                xx = x0 + (x1 - x0) * k / n
                ln(xx, y0, xx, y1, "THIN", 0.15)

    stair(200, AY["B"] + 400, 2800, AY["D"] - 1600, True, True)
    stair(81200, AY["B"] + 400, 83800, AY["D"] - 1600, True, True)
    # north centre stairs (outside envelope)
    stair(33900, AY["G"] + 300, 37500, AY["H"] - 600, True, False)
    stair(46500, AY["G"] + 300, 50100, AY["H"] - 600, True, False)

    # 花池 / platform / steps
    rectm(37500, AY["G"] + 900, 46500, AY["H"] - 900, "OUTLINE", 0.25)
    txt(42000, AY["G"] + 2200, "花池", 3.0)
    rectm(33900, 300, 50100, AY["A"] - 300, "OUTLINE", 0.3)
    txt(42000, 6000, "入口平台", 3.2)
    # steps
    for k in range(4):
        y = 300 + k * 500
        ln(37300, y, 46700, y, "THIN", 0.15)
    txt(42000, 1600, "台阶", 2.6)

    # ------------------------------------------------------------- grid / dims
    for i, x in enumerate(AX):
        d.circle(P(x, AY["H"] + 2600), 2.2, "THIN", 0.25)
        txt(x, AY["H"] + 2600, str(i + 1), 2.2, "THIN", "center")
        d.center_v(P(x, -1200)[0], P(x, AY["H"] + 1500)[1], P(x, -1200)[1], "CENTER")
        d.circle(P(x, -2600), 2.2, "THIN", 0.25)
        txt(x, -2600, str(i + 1), 2.2, "THIN", "center")
    for k, y in enumerate(AY.values()):
        letter = "ABCDEFGH"[k]
        d.circle(P(-2600, y), 2.2, "THIN", 0.25)
        txt(-2600, y, letter, 2.2, "THIN", "center")
        d.circle(P(86600, y), 2.2, "THIN", 0.25)
        txt(86600, y, letter, 2.2, "THIN", "center")
        d.center_h(P(-1200, y)[0], P(85200, y)[0], P(-1200, y)[1], "CENTER")

    # horizontal chain
    ytop = AY["H"] + 5200
    for i in range(len(AX) - 1):
        x1, x2 = AX[i], AX[i + 1]
        d.dim_h(OX + x1 * S, OX + x2 * S, OY + ytop * S, OY + (AY["H"] + 3400) * S,
                f"{x2 - x1}", h=2.1)
    d.dim_h(OX, OX + 84000 * S, OY + (ytop + 2600) * S, OY + (AY["H"] + 3400) * S,
            "84000（图面总标注 85300）", h=2.2)
    # vertical chain
    ylist = [0] + list(AY.values())
    for i in range(len(ylist) - 1):
        y1, y2 = ylist[i], ylist[i + 1]
        d.dim_v(OY + y1 * S, OY + y2 * S, 32, 44,
                f"{y2 - y1}", h=2.1)
    d.dim_v(OY, OY + 36300 * S, 25.5, 44,
            "36300（图面总标注 35400）", h=2.2)

    # --------------------------------------------------------------- title block
    tx0, ty0, tx1, ty1 = 470, 10, 584, 48
    d.rect(tx0, ty0, tx1, ty1, "FRAME", 0.6)
    d.line((tx0, 34), (tx1, 34), "THIN", 0.3)
    d.line((tx0, 24), (tx1, 24), "THIN", 0.3)
    d.line((tx0 + 78, 24), (tx0 + 78, ty1), "THIN", 0.3)
    d.text((tx0 + 5, 40), "教学楼一层平面图", 6.5, "TEXT", bold=True)
    d.text((tx0 + 5, 29), "比例 1:200", 3.0)
    d.text((tx0 + 5, 18), "依据 1440×1020 原图复绘", 2.5)
    d.text((tx0 + 82, 29), "图幅 A2", 3.0)
    d.text((tx0 + 82, 18), "2026-10-19", 2.6)

    # ------------------------------------------------------------- north arrow
    nx, ny = 540.0, 360.0
    d.line((nx, ny - 12), (nx, ny + 8), "OUTLINE", 0.5)
    d.line((nx - 3.0, ny - 4), (nx, ny + 12), "OUTLINE", 0.4)
    d.line((nx + 3.0, ny - 4), (nx, ny + 12), "OUTLINE", 0.4)
    d.text((nx, ny + 16), "N", 3.5, "TEXT", bold=True)

    # -------------------------------------------------------------------- notes
    notes = [
        "注 1：轴网横向按分尺寸求和 84000 建模；原图总标注 85300，相差 1300，疑似低分辨率数字误读。",
        "注 2：纵向 A–H 为 27300，加入口平台 9000 共 36300；原图总标注 35400，相差 900，待高清图核对。",
        "注 3：卫生间原图无文字标注，按洁具符号识别；门窗为按开间示意布置。",
    ]
    yy = 44
    for n in notes:
        d.text((40, yy), n, 2.4, "TEXT")
        yy -= 4.0

    return d


if __name__ == "__main__":
    d = build()
    OUT.mkdir(parents=True, exist_ok=True)
    png = d.render(OUT / "教学楼一层平面图_复绘.png", ppm=5.0)
    dxf = d.save_dxf(OUT / "教学楼一层平面图_复绘.dxf")
    print("PNG", png, "DXF", dxf)
