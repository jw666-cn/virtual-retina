"""Rebuild the two-stage cylindrical gear reducer assembly drawing (A1, 1:2).

Geometry is authored in model millimetres and placed on the sheet at 1:2.
Values come from artifacts/cad_rec/recognition.json (crop-verified); anything
that could not be read on the source raster is marked "—" rather than invented.
"""
from __future__ import annotations

import math
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent))
from cadgen import Draft  # noqa: E402

OUT = Path("artifacts/cad_redraw")
S = 0.5  # 1:2


def build() -> Draft:
    d = Draft(841, 594)

    # ---------------------------------------------------------------- sheet frame
    d.rect(10, 10, 831, 584, "FRAME", 0.7)
    d.rect(25, 10, 831, 584, "THIN", 0.3)

    # ------------------------------------------------------------------ title block
    tx0, ty0, tx1, ty1 = 641, 10, 831, 66
    d.rect(tx0, ty0, tx1, ty1, "FRAME", 0.6)
    d.line((tx0, 46), (tx1, 46), "THIN", 0.3)
    d.line((tx0, 32), (tx1, 32), "THIN", 0.3)
    d.line((tx0, 20), (tx1, 20), "THIN", 0.3)
    d.line((tx0 + 118, 32), (tx0 + 118, ty1), "THIN", 0.3)
    d.line((tx0 + 118, 46), (tx1, 46), "THIN", 0.3)
    d.text((tx0 + 6, 52), "两级圆柱齿轮减速器", 8, "TEXT", bold=True)
    d.text((tx0 + 6, 38), "装配图（依据低分辨率原图复绘）", 3.0, "TEXT")
    d.text((tx0 + 6, 25), "设计", 3.0, "TEXT")
    d.text((tx0 + 40, 25), "—", 3.0, "TEXT")
    d.text((tx0 + 6, 14), "制图", 3.0, "TEXT")
    d.text((tx0 + 40, 14), "虚拟视网膜", 3.0, "TEXT")
    d.text((tx0 + 122, 38), "比例  1:2", 3.2, "TEXT")
    d.text((tx0 + 122, 25), "图幅  A1", 3.2, "TEXT")
    d.text((tx0 + 122, 14), "图号  —", 3.2, "TEXT")

    # ================================================================ FRONT VIEW
    ox, oy = 70.0, 330.0

    def f(x, y):
        return (ox + x * S, oy + y * S)

    def fl(x1, y1, x2, y2, layer="OUTLINE", w=None):
        d.line(f(x1, y1), f(x2, y2), layer, w)

    def fc(x, y, r, layer="OUTLINE", w=None):
        d.circle(f(x, y), r * S, layer, w)

    def ft(x, y, s, h=2.6, layer="TEXT", align="left", rot=0):
        p = f(x, y)
        d.text(p, s, h, layer, align, rot)

    # base and feet
    d.rect(*f(0, -15), *f(80, 0), "OUTLINE")
    d.rect(*f(392, -15), *f(472, 0), "OUTLINE")
    d.rect(*f(0, 0), *f(472, 18), "OUTLINE")
    # lower housing
    d.rect(*f(15, 18), *f(457, 185), "OUTLINE")
    # split flange
    d.rect(*f(0, 185), *f(494.5, 199.42), "OUTLINE")
    # upper housing (arch)
    arch = [(15, 199.42), (15, 232), (38, 256), (95, 272), (190, 280),
            (290, 279), (370, 268), (430, 246), (457, 222), (457, 199.42)]
    d.poly([f(x, y) for x, y in arch], False, "OUTLINE")
    # split line
    d.center_h(f(0, 199.42)[0], f(494.5, 199.42)[0], f(0, 199.42)[1])
    # shafts
    d.rect(*f(-80, 179.42), *f(95, 219.42), "OUTLINE")     # input Ø40
    d.rect(*f(322, 171.92), *f(532, 226.92), "OUTLINE")    # output Ø55
    # keyways
    d.rect(*f(-55, 193), *f(-30, 205.8), "THIN")
    d.rect(*f(470, 186), *f(505, 212.8), "THIN")
    # bearing covers: (x, outer r, bolt circle r, inner r, bolts)
    for cx, ro, rb, ri, nb in [(95, 40, 49, 18, 8), (199, 44, 53, 20, 8), (322, 52, 62, 26, 8)]:
        fc(cx, 199.42, ro, "OUTLINE", 0.5)
        fc(cx, 199.42, rb, "THIN", 0.25)
        fc(cx, 199.42, ri, "OUTLINE", 0.5)
        for k in range(nb):
            a = 2 * math.pi * k / nb
            fc(cx + rb * math.cos(a), 199.42 + rb * math.sin(a), 3.2, "THIN", 0.3)
    # split bolts
    for bx in (30, 62, 132, 165, 235, 268, 360, 394, 440, 470):
        fc(bx, 192, 5, "THIN", 0.3)
    # lifting eyes
    for ex, ey in [(60, 268), (400, 262)]:
        d.rect(*f(ex - 12, ey), *f(ex + 12, ey + 16), "OUTLINE")
        fc(ex, ey + 12, 5, "THIN")
    # inspection cover
    d.poly([f(175, 283), f(300, 291), f(305, 301), f(180, 293)], True, "OUTLINE")
    for bx in (195, 230, 265, 292):
        fc(bx, 289, 2.4, "THIN")
    # ventilator
    d.rect(*f(45, 283), *f(70, 296), "OUTLINE")
    fl(57.5, 296, 57.5, 304, "THIN")
    # oil sight glass
    fc(452, 120, 9, "OUTLINE")
    fc(452, 120, 5, "THIN")
    fl(430, 120, 443, 120, "THIN")
    # drain plug
    d.rect(*f(28, 2), *f(44, 14), "OUTLINE")
    fl(36, 14, 36, 18, "THIN")

    # dimensions
    def fd_h(x1, x2, y_model, label, y_text_offset=1.2):
        yp = oy + y_model * S
        d.dim_h(ox + x1 * S, ox + x2 * S, yp, oy, label, h=2.8)

    fd_h(95, 199, -22, "104")
    fd_h(199, 322, -22, "123")
    fd_h(322, 494.5, -22, "172.5")
    fd_h(0, 472, -45, "472")
    fd_h(-80, 472, -62, "552")
    fd_h(-80, 532, -79, "612")
    # drain hole
    d.leader([f(36, 8), f(10, -5), f(-25, -5)], "φ16", 2.6)

    d.text((ox + 150 * S, oy - 105 * S), "主视图", 5, "TEXT", "center")

    # ================================================================== TOP VIEW
    tvx, tvy = 70.0, 120.0

    def t(x, y):
        return (tvx + x * S, tvy + y * S)

    def tl(x1, y1, x2, y2, layer="OUTLINE", w=None):
        d.line(t(x1, y1), t(x2, y2), layer, w)

    def tc(x, y, r, layer="OUTLINE", w=None):
        d.circle(t(x, y), r * S, layer, w)

    # housing plan 550 x 295
    d.rect(*t(0, 0), *t(550, 295), "OUTLINE")
    d.rect(*t(12, 12), *t(538, 283), "THIN")
    # split flange bolts around perimeter
    for bx in range(30, 550, 52):
        tc(bx, 6, 3.5, "THIN"); tc(bx, 289, 3.5, "THIN")
    for by in range(40, 295, 52):
        tc(6, by, 3.5, "THIN"); tc(544, by, 3.5, "THIN")
    # shafts (vertical in plan) at x 110/214/337
    for sx in (110, 214, 337):
        tl(sx, 8, sx, 287, "CENTER")
        tc(sx, 26, 20, "OUTLINE"); tc(sx, 269, 20, "OUTLINE")
        d.rect(*t(sx - 24, 8), *t(sx + 24, 47), "OUTLINE")
        d.rect(*t(sx - 24, 248), *t(sx + 24, 287), "OUTLINE")
        d.rect(*t(sx - 15, 13), *t(sx + 15, 42), "THIN")
        d.rect(*t(sx - 15, 253), *t(sx + 15, 282), "THIN")
    # gears: (x, y0, y1, half width, label)
    gears = [
        (110, 92, 150, 34, ""),
        (214, 78, 162, 52, ""),
        (214, 176, 226, 30, ""),
        (337, 96, 194, 64, ""),
    ]
    for gx, gy0, gy1, hw, _ in gears:
        pts = [t(gx - hw, gy0), t(gx + hw, gy0), t(gx + hw, gy1), t(gx - hw, gy1)]
        d.poly(pts, True, "OUTLINE")
        d.hatch(pts, 45, 2.2)
        tl(gx - hw - 6, gy0, gx - hw - 6, gy1, "THIN")
        tl(gx + hw + 6, gy0, gx + hw + 6, gy1, "THIN")
    # fits
    d.leader([t(110, 121), t(95, 70), t(40, 70)], "Ø45H7/r6", 2.5)
    d.leader([t(214, 120), t(180, 62), t(120, 62)], "Ø44H7/r6", 2.5)
    d.leader([t(214, 201), t(250, 240), t(300, 240)], "Ø49H7/r6", 2.5)
    d.leader([t(337, 145), t(370, 60), t(415, 60)], "Ø64H7/r6", 2.5)
    d.leader([t(110, 36), t(60, 30), t(30, 30)], "Ø40k6 / Ø68H7/js6", 2.4)
    d.leader([t(214, 36), t(214, 20), t(180, 20)], "Ø55k6 / Ø75H7/js6", 2.4)
    d.leader([t(337, 268), t(380, 285), t(420, 285)], "Ø60k6 / Ø95H7/js6", 2.4)
    # width dimension
    d.dim_v(tvy + 0, tvy + 295 * S, tvx + 560 * S, tvx + 550 * S, "550", h=2.8)
    d.text((tvx + 275 * S, tvy - 16), "俯视图（沿箱体结合面剖视）", 5, "TEXT", "center")

    # ================================================================ RIGHT VIEW
    rx, ry = 450.0, 330.0

    def r(x, y):
        return (rx + x * S, ry + y * S)

    d.rect(*r(-147.5, 0), *r(147.5, 18), "OUTLINE")
    d.rect(*r(-110, 18), *r(110, 185), "OUTLINE")
    d.rect(*r(-147.5, 185), *r(147.5, 199.42), "OUTLINE")
    d.poly([r(-95, 199.42), r(-95, 232), r(-70, 258), r(-20, 275),
            r(20, 275), r(70, 258), r(95, 232), r(95, 199.42)], False, "OUTLINE")
    # feet
    d.rect(*r(-140, -12), *r(-90, 0), "OUTLINE")
    d.rect(*r(90, -12), *r(140, 0), "OUTLINE")
    # bearing cover + shaft end
    d.circle(r(0, 199.42), 55 * S, "OUTLINE")
    d.circle(r(0, 199.42), 33 * S, "OUTLINE")
    d.circle(r(0, 199.42), 27.5 * S, "THIN")
    d.center_h(r(-80, 199.42)[0], r(80, 199.42)[0], r(0, 199.42)[1])
    d.center_v(r(0, 120)[0], r(0, 120)[1], r(0, 290)[1])
    # base holes
    for hx in (-121.5, 121.5):
        d.circle(r(hx, 9), 6 * S, "THIN")
        d.center_v(r(hx, 0)[0], r(hx, 0)[1], r(hx, 20)[1])
    # dimensions
    d.dim_h(rx - 121.5 * S, rx + 121.5 * S, ry - 22, ry, "243", h=2.8)
    d.dim_h(rx - 147.5 * S, rx + 147.5 * S, ry - 40, ry, "295", h=2.8)
    d.dim_v(ry + 0, ry + 199.42 * S, rx - 170, rx - 147.5 * S, "199.42", h=2.8)
    d.text((rx + 0, ry - 58), "右视图", 5, "TEXT", "center")

    # ============================================================ TECH SPEC TABLE
    d.text((622, 566), "技术特性", 4.5, "TEXT", bold=True)
    d.table(620, 558, [48, 48, 40, 40], 8,
            [["3.5", "364", "0.85", "20"]],
            ["输入功率 kW", "高速轴转速 r/min", "效率", "传动比"])

    # =========================================================== TECH REQUIREMENTS
    d.text((622, 511), "技术要求", 4.5, "TEXT", bold=True)
    reqs = [
        "1. 箱体铸件应人工时效处理，保证尺寸精度稳定。",
        "2. 剖分面允许涂密封胶，接触面贴合，并用定位销定位。",
        "3. 箱体内表面涂耐油油漆。",
        "4. 装配前零件清洗去毛刺，齿轮与轴承配合面涂油。",
        "5. 运转平稳，无冲击、异常振动和噪声；结合面及密封处不得渗漏。",
        "6. 未注圆角半径按图样要求（原图字迹不可辨）。",
    ]
    yy = 501
    for line in reqs:
        d.text((622, yy), line, 2.6, "TEXT")
        yy -= 5.6

    # ======================================================================== BOM
    d.text((622, 466), "明细表（原图分辨率不足，名称不可辨者以“—”表示）", 3.0, "TEXT", bold=True)
    rows = [[str(i), "—", "—", "—", "—", "—"] for i in range(1, 48)]
    d.table(620, 459, [10, 52, 12, 32, 52, 34], 3.95, rows,
            ["序号", "名称", "数量", "材料", "标准号", "备注"], h=1.9)

    # main-parts note (verified part categories, no invented item numbers)
    d.text((620, 246), "主要部件（结构可确认，未与明细表序号强行对应）：", 2.8, "TEXT", bold=True)
    d.text((620, 238), "箱盖、箱座、高速轴、中间轴、低速轴、两级大小齿轮、滚动轴承(6套)、", 2.7, "TEXT")
    d.text((620, 230), "轴承端盖(6组)、键、挡油环、套筒、密封圈、定位销、螺栓、螺母、垫圈、", 2.7, "TEXT")
    d.text((620, 222), "通气器、油标、放油螺塞、窥视孔盖、起吊环。", 2.7, "TEXT")

    # north arrow / note
    d.text((30, 40), "注：本图为依据 1280×905 原图放大复核后的复绘；", 2.8, "TEXT")
    d.text((30, 32), "不可辨的明细表逐项名称、个别公差与技术要求数值均未编造。", 2.8, "TEXT")

    return d


if __name__ == "__main__":
    d = build()
    OUT.mkdir(parents=True, exist_ok=True)
    png = d.render(OUT / "减速器装配图_复绘.png", ppm=4.2)
    dxf = d.save_dxf(OUT / "减速器装配图_复绘.dxf")
    print("PNG", png, "DXF", dxf)
