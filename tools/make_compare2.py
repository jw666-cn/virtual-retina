from pathlib import Path
from PIL import Image, ImageDraw, ImageFont

FONT = r"C:\Windows\Fonts\msyh.ttc"
def font(sz, bold=False): return ImageFont.truetype(FONT, sz, index=0)

def fit(im, w, h):
    im = im.copy(); im.thumbnail((w, h), Image.LANCZOS); return im

def panel(img, title, sub, w, h, header=78):
    c = Image.new("RGB", (w, h + header), "white")
    d = ImageDraw.Draw(c)
    d.text((10, 6), title, font=font(22, True), fill="#111")
    d.text((10, 38), sub, font=font(15), fill="#555")
    x = (w - img.width)//2
    c.paste(img, (x, header))
    d.rectangle((x-1, header-1, x+img.width, header+img.height), outline="#999")
    return c

def compose(orig_path, redraw_path, out_path, name,
            orig_box, redraw_box_mm, scale_px_per_mm, sheet_h, sheet_w,
            osub, rsub, dsub_o, dsub_r):
    W, H = 1000, 700
    o = Image.open(orig_path).convert("RGB")
    r = Image.open(redraw_path).convert("RGB")
    top_l = panel(fit(o, W, H), "原图", osub, W, H)
    top_r = panel(fit(r, W, H), "复绘", rsub, W, H)

    od = o.crop(orig_box)
    x0, y0, x1, y1 = redraw_box_mm
    rd = r.crop((int(x0*scale_px_per_mm), int((sheet_h-y1)*scale_px_per_mm),
                 int(x1*scale_px_per_mm), int((sheet_h-y0)*scale_px_per_mm)))
    DH = 420
    bot_l = panel(fit(od, W, DH), "原图细节", dsub_o, W, DH)
    bot_r = panel(fit(rd, W, DH), "复绘细节", dsub_r, W, DH)

    canvas = Image.new("RGB", (W*2+30, 70 + (H+78) + 30 + (DH+78)), "white")
    d = ImageDraw.Draw(canvas)
    d.text((15, 12), name, font=font(28, True), fill="#111")
    canvas.paste(top_l, (0, 70)); canvas.paste(top_r, (W+30, 70))
    canvas.paste(bot_l, (0, 70 + H + 78 + 24)); canvas.paste(bot_r, (W+30, 70 + H + 78 + 24))
    d.line((W+15, 70, W+15, canvas.height-10), fill="#ccc", width=2)
    canvas.save(out_path)
    print(out_path, canvas.size)

OUT = Path("artifacts/cad_redraw")
compose("artifacts/cad_src/gearbox.jpg", OUT/"减速器装配图_复绘.png", OUT/"原图vs复绘_减速器.png",
        "两级圆柱齿轮减速器：原图 vs 复绘（A1，1:2）",
        (170, 30, 790, 400), (15, 120, 360, 412), 4.2, 594, 841,
        "1280×905 px", "841×594 mm（A1），3532×2494 px",
        "主视图区域", "主视图 + 俯视图区域")
compose("artifacts/cad_src/building.jpg", OUT/"教学楼一层平面图_复绘.png", OUT/"原图vs复绘_教学楼.png",
        "教学楼一层平面图：原图 vs 复绘（A2，1:200）",
        (430, 230, 1100, 790), (225, 145, 355, 315), 5.0, 420, 594,
        "1440×1020 px", "594×420 mm（A2），2970×2100 px",
        "中部门厅/过厅/展示厅", "中部门厅/过厅/展示厅")
