from pathlib import Path
from PIL import Image, ImageDraw, ImageFont

FONT = r"C:\Windows\Fonts\msyh.ttc"
def font(sz, bold=False):
    return ImageFont.truetype(FONT, sz, index=0)

def panel(path, title, sub, box_w, box_h):
    im = Image.open(path).convert("RGB")
    im.thumbnail((box_w, box_h), Image.LANCZOS)
    canvas = Image.new("RGB", (box_w, box_h + 92), "white")
    d = ImageDraw.Draw(canvas)
    d.text((10, 8), title, font=font(26, True), fill="#111111")
    d.text((10, 44), sub, font=font(17), fill="#555555")
    x = (box_w - im.width) // 2
    canvas.paste(im, (x, 92))
    d.rectangle((x - 1, 91, x + im.width, 91 + im.height), outline="#999999", width=1)
    return canvas

def make(orig, redraw, out, name, osub, rsub):
    W, H = 1000, 760
    left = panel(orig, "原图", osub, W, H)
    right = panel(redraw, "复绘", rsub, W, H)
    top = 70
    canvas = Image.new("RGB", (W * 2 + 30, H + 92 + top + 20), "white")
    d = ImageDraw.Draw(canvas)
    d.text((15, 14), name, font=font(30, True), fill="#111111")
    canvas.paste(left, (0, top))
    canvas.paste(right, (W + 30, top))
    d.line((W + 15, top, W + 15, top + H + 92), fill="#cccccc", width=2)
    canvas.save(out)
    print(out, canvas.size)

OUT = Path("artifacts/cad_redraw")
make("artifacts/cad_src/gearbox.jpg", OUT / "减速器装配图_复绘.png", OUT / "原图vs复绘_减速器.png",
     "两级圆柱齿轮减速器：原图 vs 复绘（A1，1:2）",
     "1280×905 px，装配图（主/俯/右视 + 技术要求 + 明细表）",
     "841×594 mm（A1），PNG 3532×2494，另附 DXF")
make("artifacts/cad_src/building.jpg", OUT / "教学楼一层平面图_复绘.png", OUT / "原图vs复绘_教学楼.png",
     "教学楼一层平面图：原图 vs 复绘（A2，1:200）",
     "1440×1020 px，1:200 平面图",
     "594×420 mm（A2），PNG 2970×2100，另附 DXF")
