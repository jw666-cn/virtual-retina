from pathlib import Path
from PIL import Image, ImageEnhance
SRC = Path("artifacts/cad_src"); OUT = Path("artifacts/cad_rec/crops2"); OUT.mkdir(parents=True, exist_ok=True)
jobs = {
 "gearbox": (SRC/"gearbox.jpg", {
   "gb_top_dims2": (230, 250, 780, 390, 4.0),
   "gb_side2": (790, 140, 1100, 400, 3.6),
   "gb_bottom_left2": (170, 500, 520, 880, 3.4),
   "gb_bottom_right2": (500, 500, 790, 880, 3.4),
   "gb_tech_req2": (995, 400, 1280, 575, 4.0),
   "gb_tech_spec2": (805, 495, 1065, 635, 4.0),
   "gb_title2": (1025, 785, 1280, 905, 4.2),
 }),
 "building": (SRC/"building.jpg", {
   "bd_top_left": (95, 140, 760, 240, 3.2),
   "bd_top_right": (740, 140, 1370, 240, 3.2),
   "bd_left_dims2": (40, 200, 165, 720, 3.4),
   "bd_rooms_center2": (470, 265, 970, 570, 3.0),
   "bd_rooms_left2": (150, 265, 520, 570, 3.0),
   "bd_rooms_right2": (880, 265, 1270, 570, 3.0),
   "bd_title2": (470, 730, 1080, 870, 3.0),
   "bd_bottom_left": (95, 650, 760, 775, 3.2),
   "bd_bottom_right": (740, 650, 1370, 775, 3.2),
 }),
}
for name,(src,crops) in jobs.items():
    im = Image.open(src).convert("RGB")
    for key,(x0,y0,x1,y1,s) in crops.items():
        box=(max(0,x0),max(0,y0),min(im.width,x1),min(im.height,y1))
        c=im.crop(box); w,h=c.size
        c=c.resize((int(w*s),int(h*s)),Image.LANCZOS)
        c=ImageEnhance.Contrast(c).enhance(1.35)
        c=ImageEnhance.Sharpness(c).enhance(1.6)
        p=OUT/f"{key}.png"; c.save(p); print(p, c.size)
