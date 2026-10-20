from PIL import Image, ImageEnhance, ImageFilter
im = Image.open("artifacts/cad_src/building.jpg").convert("L")
crops = {
 "bd_v_bottom": (100, 535, 145, 720, 8.0),
 "bd_v_overall": (60, 330, 100, 640, 8.0),
 "bd_h_overall": (655, 160, 745, 200, 8.0),
}
for k,(x0,y0,x1,y1,s) in crops.items():
    c = im.crop((x0,y0,x1,y1)).resize((int((x1-x0)*s),int((y1-y0)*s)), Image.LANCZOS)
    c = ImageEnhance.Contrast(c).enhance(2.0)
    c = c.filter(ImageFilter.SHARPEN)
    p=f"artifacts/cad_rec/crops2/{k}.png"; c.save(p); print(p,c.size)
