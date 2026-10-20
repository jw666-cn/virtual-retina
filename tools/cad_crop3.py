from PIL import Image, ImageEnhance
im = Image.open("artifacts/cad_src/building.jpg").convert("RGB")
crops = {
 "bd_vert_8100": (58, 560, 105, 720, 6.0),
 "bd_vert_35400": (60, 390, 105, 600, 6.0),
 "bd_horiz_84000": (640, 155, 760, 205, 6.0),
 "bd_room_small_left": (170, 280, 290, 450, 5.0),
 "bd_room_topleft_class": (280, 285, 480, 390, 4.0),
}
for k,(x0,y0,x1,y1,s) in crops.items():
    c = im.crop((x0,y0,x1,y1)); w,h=c.size
    c = c.resize((int(w*s),int(h*s)), Image.LANCZOS)
    c = ImageEnhance.Contrast(c).enhance(1.5)
    p=f"artifacts/cad_rec/crops2/{k}.png"; c.save(p); print(p,c.size)
