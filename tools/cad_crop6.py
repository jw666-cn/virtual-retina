from PIL import Image, ImageEnhance
im=Image.open("artifacts/cad_src/gearbox.jpg").convert("RGB")
for k,(x0,y0,x1,y1,s) in {
 "gb_view_topleft": (160, 30, 720, 400, 2.2),
 "gb_view_bottom": (160, 420, 780, 890, 2.0),
}.items():
    c=im.crop((x0,y0,x1,y1)); c=c.resize((int(c.width*s),int(c.height*s)),Image.LANCZOS)
    c=ImageEnhance.Contrast(c).enhance(1.15)
    p=f"artifacts/cad_rec/crops2/{k}.png"; c.save(p); print(p,c.size)
