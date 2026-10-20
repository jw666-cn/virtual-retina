from PIL import Image, ImageEnhance
im=Image.open("artifacts/cad_src/gearbox.jpg").convert("RGB")
for k,(x0,y0,x1,y1,s) in {"gb_h220":(670,55,750,300,6.0),"gb_b550":(690,600,760,830,6.0)}.items():
    c=im.crop((x0,y0,x1,y1)); c=c.resize((int(c.width*s),int(c.height*s)),Image.LANCZOS)
    c=ImageEnhance.Contrast(c).enhance(1.4); p=f"artifacts/cad_rec/crops2/{k}.png"; c.save(p); print(p,c.size)
