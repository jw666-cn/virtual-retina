from PIL import Image, ImageEnhance
im = Image.open("artifacts/cad_src/building.jpg").convert("RGB")
crops = {
 "bd_center_labels": (600, 300, 780, 500, 6.0),
 "bd_center_top": (560, 210, 800, 320, 5.0),
}
for k,(x0,y0,x1,y1,s) in crops.items():
    c=im.crop((x0,y0,x1,y1)); w,h=c.size
    c=c.resize((int(w*s),int(h*s)),Image.LANCZOS); c=ImageEnhance.Contrast(c).enhance(1.4)
    p=f"artifacts/cad_rec/crops2/{k}.png"; c.save(p); print(p,c.size)
im2 = Image.open("artifacts/cad_src/gearbox.jpg").convert("RGB")
c = im2.crop((820, 545, 1010, 600)); c=c.resize((c.width*6,c.height*6),Image.LANCZOS)
c=ImageEnhance.Contrast(c).enhance(1.4); c.save("artifacts/cad_rec/crops2/gb_spec_row.png"); print(c.size)
