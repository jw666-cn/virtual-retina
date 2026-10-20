from PIL import Image, ImageEnhance, ImageFilter
im=Image.open("artifacts/cad_src/gearbox.jpg").convert("L")
regions={
 "gb_bom_top": (832, 618, 1080, 735, 7.0),
 "gb_bom_bot": (832, 735, 1080, 848, 7.0),
}
for k,(x0,y0,x1,y1,s) in regions.items():
    c=im.crop((x0,y0,x1,y1)).resize((int((x1-x0)*s),int((y1-y0)*s)),Image.LANCZOS)
    c=ImageEnhance.Contrast(c).enhance(1.8); c=c.filter(ImageFilter.SHARPEN)
    p=f"artifacts/cad_rec/crops2/{k}.png"; c.save(p); print(p,c.size)
