from pathlib import Path
from PIL import Image, ImageEnhance
import sys

SRC = Path("artifacts/cad_src")
OUT = Path("artifacts/cad_rec/crops")
OUT.mkdir(parents=True, exist_ok=True)

jobs = {
    "gearbox": {
        "src": SRC / "gearbox.jpg",
        "crops": {
            "gb_top_dims": (455, 235, 780, 405, 3.0),
            "gb_side_view": (780, 40, 1120, 410, 3.0),
            "gb_bottom_left": (180, 500, 560, 880, 3.0),
            "gb_bottom_right": (500, 500, 780, 880, 3.0),
            "gb_tech_req": (1045, 405, 1280, 590, 3.2),
            "gb_tech_spec": (820, 505, 1055, 630, 3.2),
            "gb_bom": (820, 620, 1280, 850, 3.0),
            "gb_title": (1070, 800, 1280, 905, 3.5),
        },
    },
    "building": {
        "src": SRC / "building.jpg",
        "crops": {
            "bd_top_dims": (95, 130, 1360, 240, 2.2),
            "bd_left_dims": (45, 210, 160, 710, 3.0),
            "bd_center_rooms": (480, 270, 960, 560, 2.8),
            "bd_left_rooms": (160, 270, 520, 560, 2.8),
            "bd_right_rooms": (880, 270, 1260, 560, 2.8),
            "bd_title": (500, 740, 1060, 860, 2.6),
            "bd_bottom_dims": (95, 650, 1360, 770, 2.2),
        },
    },
}

for name, spec in jobs.items():
    im = Image.open(spec["src"]).convert("RGB")
    for key, (x0, y0, x1, y1, scale) in spec["crops"].items():
        box = (max(0, x0), max(0, y0), min(im.width, x1), min(im.height, y1))
        crop = im.crop(box)
        w, h = crop.size
        crop = crop.resize((int(w * scale), int(h * scale)), Image.LANCZOS)
        crop = ImageEnhance.Contrast(crop).enhance(1.25)
        crop = ImageEnhance.Sharpness(crop).enhance(1.5)
        path = OUT / f"{key}.png"
        crop.save(path)
        print(f"{path} {crop.width}x{crop.height}")
