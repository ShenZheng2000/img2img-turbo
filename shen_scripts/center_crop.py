import os
from pathlib import Path
from PIL import Image

IN_DIR = Path("/home/shenzhen/Datasets/depth/workzone_segm/boston/image")
OUT_DIR = Path("/home/shenzhen/Datasets/depth/workzone_segm/boston/image_cc784")
TARGET = 784

IMG_EXTS = {".jpg", ".jpeg", ".png", ".webp"}

def center_crop_pil(img: Image.Image, target: int) -> Image.Image:
    w, h = img.size
    # If smaller than target, upscale first (keeps behavior consistent)
    if w < target or h < target:
        scale = max(target / w, target / h)
        new_w = int(round(w * scale))
        new_h = int(round(h * scale))
        img = img.resize((new_w, new_h), Image.BICUBIC)
        w, h = img.size

    left = (w - target) // 2
    top = (h - target) // 2
    return img.crop((left, top, left + target, top + target))

def main():
    OUT_DIR.mkdir(parents=True, exist_ok=True)

    paths = sorted([p for p in IN_DIR.iterdir() if p.suffix.lower() in IMG_EXTS])
    if not paths:
        raise RuntimeError(f"No images found in {IN_DIR}")

    for p in paths:
        img = Image.open(p).convert("RGB")
        cropped = center_crop_pil(img, TARGET)

        out_path = OUT_DIR / p.name
        # keep extension; for jpg use quality
        if out_path.suffix.lower() in [".jpg", ".jpeg"]:
            cropped.save(out_path, quality=95)
        else:
            cropped.save(out_path)

    print(f"Done. Saved {len(paths)} crops to: {OUT_DIR}")

if __name__ == "__main__":
    main()