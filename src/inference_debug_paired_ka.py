from email import parser
import os
import argparse
from pathlib import Path
from PIL import Image, ImageOps, ImageDraw
import torch
from torchvision import transforms
import torchvision.transforms.functional as F
from tqdm import tqdm
import numpy as np
from pix2pix_turbo import Pix2Pix_Turbo
from warp_utils.warp_pipeline import (
    apply_forward_warp,
    apply_unwarp,
    resize_longest_side,
    crop_to_foreground,
    detect_face_bbox,
    get_face_app,
)

# NOTE: try to run our warp on a single image!

# =====================
# CONFIG
# =====================
INPUT_PATH = Path("/home/shenzhen/Datasets/VITON/test/image/00055_00.jpg")
OUT_DIR = Path("debug")
TARGET_SIZE = 784
BW = 128
SEPARABLE = True
USE_FP16 = False

OUT_DIR.mkdir(exist_ok=True)

# NOTE: use face app here
face_app = get_face_app()


# ---------------------
# NEW: keep-aspect resize
# ---------------------
def resize_keep_aspect(img, target_size):
    w, h = img.size
    scale = target_size / max(w, h)

    new_w = round(w * scale)
    new_h = round(h * scale)

    new_w = (new_w // 8) * 8
    new_h = (new_h // 8) * 8

    return img.resize((new_w, new_h), Image.LANCZOS)


# ---------------------
# NEW: draw bbox
# ---------------------
def draw_bbox(img, bbox):
    img_draw = img.copy()
    draw = ImageDraw.Draw(img_draw)

    bbox_np = bbox.cpu().numpy().reshape(-1)
    x1, y1, x2, y2 = bbox_np[:4]

    draw.rectangle([x1, y1, x2, y2], outline="red", width=4)

    return img_draw


def run_pipeline(img, suffix, cropped_size):

    c_t = F.to_tensor(img).unsqueeze(0).cuda()

    if USE_FP16:
        c_t = c_t.half()

    with torch.no_grad():
        if BW > 0:

            # detected face bbox
            bbox = detect_face_bbox(img, face_app, include_eyes=True)

            if bbox is None:
                print(f"⚠️ No face detected, skipping.")
                return

            # draw bbox
            bbox_img = draw_bbox(img, bbox)
            bbox_img.save(OUT_DIR / f"{INPUT_PATH.stem}_bbox{suffix}.png")

            # (1) forward warp
            warped, warp_grid, saliency = apply_forward_warp(
                c_t, bbox.to(c_t.device), BW, SEPARABLE
            )

            # save warped image
            warped_pil = transforms.ToPILImage()(warped[0].cpu().clamp(0, 1))
            warped_pil = resize_longest_side(warped_pil, cropped_size, TARGET_SIZE)
            warped_pil.save(OUT_DIR / f"{INPUT_PATH.stem}_warp{suffix}.png")

            # (2) unwarp back directly
            unwarped = apply_unwarp(warp_grid, warped, SEPARABLE)

            # save unwarped image
            unwarped_pil = transforms.ToPILImage()(unwarped[0].cpu().clamp(0, 1))
            unwarped_pil = resize_longest_side(unwarped_pil, cropped_size, TARGET_SIZE)
            unwarped_pil.save(OUT_DIR / f"{INPUT_PATH.stem}_unwarp{suffix}.png")

            print(f"✅ Saved warp/unwarp {suffix}")


def process_image(input_path):

    img, cropped_size = crop_to_foreground(input_path)

    # save cropped foreground image first
    img.save(OUT_DIR / f"{input_path.stem}_cropped_fg.png")

    # -------------------
    # normal resize
    # -------------------
    img_normal = img.resize((TARGET_SIZE, TARGET_SIZE), Image.LANCZOS)

    run_pipeline(img_normal, "", cropped_size)

    # -------------------
    # keep aspect resize
    # -------------------
    img_ka = resize_keep_aspect(img, TARGET_SIZE)

    run_pipeline(img_ka, "_ka", cropped_size)


if __name__ == "__main__":
    process_image(INPUT_PATH)