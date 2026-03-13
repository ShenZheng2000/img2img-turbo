from email import parser
import os
import argparse
from pathlib import Path
from PIL import Image, ImageOps
import torch
from torchvision import transforms
import torchvision.transforms.functional as F
from tqdm import tqdm
import matplotlib.pyplot as plt
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

def save_difference_heatmap(orig_img, recon_img, save_path):
    orig = np.array(orig_img).astype(np.float32) / 255.0
    recon = np.array(recon_img).astype(np.float32) / 255.0

    diff = np.abs(orig - recon)
    diff_map = diff.mean(axis=2)   # collapse RGB

    plt.imshow(diff_map, cmap="jet", vmin=0.0, vmax=1.0)
    plt.colorbar()
    plt.axis("off")
    plt.savefig(save_path, bbox_inches="tight", pad_inches=0.05)
    plt.close()


def process_image(input_path):
    img, cropped_size = crop_to_foreground(input_path)
    img_fg = img.copy()
    img_fg_resized = resize_longest_side(img_fg, cropped_size, TARGET_SIZE)

    # save cropped foreground image first
    img.save(OUT_DIR / f"{input_path.stem}_cropped_fg.png")

    img = img.resize((TARGET_SIZE, TARGET_SIZE), Image.LANCZOS)

    c_t = F.to_tensor(img).unsqueeze(0).cuda()
    if USE_FP16:
        c_t = c_t.half()

    with torch.no_grad():
        if BW > 0:
            # detected face bbox
            bbox = detect_face_bbox(img, face_app, include_eyes=True)

            if bbox is None:
                print(f"⚠️ No face detected for {input_path.name}, skipping.")
                return

            # (1) forward warp
            warped, warp_grid, saliency = apply_forward_warp(
                c_t, bbox.to(c_t.device), BW, SEPARABLE
            )

            # save warped image
            warped_pil = transforms.ToPILImage()(warped[0].cpu().clamp(0, 1))
            warped_pil = resize_longest_side(warped_pil, cropped_size, TARGET_SIZE)
            warped_pil.save(OUT_DIR / f"{input_path.stem}_warp.png")

            # (2) unwarp back directly
            unwarped = apply_unwarp(warp_grid, warped, SEPARABLE)

            # save unwarped image
            unwarped_pil = transforms.ToPILImage()(unwarped[0].cpu().clamp(0, 1))
            unwarped_pil = resize_longest_side(unwarped_pil, cropped_size, TARGET_SIZE)
            unwarped_pil.save(OUT_DIR / f"{input_path.stem}_unwarp.png")

            # save difference heatmap between original cropped foreground and unwarped result
            save_difference_heatmap(
                img_fg_resized,            # original cropped fg (PIL)
                unwarped_pil,   # resized unwarped (PIL)
                OUT_DIR / f"{input_path.stem}_diff_heatmap.png"
            )

            print(f"✅ Saved cropped_fg / warp / unwarp to {OUT_DIR}")


if __name__ == "__main__":
    process_image(INPUT_PATH)