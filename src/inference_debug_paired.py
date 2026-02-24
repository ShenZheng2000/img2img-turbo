
from email import parser
import os
import argparse
from pathlib import Path
from PIL import Image, ImageOps
import torch
from torchvision import transforms
import torchvision.transforms.functional as F
from tqdm import tqdm

from pix2pix_turbo import Pix2Pix_Turbo
from warp_utils.warp_pipeline import apply_forward_warp, resize_longest_side, crop_to_foreground

# NOTE: try to simulate instance-warp, NOT our warp!

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


def process_image(input_path):
    img, cropped_size = crop_to_foreground(input_path)

    img = img.resize((TARGET_SIZE, TARGET_SIZE), Image.LANCZOS)

    c_t = F.to_tensor(img).unsqueeze(0).cuda()
    if USE_FP16:
        c_t = c_t.half()

    output_path = OUT_DIR / input_path.name

    with torch.no_grad():
        if BW > 0:
            # FULL IMAGE bbox
            bbox = torch.tensor(
                [[0, 0, TARGET_SIZE - 1, TARGET_SIZE - 1]],
                dtype=torch.float32,
                device=c_t.device,
            )

            warped, warp_grid, saliency = apply_forward_warp(
                c_t, bbox, BW, SEPARABLE
            )

            # save warped
            warped_pil = transforms.ToPILImage()(warped[0].cpu().clamp(0, 1))
            warped_pil = resize_longest_side(warped_pil, cropped_size, TARGET_SIZE)
            warped_pil.save(OUT_DIR / f"{input_path.stem}_instance_warp.png")

            # save saliency
            sal = saliency.float()[0, 0]
            sal = (sal - sal.min()) / (sal.max() - sal.min() + 1e-8)
            sal_pil = transforms.ToPILImage()(sal.unsqueeze(0).cpu())
            sal_pil.save(OUT_DIR / f"{input_path.stem}git.png")

            print(f"✅ Saved to {OUT_DIR}")


if __name__ == "__main__":
    process_image(INPUT_PATH)