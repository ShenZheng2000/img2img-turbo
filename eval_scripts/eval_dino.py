#!/usr/bin/env python3
import argparse
from pathlib import Path
import numpy as np
from PIL import Image
from tqdm import tqdm
import torch
from src.my_utils.dino_struct import DinoStructureLoss

IMG_EXTS = {".png", ".jpg", ".jpeg", ".webp", ".bmp"}


def list_images(folder: str, suffix: str | None = None):
    if suffix == "":
        suffix = None

    p = Path(folder)
    files = []
    for x in p.iterdir():
        if x.is_file() and x.suffix.lower() in IMG_EXTS:
            if suffix is None or x.stem.endswith(suffix):
                files.append(x)

    return sorted(files, key=lambda x: x.name)


def compute_dino_struct(src_dir: str, gen_dir: str, gen_suffix: str | None, max_pairs: int, resize: int):

    src_paths = list_images(src_dir)
    gen_paths = list_images(gen_dir, suffix=gen_suffix)

    if len(src_paths) != len(gen_paths):
        print("\n[ERROR] Number of images mismatch after filtering!")
        print(f"  src images: {len(src_paths)}")
        print(f"  gen images: {len(gen_paths)}")
        print("Please check naming, missing files, or suffix filter.\n")
        raise RuntimeError("DINO-Struct pairing mismatch — aborting.")

    n = len(src_paths)

    if max_pairs is not None and max_pairs > 0:
        n = min(n, max_pairs)

    if n == 0:
        raise RuntimeError("No matched images found.")

    net_dino = DinoStructureLoss()
    scores = []

    for i in tqdm(range(n), desc="DINO-Struct"):

        try:
            src_pil = Image.open(src_paths[i]).convert("RGB")
            gen_pil = Image.open(gen_paths[i]).convert("RGB")
        except Exception:
            print(f"[SKIP] corrupted image: {gen_paths[i]}")
            continue

        if resize and resize > 0:
            src_pil = src_pil.resize((resize, resize), Image.BICUBIC)
            gen_pil = gen_pil.resize((resize, resize), Image.BICUBIC)

        with torch.no_grad():
            a = net_dino.preprocess(src_pil).unsqueeze(0).cuda()
            b = net_dino.preprocess(gen_pil).unsqueeze(0).cuda()
            s = net_dino.calculate_global_ssim_loss(a, b).item()

        scores.append(s)

    if len(scores) == 0:
        raise RuntimeError("All image pairs failed to load.")

    return float(np.mean(scores)), float(np.std(scores)), len(scores)


def main():

    ap = argparse.ArgumentParser()
    ap.add_argument("--real_A", type=str, required=True)
    ap.add_argument("--result_A", type=str, required=True)
    ap.add_argument("--gen_suffix", type=str, default="")
    ap.add_argument("--eval_resize", type=int, default=0)
    ap.add_argument("--max_pairs", type=int, default=-1)

    args = ap.parse_args()

    mean, std, n = compute_dino_struct(
        src_dir=args.real_A,
        gen_dir=args.result_A,
        gen_suffix=args.gen_suffix,
        max_pairs=args.max_pairs,
        resize=args.eval_resize,
    )

    print("\n###############################################")
    print(f"# Valid Pairs: {n}")
    print(f"# DINO-Struct (x100): {mean*100.0:.4f} ± {std*100.0:.4f}")
    print("###############################################\n")


if __name__ == "__main__":
    main()