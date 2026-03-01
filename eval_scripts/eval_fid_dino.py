#!/usr/bin/env python3
import argparse
import tempfile
import shutil
from pathlib import Path
import numpy as np
from PIL import Image
from tqdm import tqdm
import torch
from cleanfid import fid
from src.my_utils.dino_struct import DinoStructureLoss
import os

IMG_EXTS = {".png", ".jpg", ".jpeg", ".webp", ".bmp"}


def list_images(folder: str, suffix: str = None):
    p = Path(folder)
    files = []
    for x in p.iterdir():
        if x.is_file() and x.suffix.lower() in IMG_EXTS:
            if suffix is None or x.stem.endswith(suffix):
                files.append(x)
    return sorted(files, key=lambda x: x.name)


def compute_dino_struct(src_dir: str, gen_dir: str, gen_suffix: str = None, max_pairs: int = -1, resize: int = 0):
    # DINO-Struct compares <src, gen>
    src_paths = list_images(src_dir)
    gen_paths = list_images(gen_dir, suffix=gen_suffix)

    # ✅ enforce strict matching count
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

    # ✅ print preview of pairs
    # print("\n[DINO-Struct] Preview first 5 pairs:")
    # for i in range(min(5, n)):
    #     print(f"  {src_paths[i].name}  <->  {gen_paths[i].name}")
    # print()

    net_dino = DinoStructureLoss()
    scores = []

    for i in tqdm(range(n), desc="DINO-Struct"):
        src_pil = Image.open(src_paths[i]).convert("RGB")
        gen_pil = Image.open(gen_paths[i]).convert("RGB")

        if resize > 0:
            src_pil = src_pil.resize((resize, resize), Image.BICUBIC)
            gen_pil = gen_pil.resize((resize, resize), Image.BICUBIC)

        with torch.no_grad():
            a = net_dino.preprocess(src_pil).unsqueeze(0).cuda()
            b = net_dino.preprocess(gen_pil).unsqueeze(0).cuda()
            s = net_dino.calculate_global_ssim_loss(a, b).item()

        scores.append(s)

    return float(np.mean(scores)), float(np.std(scores)), n


def compute_fid_filtered(gen_dir: str, tgt_dir: str, mode: str, gen_suffix: str = None, resize: int = 0):
    if resize != 0:
        raise RuntimeError("This symlink version assumes eval_resize=0. Set eval_resize=0 or keep the copy+resize path.")

    gen_paths = list_images(gen_dir, suffix=gen_suffix)
    if len(gen_paths) == 0:
        raise RuntimeError("No generated images match the given suffix.")

    with tempfile.TemporaryDirectory() as gen_tmp:
        gen_tmp = Path(gen_tmp)

        # symlink generated only
        for p in gen_paths:
            os.symlink(p.resolve(), gen_tmp / p.name)

        # direct compare: (filtered gen tmp) vs (real folder)
        return fid.compute_fid(str(gen_tmp), tgt_dir, mode=mode)


def main():
    parser = argparse.ArgumentParser()

    parser.add_argument("--real_A", type=str, required=True)
    parser.add_argument("--real_B", type=str, required=True)
    parser.add_argument("--result_A", type=str, required=True)
    parser.add_argument("--result_B", type=str, required=True)

    parser.add_argument("--mode", type=str, default="clean")
    parser.add_argument("--max_pairs", type=int, default=-1)
    parser.add_argument("--gen_suffix", type=str, default="warp_relight_unwarp")
    parser.add_argument("--eval_resize", type=int, default=0)

    args = parser.parse_args()

    fid_ab = compute_fid_filtered(
        gen_dir=args.result_A,
        tgt_dir=args.real_B,
        mode=args.mode,
        gen_suffix=args.gen_suffix,
        resize=args.eval_resize,
    )

    fid_ba = compute_fid_filtered(
        gen_dir=args.result_B,
        tgt_dir=args.real_A,
        mode=args.mode,
        gen_suffix=args.gen_suffix,
        resize=args.eval_resize,
    )

    dino_ab_mean, _, _ = compute_dino_struct(
        src_dir=args.real_A,
        gen_dir=args.result_A,
        gen_suffix=args.gen_suffix,
        max_pairs=args.max_pairs,
        resize=args.eval_resize,
    )

    dino_ba_mean, _, _ = compute_dino_struct(
        src_dir=args.real_B,
        gen_dir=args.result_B,
        gen_suffix=args.gen_suffix,
        max_pairs=args.max_pairs,
        resize=args.eval_resize,
    )

    print("\n################################################################################")
    print("| Metric       | A -> B          | B -> A          |")
    print("|------------------------------------------------------------------------------|")
    print(f"| FID          | {fid_ab:<15.4f} | {fid_ba:<15.4f} |")
    print(f"| DINO (x100)  | {dino_ab_mean * 100.0:<15.4f} | {dino_ba_mean * 100.0:<15.4f} |")
    print("################################################################################\n")


if __name__ == "__main__":
    main()