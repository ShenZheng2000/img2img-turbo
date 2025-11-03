import os
import argparse
import numpy as np
from PIL import Image
import torch
from torchvision import transforms
import torchvision.transforms.functional as F
from pix2pix_turbo import Pix2Pix_Turbo
from tqdm import tqdm
from pathlib import Path
from PIL import ImageOps

# ✅ New imports for face detection + warp
# from warp_utils.warping_layers import PlainKDEGrid, warp, invert_grid
# from train_pix2pix_turbo import unwarp  # still used for final unwarp()
from warp_utils.warp_pipeline import detect_face_bbox, apply_forward_warp, apply_unwarp
from warp_utils.warp_pipeline import get_face_app


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--input_dir', type=str, required=True, help='path to the folder containing input images')
    parser.add_argument('--prompt', type=str, required=True, help='the prompt to be used')
    parser.add_argument('--model_name', type=str, default='', help='name of the pretrained model to be used')
    parser.add_argument('--model_path', type=str, default='', help='path to a model state dict to be used')
    parser.add_argument('--output_dir', type=str, default='output', help='the directory to save the output')
    parser.add_argument('--use_fp16', action='store_true', help='Use Float16 precision for faster inference')
    parser.add_argument('--target_size', type=int, default=784,
                        help='Resize input images for both width and height (default: 784)')
    parser.add_argument('--bw', type=int, default=0,
                        help='Bandwidth scale for face-based warping (default: 0, which disables warping)')
    args = parser.parse_args()

    if args.model_name == '' != args.model_path == '':
        raise ValueError('Either model_name or model_path should be provided')

    os.makedirs(args.output_dir, exist_ok=True)

    # ==============================================================
    # ✅ Load model
    # ==============================================================
    model = Pix2Pix_Turbo(pretrained_name=args.model_name, pretrained_path=args.model_path)
    model.set_eval()
    if args.use_fp16:
        model.half()

    # ✅ Initialize face detector once
    face_app = get_face_app()

    # ==============================================================
    # ✅ AUTO-DETECT FOLDER STRUCTURE (SpreeAI-style)
    # ==============================================================
    input_root = Path(args.input_dir)
    flat_images = sorted([p for p in input_root.iterdir() if p.suffix.lower() in ('.png', '.jpg', '.jpeg')])

    if len(flat_images) > 0:
        img_list = flat_images
        nested_mode = False
    else:
        # ✅ Only look one level deep, and only take files starting with "bdy_"
        img_list = sorted([
            p for p in input_root.glob('*/bdy_*')
            if p.suffix.lower() in ('.png', '.jpg', '.jpeg', '.webp')
        ])
        nested_mode = True

    # ==============================================================
    for input_path in tqdm(img_list, desc="Generating images"):
        # ✅ preserve subfolder structure if nested
        if nested_mode:
            rel_path = input_path.relative_to(input_root)
            output_path = Path(args.output_dir) / rel_path
            output_path.parent.mkdir(parents=True, exist_ok=True)
        else:
            output_path = Path(args.output_dir) / input_path.name

        # ==================== PREPROCESS ====================
        ### >>> MODIFIED: locate and crop using mask inside pre_processing/
        mask_path = input_path.parent / "pre_processing" / "black_fg_mask_groundedsam2.png"

        if mask_path.exists():
            img = Image.open(input_path).convert("RGB")
            mask = Image.open(mask_path).convert("L")
            inverted_mask = ImageOps.invert(mask)
            bbox = inverted_mask.getbbox()

            if bbox is None:
                bbox = (0, 0, img.width, img.height)

            input_image = img.crop(bbox)
            cropped_size = input_image.size
        else:
            print("WARNING: Mask not found, using full image.")
            input_image = Image.open(input_path).convert("RGB")
            cropped_size = input_image.size
        ### <<< MODIFIED <<<

        # 🔧 Fixed resize for stable inference
        input_image = input_image.resize((args.target_size, args.target_size), Image.LANCZOS)

        # ==================== INFERENCE (with face warp/unwarp) ====================
        with torch.no_grad():
            c_t = F.to_tensor(input_image).unsqueeze(0).cuda()
            if args.use_fp16:
                c_t = c_t.half()

            # --- FACE DETECT + FORWARD WARP ---
            if args.bw > 0:
                bbox = detect_face_bbox(input_image, face_app)
                warped, warp_grid = apply_forward_warp(c_t, bbox.to(c_t.device), args.bw)
                output_image = model(warped, args.prompt)
                output_image = apply_unwarp(warp_grid, output_image)
            else:
                output_image = model(c_t, args.prompt)

            # ==================== SAVE ====================
            output_pil = transforms.ToPILImage()(output_image[0].cpu() * 0.5 + 0.5)

            ### >>> MODIFIED: resize with preserved aspect ratio (same as fg mask)
            cw, ch = cropped_size
            aspect_ratio = cw / ch

            # keep the longest side as 784 (args.target_size)
            if aspect_ratio >= 1:
                # landscape: width longer
                new_w, new_h = args.target_size, int(args.target_size / aspect_ratio)
            else:
                # portrait: height longer
                new_w, new_h = int(args.target_size * aspect_ratio), args.target_size

            # resize while preserving mask ratio (no stretching)
            output_pil = output_pil.resize((new_w, new_h), Image.LANCZOS)
            ### <<< MODIFIED <<<

            # # ✅ Resize back keeping aspect ratio, longest side = target_size
            # orig_w, orig_h = orig_size
            # aspect_ratio = orig_w / orig_h
            # if aspect_ratio >= 1:
            #     new_w, new_h = args.target_size, int(args.target_size / aspect_ratio)
            # else:
            #     new_w, new_h = int(args.target_size * aspect_ratio), args.target_size
            # output_pil = output_pil.resize((new_w, new_h), Image.LANCZOS)

            # ✅ Always save output as PNG (lossless, better visualization)
            output_path = output_path.with_suffix('.png')
            output_pil.save(output_path)