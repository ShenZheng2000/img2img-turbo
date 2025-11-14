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
from warp_utils.warp_pipeline import detect_face_bbox, apply_forward_warp, apply_unwarp, get_face_app, resize_longest_side, crop_to_foreground


# ============================================================
# Helper: parse arguments
# ============================================================
def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--input_dir', type=str, required=True)
    parser.add_argument('--prompt', type=str, required=True)
    parser.add_argument('--model_name', type=str, default='')
    parser.add_argument('--model_path', type=str, default='')
    parser.add_argument('--output_dir', type=str, default='output')
    parser.add_argument('--use_fp16', action='store_true')
    parser.add_argument('--target_size', type=int, default=784)
    parser.add_argument('--bw', type=int, default=0)
    parser.add_argument(
        "--no-separable",
        action="store_false",
        dest="separable",
        help="Disable separable KDE grid (default: True)"
    )
    parser.add_argument(
        "--include-eyes",
        action="store_true",
        help="If set, include left/right eye boxes in face detection"
    )
    parser.set_defaults(separable=True)
    args = parser.parse_args()

    if args.model_name == '' != args.model_path == '':
        raise ValueError('Either model_name or model_path should be provided')

    os.makedirs(args.output_dir, exist_ok=True)
    return args



# ============================================================
# Main processing for one image
# ============================================================
def process_image(input_path, model, face_app, args):
    img, cropped_size = crop_to_foreground(input_path)

    img = img.resize((args.target_size, args.target_size), Image.LANCZOS)

    c_t = F.to_tensor(img).unsqueeze(0).cuda()
    if args.use_fp16:
        c_t = c_t.half()

    # prepare output folder structure
    rel_dir = input_path.parent.relative_to(args.input_dir)
    out_dir = Path(args.output_dir) / rel_dir
    out_dir.mkdir(parents=True, exist_ok=True)
    output_path = out_dir / input_path.name

    # --- warp / relight / unwarp pipeline ---
    with torch.no_grad():
        if args.bw > 0:

            # (0) get bbox and apply forward warp
            bbox = detect_face_bbox(img, face_app, include_eyes=args.include_eyes)
            warped, warp_grid = apply_forward_warp(c_t, bbox.to(c_t.device), args.bw, args.separable)

            # (1) save warped image
            # print(f"📊 warped tensor range: min={warped.min().item():.3f}, max={warped.max().item():.3f}")
            warped_pil = transforms.ToPILImage()(warped[0].cpu().clamp(0, 1))
            warped_pil = resize_longest_side(warped_pil, cropped_size, args.target_size)
            warped_pil.save(output_path.with_name(output_path.stem + "_warp.png"))

            # (2) relight
            output_image = model(warped, args.prompt)
            # print(f"📊 relit tensor range:  min={output_image.min().item():.3f}, max={output_image.max().item():.3f}")

            # (3) save warped+relit
            warped_relit_pil = transforms.ToPILImage()(output_image[0].cpu() * 0.5 + 0.5)
            warped_relit_pil = resize_longest_side(warped_relit_pil, cropped_size, args.target_size)
            warped_relit_pil.save(output_path.with_name(output_path.stem + "_warp_relight.png"))

            # (4) unwarp back
            output_image = apply_unwarp(warp_grid, output_image, args.separable)
        else:
            output_image = model(c_t, args.prompt)

        # (5) save final warp→relight→unwarp
        # print(f"📊 final unwarped range: min={output_image.min().item():.3f}, max={output_image.max().item():.3f}")
        output_pil = transforms.ToPILImage()(output_image[0].cpu() * 0.5 + 0.5)
        output_pil = resize_longest_side(output_pil, cropped_size, args.target_size)
        final_path = output_path.with_name(output_path.stem + "_warp_relight_unwarp.png")
        output_pil.save(final_path)
        print(f"✅ Saved results for {input_path.name} in {out_dir}")


# ============================================================
# Main
# ============================================================
def main():
    args = parse_args()

    model = Pix2Pix_Turbo(pretrained_name=args.model_name, pretrained_path=args.model_path)
    model.set_eval()
    if args.use_fp16:
        model.half()

    face_app = get_face_app()

    input_root = Path(args.input_dir)
    subfolders = sorted([p for p in input_root.iterdir() if p.is_dir()])

    img_list = []
    for folder in subfolders:
        # find all body images
        body_imgs = sorted([
            p for p in folder.glob("bdy_*.*")
            if p.suffix.lower() in (".jpg", ".png", ".jpeg", ".webp")
        ])
        if len(body_imgs) == 1:
            img_list.append(body_imgs[0])
        elif len(body_imgs) == 0:
            print(f"⚠️ No body images found in {folder}")
        else:
            print(f"⚠️ Multiple body images found in {folder}, skipping.")

    print(f"Found {len(img_list)} body images to process (one per folder).")

    for input_path in tqdm(img_list, desc="Processing"):
        process_image(input_path, model, face_app, args)


if __name__ == "__main__":
    main()