import os
import argparse
from pathlib import Path
from PIL import Image, ImageOps
import torch
from torchvision import transforms
import torchvision.transforms.functional as F
from tqdm import tqdm

from pix2pix_turbo import Pix2Pix_Turbo
from warp_utils.warp_pipeline import (
    detect_face_bbox,
    apply_forward_warp,
    apply_unwarp,
    get_face_app,
    resize_longest_side,
    crop_to_foreground,
    center_crop_pil,
    custom_classes,
    detect_yolo_bbox,
    largest_divisible_by_32_leq,
    resize_keep_aspect,
    resize_keep_aspect_min
)

from ultralytics import YOLOWorld

import time

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
    parser.add_argument('--center_crop', action='store_true',
                    help='If set, center-crop (or resize-up then crop) to target_size x target_size, instead of crop_to_foreground.')
    parser.add_argument(
        "--use-yoloworld",
        action="store_true",
        help="Use YOLO-World to detect bbox on-the-fly (no face_app bbox)."
    )    
    parser.add_argument(
        '--crop_resize_size',
        type=int,
        default=None,
        help='If set, after center crop to target_size, resize to this size for model inference.'
    )
    parser.add_argument(
        '--keep_aspect',
        action='store_true',
        help='Resize with aspect ratio preserved so the longest side becomes target_size.'
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
def process_image(input_path, model, face_app, yolo_model, args):
    # img, cropped_size = crop_to_foreground(input_path)
    # img = img.resize((args.target_size, args.target_size), Image.LANCZOS)

    if args.center_crop:
        img = Image.open(input_path).convert("RGB")

        if args.keep_aspect:
            img = resize_keep_aspect_min(img, args.target_size)

        # (1) center crop
        img = center_crop_pil(img, args.target_size, args.target_size)
        cropped_size = (args.target_size, args.target_size)

        # (2) resize here ONLY for no-warp inference
        if args.bw == 0:
            if (args.crop_resize_size is not None) and (args.crop_resize_size != args.target_size):
                img = img.resize(
                    (args.crop_resize_size, args.crop_resize_size),
                    Image.LANCZOS
                )
    else:
        img, cropped_size = crop_to_foreground(input_path)

        if args.keep_aspect:
            img = resize_keep_aspect(img, args.target_size)
        else:
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
            if args.use_yoloworld and yolo_model is not None:
                w, h = img.size
                imgsz = largest_divisible_by_32_leq(min(h, w))
                bbox = detect_yolo_bbox(img, yolo_model, imgsz=imgsz)
            else:
                bbox = detect_face_bbox(img, face_app, include_eyes=args.include_eyes)

            if bbox is None:
                output_image = model(c_t, args.prompt)
            else:
                
                # TODO: now hardcode crop_resize as warp_output_shape. change later
                # warp-resize instead of just warp
                if args.crop_resize_size is not None:
                    warp_output_shape = (args.crop_resize_size, args.crop_resize_size)
                else:
                    warp_output_shape = None

                warped, warp_grid, saliency = apply_forward_warp(
                    c_t,
                    bbox.to(c_t.device),
                    args.bw,
                    args.separable,
                    output_shape=warp_output_shape
                )

                # (1) save warped image
                # print(f"📊 warped tensor range: min={warped.min().item():.3f}, max={warped.max().item():.3f}")
                warped_pil = transforms.ToPILImage()(warped[0].cpu().clamp(0, 1))
                if not args.keep_aspect:
                    warped_pil = resize_longest_side(warped_pil, cropped_size, args.target_size)
                warped_pil.save(output_path.with_name(output_path.stem + "_warp.png"))

                # (1.5) save saliency map at native resolution (NO resize)
                sal = saliency.float()          # shape [1,1,Hs,Ws] e.g. 31x51
                sal = sal[0, 0]                 # -> [Hs, Ws]
                # normalize to [0,1]
                sal = (sal - sal.min()) / (sal.max() - sal.min() + 1e-8)
                sal_pil = transforms.ToPILImage()(sal.unsqueeze(0).cpu())  # 1xHs×Ws grayscale
                sal_pil.save(output_path.with_name(output_path.stem + "_saliency.png"))
                
                # (2) relight
                output_image = model(warped, args.prompt)
                # print(f"📊 relit tensor range:  min={output_image.min().item():.3f}, max={output_image.max().item():.3f}")

                # (3) save warped+relit
                warped_relit_pil = transforms.ToPILImage()(output_image[0].cpu() * 0.5 + 0.5)
                if not args.keep_aspect:
                    warped_relit_pil = resize_longest_side(warped_relit_pil, cropped_size, args.target_size)
                warped_relit_pil.save(output_path.with_name(output_path.stem + "_warp_relight.png"))

                # (4) unwarp back
                output_image = apply_unwarp(warp_grid, output_image, args.separable)
        else:
            output_image = model(c_t, args.prompt)

        # (5) save final warp→relight→unwarp
        # print(f"📊 final unwarped range: min={output_image.min().item():.3f}, max={output_image.max().item():.3f}")
        output_pil = transforms.ToPILImage()(output_image[0].cpu() * 0.5 + 0.5)
        if not args.keep_aspect:
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

    yolo_model = None
    if args.use_yoloworld:
        yolo_model = YOLOWorld("yolov8x-world.pt")
        yolo_model.set_classes(custom_classes)

    input_root = Path(args.input_dir)
    subfolders = sorted([p for p in input_root.iterdir() if p.is_dir()])

    img_list = []


    # --------------------------------------------------
    # Case 1: flat images: root/image/*.jpg
    # --------------------------------------------------
    images_dir = input_root / "image"
    if images_dir.exists() and images_dir.is_dir():
        img_list = sorted([
            p for p in images_dir.glob("*")
            if p.suffix.lower() in (".jpg", ".png", ".jpeg", ".webp")
        ])
        print(f"📁 Using flat image mode: {len(img_list)} images found in {images_dir}")

    # --------------------------------------------------
    # Case 2: legacy structure: root/subfolder/bdy_*.*
    # --------------------------------------------------
    else:
        subfolders = sorted([p for p in input_root.iterdir() if p.is_dir()])
        for folder in subfolders:
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

        print(f"📁 Using legacy folder mode: {len(img_list)} images found")

    if len(img_list) == 0:
        raise RuntimeError("❌ No images found to process.")

    torch.cuda.synchronize() # Wait for everything to be ready
    start_time = time.perf_counter()

    for input_path in tqdm(img_list, desc="Processing"):
        process_image(input_path, model, face_app, yolo_model, args)

    torch.cuda.synchronize()
    end_time = time.perf_counter()
    total_time = end_time - start_time
    print(f"\n⏱ Total inference time: {total_time:.2f} seconds")
    print(f"⏱ Avg time per image: {total_time / len(img_list):.3f} seconds")


if __name__ == "__main__":
    main()