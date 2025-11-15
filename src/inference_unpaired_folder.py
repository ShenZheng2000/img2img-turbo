import os
import argparse
from PIL import Image
import torch
from torchvision import transforms
from tqdm import tqdm
from cyclegan_turbo import CycleGAN_Turbo
from my_utils.training_utils import build_transform
from warp_utils.warp_pipeline import get_gt_bbox, load_bbox_map, apply_forward_warp, apply_unwarp
from pathlib import Path

def is_image_file(filename):
    return any(filename.endswith(extension) for extension in ['.png', '.jpg', '.jpeg', '.bmp', '.gif'])


# ============================================================
# Helper: parse arguments
# ============================================================
def parse_args():
    parser = argparse.ArgumentParser()

    parser.add_argument('--input_dir', type=str, required=True)
    parser.add_argument('--prompt', type=str, required=False)
    parser.add_argument('--model_name', type=str, default=None)
    parser.add_argument('--model_path', type=str, default=None)
    parser.add_argument('--output_dir', type=str, default=None)
    parser.add_argument('--image_prep', type=str, default='resize_512x512')
    parser.add_argument('--direction', type=str, default=None)

    # warp-related
    parser.add_argument('--bw', type=int, default=0)
    parser.add_argument('--train-bbox-json', type=str, default=None)
    parser.add_argument('--val-bbox-json', type=str, default=None)
    parser.add_argument('--separable', action="store_true", default=True)

    args = parser.parse_args()

    # validation
    if (args.model_name is None) == (args.model_path is None):
        raise ValueError("You must provide EITHER model_name OR model_path, but not both.")

    if args.model_path is not None and args.prompt is None:
        raise ValueError("Prompt is required when loading a custom model_path.")

    if args.model_name is not None:
        if args.prompt is not None:
            raise ValueError("Prompt is not required when loading a pretrained model.")
        if args.direction is not None:
            raise ValueError("Direction is not required when loading a pretrained model.")

    return args


# ============================================================
# Main processing for one image
# ============================================================
def process_image(input_path, output_path, model, bbox_map, args, T_val):
    input_image = Image.open(input_path).convert('RGB')
    orig_w, orig_h = input_image.width, input_image.height
    out_path = Path(output_path)
    ext = out_path.suffix   # keep original extension (.jpg / .png)

    with torch.no_grad():
        input_img = T_val(input_image)
        
        # --- warp / relight / unwarp pipeline ---
        if args.bw > 0:

            x_01 = transforms.ToTensor()(input_img).unsqueeze(0).cuda()   # UNNORMALIZE version of x_t

            # (0) get GT bbox and apply forward warp
            base_name = os.path.basename(input_path)
            bbox = get_gt_bbox(base_name, input_image, bbox_map, device=x_01.device)
            # warped, warp_grid = apply_forward_warp(x_t, bbox, bw=args.bw, separable=args.separable)
            warped_01, warp_grid = apply_forward_warp(x_01, bbox, bw=args.bw, separable=args.separable)

            # (1) save warped image
            warped_pil = transforms.ToPILImage()(warped_01[0].cpu().clamp(0,1))
            warped_pil = warped_pil.resize((orig_w, orig_h), Image.LANCZOS)
            warped_pil.save(out_path.with_name(out_path.stem + "_warp" + ext))

            # (2) model on warped
            warped_norm = (warped_01 * 2.0 - 1.0)
            output_image = model(warped_norm, direction=args.direction, caption=args.prompt)

            # (3) save warped + relit
            warped_relit_pil = transforms.ToPILImage()(output_image[0].cpu() * 0.5 + 0.5)
            warped_relit_pil = warped_relit_pil.resize((orig_w, orig_h), Image.LANCZOS)
            warped_relit_pil.save(out_path.with_name(out_path.stem + "_warp_relight" + ext))

            # (4) unwarp back
            output_image = apply_unwarp(warp_grid, output_image, separable=args.separable)

        else:
            # no warp → just model(x_t)
            x_t = transforms.ToTensor()(input_img)
            x_t = transforms.Normalize([0.5], [0.5])(x_t).unsqueeze(0).cuda()
            output_image = model(x_t, direction=args.direction, caption=args.prompt)

        # (5) ALWAYS save final as warp_relight_unwarp
        final_pil = transforms.ToPILImage()(output_image[0].cpu() * 0.5 + 0.5)
        final_pil = final_pil.resize((orig_w, orig_h), Image.LANCZOS)
        final_pil.save(out_path.with_name(out_path.stem + "_warp_relight_unwarp" + ext))


# ============================================================
# Main
# ============================================================
def main():
    args = parse_args()

    model = CycleGAN_Turbo(pretrained_name=args.model_name, pretrained_path=args.model_path)
    model.eval()
    model.unet.enable_xformers_memory_efficient_attention()

    T_val = build_transform(args.image_prep)

    os.makedirs(args.output_dir, exist_ok=True)

    bbox_map = load_bbox_map(args.train_bbox_json, args.val_bbox_json)

    # gather images
    image_paths = []
    for root, _, files in os.walk(args.input_dir):
        for filename in files:
            if is_image_file(filename):
                image_paths.append(os.path.join(root, filename))

    # run
    for input_path in tqdm(image_paths, desc="Processing images", ncols=100):
        relative_root = os.path.relpath(os.path.dirname(input_path), args.input_dir)
        if relative_root == ".":
            relative_root = ""
        output_folder = os.path.join(args.output_dir, relative_root)
        os.makedirs(output_folder, exist_ok=True)

        filename = os.path.basename(input_path)
        output_path = os.path.join(output_folder, filename)

        # process
        process_image(input_path, output_path, model, bbox_map, args, T_val)


if __name__ == "__main__":
    main()