# import argparse
# from pathlib import Path
# from PIL import Image
# import torch
# from torchvision import transforms
# import torchvision.transforms.functional as F

# '''
# Example Usage:
# python src/inference_debug_paired.py \
#     --warp_src /home/shenzhen/Datasets/VITON/test/image/00055_00.jpg \
#     --unwarp_src "/home/shenzhen/Relight_Projects/img2img-turbo/data/ChatGPT Image Feb 11, 2026, 09_21_46 PM.png" \
#     --out output/debug_gpt52.png \
#     --bw 128 \
#     --include-eyes
# '''

# from warp_utils.warp_pipeline import (
#     detect_face_bbox,
#     apply_forward_warp,
#     apply_unwarp,
#     get_face_app,
# )

# def load_as_tensor(pil_img: Image.Image, target_size: int, use_fp16: bool):
#     # Resize to square like your original pipeline
#     pil_img = pil_img.resize((target_size, target_size), Image.LANCZOS)
#     t = F.to_tensor(pil_img).unsqueeze(0).cuda()  # [1,3,H,W], in [0,1]
#     if use_fp16:
#         t = t.half()
#     return t

# @torch.no_grad()
# def main():
#     ap = argparse.ArgumentParser()
#     ap.add_argument("--warp_src", type=str, required=True, help="Image used to COMPUTE warp grid (face bbox etc.)")
#     ap.add_argument("--unwarp_src", type=str, required=True, help="Image to be UNWARPED using that grid")
#     ap.add_argument("--out", type=str, required=True, help="Output path for unwarped image")
#     ap.add_argument("--target_size", type=int, default=784)
#     ap.add_argument("--bw", type=int, default=80, help="Same bw you used before (must be >0 to create grid)")
#     ap.add_argument("--use_fp16", action="store_true")
#     ap.add_argument(
#         "--no-separable",
#         action="store_false",
#         dest="separable",
#         help="Disable separable KDE grid (default: True)",
#     )
#     ap.add_argument("--include-eyes", action="store_true")
#     ap.set_defaults(separable=True)
#     args = ap.parse_args()

#     device = torch.device("cuda")

#     face_app = get_face_app()

#     # -------------------------
#     # A) Compute warp grid from warp_src
#     # -------------------------
#     img1 = Image.open(args.warp_src).convert("RGB")
#     t1 = load_as_tensor(img1, args.target_size, args.use_fp16)

#     bbox = detect_face_bbox(img1, face_app, include_eyes=args.include_eyes)  # expects PIL image
#     bbox = bbox.to(device)

#     # We discard "warped", keep "warp_grid"
#     warped, warp_grid = apply_forward_warp(t1, bbox, args.bw, args.separable)
#     _, _, H, W = warped.shape

#     # -------------------------
#     # B) Unwarp a second image using warp_grid
#     # -------------------------
#     img2 = Image.open(args.unwarp_src).convert("RGB")
#     img2 = img2.resize((W, H), Image.LANCZOS)   # resize to warped size
#     t2 = F.to_tensor(img2).unsqueeze(0).cuda()
#     if args.use_fp16:
#         t2 = t2.half()

#     # apply_unwarp expects a tensor in the same normalized space as your pipeline.
#     # If your images are in [0,1], that's fine; unwarp will just remap pixels.
#     out_t = apply_unwarp(warp_grid, t2, args.separable)

#     # Save
#     out_img = transforms.ToPILImage()(out_t[0].float().cpu().clamp(0, 1))
#     Path(args.out).parent.mkdir(parents=True, exist_ok=True)
#     out_img.save(args.out)
#     print(f"✅ saved: {args.out}")

# if __name__ == "__main__":
#     main()