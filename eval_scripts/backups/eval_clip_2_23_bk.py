# # Ref: https://lightning.ai/docs/torchmetrics/stable/multimodal/clip_score.html

# import os
# import glob
# import argparse
# from PIL import Image
# import torch
# from torchvision import transforms
# from torchmetrics.multimodal.clip_score import CLIPScore
# import warnings

# warnings.filterwarnings(
#     "ignore",
#     message="`clean_up_tokenization_spaces` was not set",
#     category=FutureWarning,
# )

# def parse_args():
#     p = argparse.ArgumentParser()
#     p.add_argument("--input", required=True, help="image file or folder")
#     p.add_argument("--caption", required=True, help="GT caption")
#     p.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
#     # NOTE: optional suffix (default keeps old behavior)
#     p.add_argument("--suffix", default="_warp_relight_unwarp.png",
#                    help="image suffix filter; use '' to evaluate all images")
#     return p.parse_args()

# def collect_images(path, suffix):
#     if os.path.isfile(path):
#         if suffix == "" or path.endswith(suffix):
#             return [path]
#         else:
#             print("File ignored (suffix mismatch)")
#             return []
#     else:
#         if suffix == "":
#             # no suffix → load all common images
#             exts = ("*.png", "*.jpg", "*.jpeg", "*.webp")
#             files = []
#             for e in exts:
#                 files.extend(glob.glob(os.path.join(path, e)))
#             return sorted(files)
#         else:
#             return sorted(glob.glob(os.path.join(path, f"*{suffix}")))

# def main():
#     args = parse_args()
#     device = args.device

#     metric = CLIPScore(model_name_or_path="openai/clip-vit-large-patch14").to(device)
#     to_tensor = transforms.ToTensor()

#     img_paths = collect_images(args.input, args.suffix)
#     if len(img_paths) == 0:
#         print("No valid images found.")
#         return

#     scores = []
#     for p in img_paths:
#         img = Image.open(p).convert("RGB")
#         image = (to_tensor(img) * 255).to(torch.uint8).to(device)
#         score = metric(image, args.caption)
#         scores.append(float(score))

#     print(f"{sum(scores) / len(scores):.3f}")

# if __name__ == "__main__":
#     main()