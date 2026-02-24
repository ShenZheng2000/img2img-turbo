# # Ref: https://lightning.ai/docs/torchmetrics/stable/multimodal/clip_score.html
# # LAION aesthetic head Ref: https://github.com/LAION-AI/aesthetic-predictor

# import os
# import glob
# import argparse
# from PIL import Image
# import torch
# from torchvision import transforms
# from torchmetrics.multimodal.clip_score import CLIPScore

# # NEW (LAION aesthetic)
# import torch.nn as nn
# from os.path import expanduser
# from urllib.request import urlretrieve
# import open_clip

# import warnings
# warnings.filterwarnings("ignore")

# # ---------------------------
# # LAION aesthetic head loader (your code)
# # ---------------------------
# def get_aesthetic_model(clip_model="vit_l_14"):
#     """load the aesthetic model (linear head)"""
#     home = expanduser("~")
#     cache_folder = home + "/.cache/emb_reader"
#     path_to_model = cache_folder + "/sa_0_4_" + clip_model + "_linear.pth"
#     if not os.path.exists(path_to_model):
#         os.makedirs(cache_folder, exist_ok=True)
#         url_model = (
#             "https://github.com/LAION-AI/aesthetic-predictor/blob/main/sa_0_4_"
#             + clip_model
#             + "_linear.pth?raw=true"
#         )
#         urlretrieve(url_model, path_to_model)
#     if clip_model == "vit_l_14":
#         m = nn.Linear(768, 1)
#     elif clip_model == "vit_b_32":
#         m = nn.Linear(512, 1)
#     else:
#         raise ValueError()
#     s = torch.load(path_to_model, map_location="cpu", weights_only=True)
#     m.load_state_dict(s)
#     m.eval()
#     return m


# def parse_args():
#     p = argparse.ArgumentParser()
#     p.add_argument("--input", required=True, help="image file or folder")
#     p.add_argument("--caption", required=True, help="GT caption")
#     p.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
#     # NOTE: default MUST be _warp_relight_unwarp!
#     p.add_argument(
#         "--suffix",
#         default="_warp_relight_unwarp.png",
#         help="image suffix filter; use '' to evaluate all images",
#     )
#     p.add_argument(
#         "--metric",
#         default="both",
#         choices=["clip", "aesthetic", "both"],
#         help="which metric to run",
#     )    
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

#     to_tensor = transforms.ToTensor()

#     # init only what we need
#     clip_metric = None
#     aest_head, aest_clip, aest_preprocess = None, None, None

#     if args.metric in ("clip", "both"):
#         clip_metric = CLIPScore(model_name_or_path="openai/clip-vit-large-patch14").to(device)

#     if args.metric in ("aesthetic", "both"):
#         aest_head = get_aesthetic_model("vit_l_14").to(device)
#         aest_clip, _, aest_preprocess = open_clip.create_model_and_transforms(
#             "ViT-L-14", pretrained="openai"
#         )
#         aest_clip = aest_clip.to(device).eval()

#     img_paths = collect_images(args.input, args.suffix)
#     if len(img_paths) == 0:
#         print("No valid images found.")
#         return

#     clip_scores = []
#     aest_scores = []

#     for p in img_paths:
#         img = Image.open(p).convert("RGB")

#         if clip_metric is not None:
#             # CLIPScore (torchmetrics expects uint8 0..255)
#             image_u8 = (to_tensor(img) * 255).to(torch.uint8).to(device)
#             s_clip = clip_metric(image_u8, args.caption)
#             clip_scores.append(float(s_clip))

#         if aest_head is not None:
#             # Aesthetic score (LAION): embedding -> normalize -> linear head
#             x = aest_preprocess(img).unsqueeze(0).to(device)
#             with torch.no_grad():
#                 emb = aest_clip.encode_image(x)
#                 emb = emb / emb.norm(dim=-1, keepdim=True)
#                 s_a = aest_head(emb).squeeze().item()
#             aest_scores.append(float(s_a))

#     if len(clip_scores) > 0:
#         print(f"CLIP  {sum(clip_scores) / len(clip_scores):.3f}")
#     if len(aest_scores) > 0:
#         print(f"AESTH {sum(aest_scores) / len(aest_scores):.3f}")


# if __name__ == "__main__":
#     main()