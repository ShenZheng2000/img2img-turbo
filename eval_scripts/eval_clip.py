import os
import glob
import argparse
from PIL import Image
import torch
from torchvision import transforms
from torchmetrics.multimodal.clip_score import CLIPScore
import ImageReward as RM
import contextlib, io
import warnings

warnings.filterwarnings("ignore")

# -----------------------
# Args
# -----------------------
def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--input", required=True, help="image file or folder")
    p.add_argument("--caption", required=True, help="text prompt")
    p.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    p.add_argument(
        "--suffix",
        default="_warp_relight_unwarp.png",
        help="image suffix filter; '' = all images",
    )
    p.add_argument(
        "--metric",
        default="both",
        choices=["clip", "imagereward", "both"],
        help="which metric to run",
    )
    p.add_argument(
        "--ir_bs",
        type=int,
        default=64,
        help="ImageReward batch size (reduce if OOM)",
    )
    return p.parse_args()

# -----------------------
# Collect images
# -----------------------
def collect_images(path, suffix):
    if os.path.isfile(path):
        return [path]

    if suffix == "":
        exts = ("*.png", "*.jpg", "*.jpeg", "*.webp")
        files = []
        for e in exts:
            files.extend(glob.glob(os.path.join(path, e)))
        return sorted(files)

    return sorted(glob.glob(os.path.join(path, f"*{suffix}")))

# -----------------------
# Main
# -----------------------
def main():
    args = parse_args()
    device = args.device

    # ✅ build merged caption
    final_caption = args.caption

    img_paths = collect_images(args.input, args.suffix)
    if len(img_paths) == 0:
        print("No valid images found.")
        return

    # ---------------- CLIP ----------------
    if args.metric in ("clip", "both"):
        clip_metric = CLIPScore(model_name_or_path="openai/clip-vit-large-patch14").to(device)
        to_tensor = transforms.ToTensor()

        clip_scores = []
        for p in img_paths:
            img = Image.open(p).convert("RGB")
            image = (to_tensor(img) * 255).to(torch.uint8).to(device)
            score = clip_metric(image, final_caption)
            clip_scores.append(float(score))

        print(f"CLIP  {sum(clip_scores)/len(clip_scores):.3f}")

    # ---------------- ImageReward ----------------
    if args.metric in ("imagereward", "both"):
        print("Loading ImageReward model...")

        f = io.StringIO()
        with contextlib.redirect_stdout(f):
            ir_model = RM.load("ImageReward-v1.0", device=device)

        scores_list = []
        bs = max(1, int(args.ir_bs))

        for i in range(0, len(img_paths), bs):
            batch_paths = img_paths[i : i + bs]
            batch_scores = ir_model.score(final_caption, batch_paths)

            # # TODO: remove this later for debug
            # print("batch_scores is ", batch_scores)
            # print("average batch score is ", sum(batch_scores)/len(batch_scores))
            # exit(0)

            if isinstance(batch_scores, float):
                scores_list.append(float(batch_scores))
            elif hasattr(batch_scores, "detach"):
                scores_list.extend(batch_scores.detach().cpu().flatten().tolist())
            else:
                scores_list.extend([float(x) for x in batch_scores])

        print(f"IMGRWD {sum(scores_list)/len(scores_list):.3f}")

# -----------------------
if __name__ == "__main__":
    main()