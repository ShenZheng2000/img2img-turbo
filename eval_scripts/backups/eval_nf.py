import os
import glob
import argparse
import torch
import pyiqa
from PIL import Image
import torchvision.transforms.functional as TF
import contextlib, io
import warnings
warnings.filterwarnings("ignore", message="pkg_resources is deprecated")

# -----------------------
# Args (MIN)
# -----------------------
def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--input", required=True, help="image folder")
    p.add_argument("--suffix", default="warp_relight_unwarp.png",
                   help="'' = all images, or e.g. 'warp_relight_unwarp.png'")
    p.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    p.add_argument("--metrics", default="musiq,clipiqa,dbcnn,nima",
                   help="comma-separated, e.g. musiq,clipiqa")
    return p.parse_args()

# -----------------------
# Main
# -----------------------
def main():
    args = parse_args()
    device = torch.device(args.device)

    # GPU id for printing
    if device.type == "cuda":
        gpu_id = int(os.environ.get("CUDA_VISIBLE_DEVICES", "").split(",")[0]) if os.environ.get("CUDA_VISIBLE_DEVICES") else torch.cuda.current_device()
    else:
        gpu_id = "cpu"

    METRICS = [m.strip() for m in args.metrics.split(",") if m.strip()]

    # suppress pyiqa init prints
    buf = io.StringIO()
    with contextlib.redirect_stdout(buf), contextlib.redirect_stderr(buf):
        metrics = {m: pyiqa.create_metric(m, device=device) for m in METRICS}

    # collect paths
    if args.suffix == "":
        paths = sorted(glob.glob(os.path.join(args.input, "*")))
    else:
        paths = sorted(glob.glob(os.path.join(args.input, f"*{args.suffix}")))
    paths = [p for p in paths if p.lower().endswith((".png", ".jpg", ".jpeg", ".bmp", ".webp"))]

    n_img = len(paths)
    if n_img == 0:
        print(f"[GPU {gpu_id}] DIR: {args.input} | N=0")
        return

    def load_img(p):
        img = Image.open(p).convert("RGB")
        return TF.to_tensor(img).unsqueeze(0).to(device)

    sums = {m: 0.0 for m in METRICS}

    with torch.no_grad():
        for p in paths:
            x = load_img(p)
            for m, metric in metrics.items():
                sums[m] += float(metric(x))

    # ---- scores in ONE row ----
    parts = [f"{m}={sums[m]/n_img:.3f}" for m in METRICS]
    print("Scores: " + " | ".join(parts))

if __name__ == "__main__":
    main()