import os
os.environ["INSIGHTFACE_FORCE_TORCH"] = "1"
import shutil
import torch
import argparse
from PIL import Image
from tqdm import tqdm

# ✅ use the shared warp pipeline utils
from src.warp_utils.warp_pipeline import get_face_app, detect_face_bbox, apply_forward_warp
from src.warp_utils.warping_layers import invert_grid, save_img_warp, load_img_warp

# ============================================================
# 1. Args
# ============================================================
parser = argparse.ArgumentParser()
parser.add_argument("--target_prefix", type=str, required=True, help="e.g. exp_10_11")
parser.add_argument("--relight_type", type=str, required=True, help="e.g. candlelight_1")
parser.add_argument("--bw", type=int, default=512, help="bandwidth scale")
parser.add_argument(
    "--no-separable",
    action="store_false",
    dest="separable",
    help="Disable separable KDE grid (default: True)"
)
parser.set_defaults(separable=True)
args = parser.parse_args()

target_prefix = args.target_prefix
relight_type = args.relight_type
bandwidth_scale = args.bw

print(f"🔧 Using target_prefix={target_prefix}, relight_type={relight_type}, bw={bandwidth_scale}")

# ============================================================
# 2. Dataset paths
# ============================================================
input_root  = f"/home/shenzhen/Datasets/relighting/{target_prefix}/{relight_type}"

# output_root = f"/home/shenzhen/Datasets/relighting/{target_prefix}_warped_{bandwidth_scale}/{relight_type}"
mode_tag = "" if args.separable else "_nonsep"
output_root = f"/home/shenzhen/Datasets/relighting/{target_prefix}_warped_{bandwidth_scale}{mode_tag}/{relight_type}"

subfolders_to_warp = ["train_A", "test_A"]
subfolders_to_copy = ["train_B", "test_B"]

# ============================================================
# 3. Init face detector once
# ============================================================
device = "cuda" if torch.cuda.is_available() else "cpu"
face_app = get_face_app(local_rank=0)

# ============================================================
# 4. Core function: warp one image
# ============================================================
def process_image(img_path, warped_dir):
    # keep PIL only for face detection (warp pipeline expects PIL here)
    img_pil = Image.open(img_path).convert("RGB")

    # >>> use the original loader to keep the exact normalization
    c_t = load_img_warp(img_path).to(device)

    # --- FACE DETECT + FORWARD WARP ---
    bbox = detect_face_bbox(img_pil, face_app).to(device)
    warped_img, warp_grid = apply_forward_warp(c_t, bbox, bandwidth_scale, separable=args.separable)

    # --- Compute inverse grid ---
    _, _, h, w = c_t.shape
    inverse_grid = invert_grid(warp_grid, (1, 3, h, w), separable=args.separable)

    # --- Save warped image + inverse grid ---
    base = os.path.splitext(os.path.basename(img_path))[0]
    warped_path = os.path.join(warped_dir, f"{base}.png")
    inv_grid_path = os.path.join(warped_dir, f"{base}.inv.pth")

    save_img_warp(warped_img, warped_path)
    torch.save(inverse_grid.cpu(), inv_grid_path)

# ============================================================
# 5. Folder helpers
# ============================================================
def process_and_warp_folder(sub_folder):
    src_dir = os.path.join(input_root, sub_folder)
    dst_dir = os.path.join(output_root, sub_folder)
    os.makedirs(dst_dir, exist_ok=True)

    imgs = sorted([f for f in os.listdir(src_dir) if f.lower().endswith((".png", ".jpg", ".jpeg"))])
    for fname in tqdm(imgs, desc=f"Warping {sub_folder}", ncols=80):
        process_image(os.path.join(src_dir, fname), dst_dir)

def symlink_folder(sub_folder):
    src_dir = os.path.join(input_root, sub_folder)
    dst_dir = os.path.join(output_root, sub_folder)
    os.makedirs(dst_dir, exist_ok=True)

    imgs = sorted([f for f in os.listdir(src_dir) if f.lower().endswith((".png", ".jpg", ".jpeg"))])
    for fname in imgs:
        src = os.path.join(src_dir, fname)
        dst = os.path.join(dst_dir, fname)
        if not os.path.exists(dst):
            os.symlink(os.path.abspath(src), dst)
    print(f"🔗 Linked {sub_folder} → {dst_dir}")

# ============================================================
# 6. Main
# ============================================================
if __name__ == "__main__":
    os.makedirs(output_root, exist_ok=True)

    # Warp train_A + test_A
    for sub in subfolders_to_warp:
        process_and_warp_folder(sub)

    # Symlink train_B + test_B
    for sub in subfolders_to_copy:
        symlink_folder(sub)

    # Copy JSON prompts
    for json_name in ["train_prompts.json", "test_prompts.json"]:
        src = os.path.join(input_root, json_name)
        dst = os.path.join(output_root, json_name)
        if os.path.exists(src):
            shutil.copy2(src, dst)
            print(f"📄 Copied {json_name}")
        else:
            print(f"⚠️ {json_name} not found in {input_root}")

    print("✅ Warp + Copy complete. Warped dataset is now self-contained ✅")