import os
os.environ["INSIGHTFACE_FORCE_TORCH"] = "1"
import shutil
import torch
import argparse
from PIL import Image
from tqdm import tqdm
import json
import cv2
import numpy as np

# ✅ warp utils
from src.warp_utils.warp_pipeline import get_face_app, detect_face_bbox, apply_forward_warp
from src.warp_utils.warping_layers import invert_grid, save_img_warp, load_img_warp


# ============================================================
# 1. Parse Args
# ============================================================
def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--target_prefix", type=str, required=True)
    p.add_argument("--relight_type", type=str, default=None)
    p.add_argument("--bw", type=int, default=512)
    p.add_argument("--no-separable", action="store_false", dest="separable")
    p.add_argument("--include-eyes", action="store_true")
    p.add_argument("--input_root", type=str, default="/home/shenzhen/Datasets/relighting")
    p.add_argument("--train-bbox-json", type=str, default=None,
                help="COCO JSON file for training set bboxes")
    p.add_argument("--val-bbox-json", type=str, default=None,
                help="COCO JSON file for validation/test set bboxes")
    p.add_argument("--warp-subfolders", nargs="+", default=["train_A", "test_A"],
                help="List of subfolders to warp (others will be copied only)")
    p.set_defaults(separable=True)
    return p.parse_args()


# ============================================================
# 2. Load bbox map
# ============================================================
def load_bbox_map(train_json, val_json):
    """Load and merge COCO JSON bbox maps for train + val/test."""
    def _load_one(path):
        if not path or not os.path.exists(path):
            return {}
        with open(path, "r") as f:
            coco = json.load(f)
        id2name = {img["id"]: img["file_name"] for img in coco["images"]}
        out = {}
        for ann in coco["annotations"]:
            fn = id2name[ann["image_id"]]
            out.setdefault(fn, []).append(ann["bbox"])  # [x, y, w, h]
        print(f"📘 Loaded {len(out)} bbox entries from {path}")
        return out

    merged = {}
    for src in [train_json, val_json]:
        part = _load_one(src)
        merged.update(part)
    print(f"✅ Total merged bbox entries: {len(merged)}")
    return merged if merged else None

# ============================================================
# 3. Main pipeline
# ============================================================
class WarpDatasetPipeline:
    def __init__(self, args):
        self.args = args
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        self.bandwidth_scale = args.bw

        # --------------- paths ----------------
        if args.relight_type:
            self.input_root = os.path.join(args.input_root, args.target_prefix, args.relight_type)
        else:
            self.input_root = os.path.join(args.input_root, args.target_prefix)

        mode_tag = "" if args.separable else "_nonsep"
        eye_tag = "_eyes" if args.include_eyes else ""
        self.output_root = os.path.join(
            args.input_root,
            f"{args.target_prefix}_warped_{self.bandwidth_scale}{eye_tag}{mode_tag}"
        )
        if args.relight_type:
            self.output_root = os.path.join(self.output_root, args.relight_type)
        os.makedirs(self.output_root, exist_ok=True)

        # --------------- bbox mode ----------------
        self.bbox_map = load_bbox_map(args.train_bbox_json, args.val_bbox_json)
        if self.bbox_map is not None:
            self.face_app = None
            print("✅ Using merged GT bounding boxes (train + val). Face detector disabled.")
        else:
            self.bbox_map = None
            self.face_app = get_face_app(local_rank=0)
            print("✅ Using face detector (no bbox-jsons provided).")

        print(f"📂 input_root  = {self.input_root}")
        print(f"📂 output_root = {self.output_root}")

    # --------------------------------------------------------
    def get_gt_bbox(self, base, img_pil):
        """Return bbox tensor based on GT map, or full image if missing."""
        if base in self.bbox_map:
            valid = []
            for (x, y, w, h) in self.bbox_map[base]:
                if w <= 1 or h <= 1 or any(v != v for v in (x, y, w, h)):
                    continue
                valid.append([x, y, x + w, y + h])
            if not valid:  # fallback: full image -> No effect on warping since the grid will be identity
                w_img, h_img = img_pil.size
                bbox = torch.tensor([[0.0, 0.0, float(w_img), float(h_img)]], device=self.device)
            else:
                bbox = torch.tensor(valid, device=self.device)
        else:
            # no GT at all → full image
            w_img, h_img = img_pil.size
            bbox = torch.tensor([[0.0, 0.0, float(w_img), float(h_img)]], device=self.device)
        return bbox

    # --------------------------------------------------------
    def visualize_bbox(self, img_pil, bbox, base_name):
        """Draw bbox rectangles and save for debugging."""
        try:
            debug_dir = os.path.join(self.output_root, "_debug_bbox")
            os.makedirs(debug_dir, exist_ok=True)

            img_cv = np.array(img_pil)[:, :, ::-1].copy()
            for box in bbox:
                x1, y1, x2, y2 = map(int, box.tolist())
                cv2.rectangle(img_cv, (x1, y1), (x2, y2), (0, 255, 0), 2)

            out_path = os.path.join(debug_dir, base_name)
            cv2.imwrite(out_path, img_cv)
        except Exception as e:
            print(f"⚠️ bbox visualization failed for {base_name}: {e}")

    # --------------------------------------------------------
    def process_image(self, img_path, warped_dir):
        img_pil = Image.open(img_path).convert("RGB")
        c_t = load_img_warp(img_path).to(self.device)
        base = os.path.basename(img_path)

        # --- bbox source ---
        if self.bbox_map is not None:  # GT-only mode (no detector)
            bbox = self.get_gt_bbox(base, img_pil)
        else:
            # detector mode
            bbox = detect_face_bbox(img_pil, self.face_app, include_eyes=self.args.include_eyes).to(self.device)

        # --- visualize bbox ---
        self.visualize_bbox(img_pil, bbox, base)

        # --- warp images and compute inverse grid ---
        warped_img, warp_grid = apply_forward_warp(
            c_t, bbox, self.bandwidth_scale, separable=self.args.separable)
        _, _, h_img, w_img = c_t.shape
        inverse_grid = invert_grid(warp_grid, (1, 3, h_img, w_img), separable=self.args.separable)

        # --- save images and grids ---
        base_noext, ext = os.path.splitext(base)
        warped_path = os.path.join(warped_dir, f"{base_noext}{ext}")
        inv_grid_path = os.path.join(warped_dir, f"{base_noext}.inv.pth")
        save_img_warp(warped_img, warped_path)
        torch.save(inverse_grid.cpu(), inv_grid_path)

    # --------------------------------------------------------
    def process_and_warp_folder(self, sub_folder):
        src_dir = os.path.join(self.input_root, sub_folder)
        if not os.path.isdir(src_dir):
            return
        dst_dir = os.path.join(self.output_root, sub_folder)
        os.makedirs(dst_dir, exist_ok=True)

        imgs = sorted([f for f in os.listdir(src_dir)
                       if f.lower().endswith((".png", ".jpg", ".jpeg"))])
        for fname in tqdm(imgs, desc=f"Warping {sub_folder}", ncols=80):
            self.process_image(os.path.join(src_dir, fname), dst_dir)

    def copy_prompts(self):
        for fname in os.listdir(self.input_root):
            if fname.endswith((".json", ".txt")):
                shutil.copy2(
                    os.path.join(self.input_root, fname),
                    os.path.join(self.output_root, fname)
                )

    # --------------------------------------------------------
    def run(self):
        all_subfolders = ["train_A", "train_B", "test_A", "test_B"]
        warp_set = set(self.args.warp_subfolders)

        for sub in all_subfolders:
            if sub in warp_set:
                self.process_and_warp_folder(sub)
            else:
                # copy-only (non-warped)
                src_dir = os.path.join(self.input_root, sub)
                dst_dir = os.path.join(self.output_root, sub)
                if os.path.isdir(src_dir):
                    os.makedirs(dst_dir, exist_ok=True)
                    for fname in os.listdir(src_dir):
                        src_path = os.path.join(src_dir, fname)
                        dst_path = os.path.join(dst_dir, fname)
                        if os.path.isfile(src_path):
                            shutil.copy2(src_path, dst_path)
        self.copy_prompts()
        print(f"✅ Warp done for {sorted(warp_set)}; others copied only.")


# ============================================================
def main():
    args = parse_args()
    WarpDatasetPipeline(args).run()


if __name__ == "__main__":
    main()