import os
from pathlib import Path
from PIL import Image
from tqdm import tqdm
import argparse

'''
Example usage: 
python compare_relighting_versions.py \
  --orig_dir /home/shenzhen/Datasets/dataset_with_garment_bigface_100 \
  --no_warp_dir /home/shenzhen/Relight_Projects/img2img-turbo/output/pix2pix_turbo/exp_10_16/noon_sunlight_1/result_A_dataset_with_garment_bigface_100 \
  --warp_face_dir /home/shenzhen/Relight_Projects/img2img-turbo/output/pix2pix_turbo/exp_10_16_warped_128/noon_sunlight_1/result_A_dataset_with_garment_bigface_100 \
  --warp_eyes_dir /home/shenzhen/Relight_Projects/img2img-turbo/output/pix2pix_turbo/exp_10_16_warped_128_eyes/noon_sunlight_1/result_A_dataset_with_garment_bigface_100 \
  --output_dir /home/shenzhen/Relight_Projects/img2img-turbo/comparisons/exp_10_16/noon_sunlight_1
'''

# ================================================
# Helper: resize all to same height, then concat
# ================================================
def resize_keep_aspect(img, target_h):
    w, h = img.size
    new_w = int(w * (target_h / h))
    return img.resize((new_w, target_h), Image.LANCZOS)

def concat_images_horizontally(images, target_height=None):
    """Combine multiple PIL images horizontally."""
    imgs_resized = [resize_keep_aspect(img, target_height) for img in images]
    widths = [i.size[0] for i in imgs_resized]
    total_width = sum(widths)
    combined = Image.new("RGB", (total_width, target_height))
    x_offset = 0
    for img in imgs_resized:
        combined.paste(img, (x_offset, 0))
        x_offset += img.width
    return combined


# ================================================
# Main logic
# ================================================
def compare_images(base_original, base_no_warp, base_warp_face, base_warp_eyes, output_dir):
    base_original = Path(base_original)
    base_no_warp = Path(base_no_warp)
    base_warp_face = Path(base_warp_face)
    base_warp_eyes = Path(base_warp_eyes)
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    folders = sorted([f for f in base_original.iterdir() if f.is_dir()])
    for folder in tqdm(folders, desc="Building comparisons"):
        folder_name = folder.name

        # Paths for the same folder in each version
        orig_path = next(folder.glob("bdy_*.*"), None)
        if orig_path is None:
            continue

        relit_paths = [
            base_no_warp / folder_name / "bdy_*_warp_relight_unwarp.png",
            base_warp_face / folder_name / "bdy_*_warp_relight_unwarp.png",
            base_warp_eyes / folder_name / "bdy_*_warp_relight_unwarp.png",
        ]

        # Resolve glob results
        resolved_relits = []
        for p in relit_paths:
            matches = list(p.parent.glob(p.name))
            resolved_relits.append(matches[0] if matches else None)

        # Skip if any missing
        if not all(resolved_relits):
            print(f"⚠️ Missing some results for {folder_name}")
            continue

        try:
            imgs = [
                Image.open(orig_path).convert("RGB"),
                Image.open(resolved_relits[0]).convert("RGB"),
                Image.open(resolved_relits[1]).convert("RGB"),
                Image.open(resolved_relits[2]).convert("RGB"),
            ]
            combined = concat_images_horizontally(imgs, target_height=784)
            combined.save(output_dir / f"{folder_name}.png")
        except Exception as e:
            print(f"⚠️ Error in {folder_name}: {e}")


# ================================================
# Entry point
# ================================================
def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--orig_dir", required=True, help="Path to original dataset (bdy_* images)")
    parser.add_argument("--no_warp_dir", required=True, help="Path to no-warp relight results")
    parser.add_argument("--warp_face_dir", required=True, help="Path to warp-on-face results")
    parser.add_argument("--warp_eyes_dir", required=True, help="Path to warp-on-face+eyes results")
    parser.add_argument("--output_dir", required=True, help="Path to save concatenated comparisons")
    args = parser.parse_args()

    compare_images(
        args.orig_dir,
        args.no_warp_dir,
        args.warp_face_dir,
        args.warp_eyes_dir,
        args.output_dir,
    )

if __name__ == "__main__":
    main()