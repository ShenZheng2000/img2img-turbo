# Input: /scratch/shenzhen/Datasets/ROADWork/extracted_frames/{video_id}/image/*.jpg
# video_id
    # boston_2d8e13b1a8304d8395dcf6479ca61814_000006_01530_snippet
    # boston_2d8e13b1a8304d8395dcf6479ca61814_000008_09810_snippet
    # boston_2d8e13b1a8304d8395dcf6479ca61814_000010_05160_snippet
# relight_type
    # golden_sunlight_1
    # foggy_1
# Output (IC-Light): /home/shenzhen/Relight_Projects/IC-Light/outputs/512x512/golden_sunlight_1/ROADWork/extracted_frames/{video_id}/*.png
# Output (Ours): /home/shenzhen/Relight_Projects/img2img-turbo/output/pix2pix_turbo/2_24_drive_v2_warped_128/{relight_type}/ROADWork/extracted_frames/{video_id}/image/*_warp_relight_unwarp.png

import os
from PIL import Image

# ============================================================
# Paths
# ============================================================

INPUT_ROOT = "/scratch/shenzhen/Datasets/ROADWork/extracted_frames"
ICLIGHT_ROOT = "/home/shenzhen/Relight_Projects/IC-Light/outputs/512x512"
OURS_ROOT = "/home/shenzhen/Relight_Projects/img2img-turbo/output/pix2pix_turbo/2_24_drive_v2_warped_128"
SAVE_ROOT = "/scratch/shenzhen/Datasets/concat_frames/ROADWork"
CROP_SIZE = 784


# ============================================================
# Config
# ============================================================

VIDEO_IDS = [
    # "boston_2d8e13b1a8304d8395dcf6479ca61814_000006_01530_snippet",
    # "boston_2d8e13b1a8304d8395dcf6479ca61814_000008_09810_snippet",
    "boston_2d8e13b1a8304d8395dcf6479ca61814_000010_05160_snippet",
]

RELIGHT_TYPES = [
    "golden_sunlight_1",
    "foggy_1",
]


def center_crop(img, size):
    w, h = img.size
    left = (w - size) // 2
    top = (h - size) // 2
    return img.crop((left, top, left + size, top + size))


# ============================================================
# Concat logic
# ============================================================

def concat_one_video(video_id, relight_type):

    src_dir = os.path.join(INPUT_ROOT, video_id, "image")

    iclight_dir = os.path.join(
        ICLIGHT_ROOT,
        relight_type,
        "ROADWork",
        "extracted_frames",
        video_id,
    )

    ours_dir = os.path.join(
        OURS_ROOT,
        relight_type,
        "ROADWork",
        "extracted_frames",
        video_id,
        "image",
    )

    out_dir = os.path.join(
        SAVE_ROOT,
        relight_type,
        video_id,
    )

    os.makedirs(out_dir, exist_ok=True)

    if not os.path.isdir(src_dir):
        print(f"[Skip] missing src_dir: {src_dir}")
        return

    if not os.path.isdir(iclight_dir):
        print(f"[Skip] missing iclight_dir: {iclight_dir}")
        return

    if not os.path.isdir(ours_dir):
        print(f"[Skip] missing ours_dir: {ours_dir}")
        return

    files = sorted(f for f in os.listdir(src_dir) if f.endswith(".jpg"))

    print(f"[Start] {relight_type} | {video_id} | {len(files)} frames")

    for f in files:
        stem = os.path.splitext(f)[0]

        src_path = os.path.join(src_dir, f)
        iclight_path = os.path.join(iclight_dir, stem + ".png")
        ours_path = os.path.join(ours_dir, stem + "_warp_relight_unwarp.png")
        # out_path = os.path.join(out_dir, stem + ".png")
        out_path = os.path.join(out_dir, stem + ".jpg")

        if not os.path.exists(iclight_path):
            print(f"[Warn] missing iclight: {iclight_path}")
            continue

        if not os.path.exists(ours_path):
            print(f"[Warn] missing ours: {ours_path}")
            continue

        img_input = Image.open(src_path).convert("RGB")

        # NOTE: add center crop here
        if CROP_SIZE is not None:
            img_input = center_crop(img_input, CROP_SIZE)

        img_iclight = Image.open(iclight_path).convert("RGB")
        img_ours = Image.open(ours_path).convert("RGB")

        w, h = img_input.size

        if img_iclight.size != (w, h):
            img_iclight = img_iclight.resize((w, h), Image.LANCZOS)

        if img_ours.size != (w, h):
            img_ours = img_ours.resize((w, h), Image.LANCZOS)

        canvas = Image.new("RGB", (w * 3, h))
        canvas.paste(img_input, (0, 0))
        canvas.paste(img_iclight, (w, 0))
        canvas.paste(img_ours, (2 * w, 0))
        # canvas.save(out_path)
        canvas.save(out_path, "JPEG", quality=95, subsampling=0)

    print(f"[Done] {relight_type} | {video_id}")


# ============================================================
# Main
# ============================================================

def main():

    for relight_type in RELIGHT_TYPES:
        for video_id in VIDEO_IDS:
            concat_one_video(video_id, relight_type)

    print("All done.")


if __name__ == "__main__":
    main()