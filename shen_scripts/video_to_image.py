import os
import cv2
import argparse
from pathlib import Path
from tqdm import tqdm


'''
Example runs (--debug_mode optional):
python /home/shenzhen/Relight_Projects/img2img-turbo/shen_scripts/video_to_image.py \
  --input_dir /ssd0/shenzhen/Datasets/ROADWork/videos_compressed \
  --output_dir /ssd0/shenzhen/Datasets/ROADWork/extracted_frames \
  --prefix boston_

python /home/shenzhen/Relight_Projects/img2img-turbo/shen_scripts/video_to_image.py \
  --input_dir /scratch/shenzhen/Datasets/bdd100k/videos/test \
  --output_dir /scratch/shenzhen/Datasets/bdd100k/extracted_frames \
  --max_videos 50
'''


def parse_args():
    parser = argparse.ArgumentParser(
        description="Extract frames from all videos in a folder into image subfolders."
    )
    parser.add_argument(
        "--input_dir",
        type=str,
        required=True,
        help="Folder containing input videos, e.g. videos_compressed/",
    )
    parser.add_argument(
        "--output_dir",
        type=str,
        required=True,
        help="Output root folder, e.g. extracted_frames/",
    )
    parser.add_argument(
        "--ext",
        type=str,
        default=".jpg", # NOTE: use jpg to save space!!!
        choices=[".png", ".jpg", ".jpeg"],
        help="Image extension for extracted frames.",
    )
    parser.add_argument(
        "--video_exts",
        type=str,
        nargs="+",
        default=[".mp4", ".mov", ".avi", ".mkv"],
        help="Video extensions to search for.",
    )
    parser.add_argument(
        "--skip_existing",
        action="store_true",
        help="Skip a video if its output image folder already contains frames.",
    )
    parser.add_argument(
        "--every_n_frames",
        type=int,
        default=1,
        help="Keep one frame every N frames. Default=1 means keep all frames.",
    )
    parser.add_argument(
        "--debug_mode",
        action="store_true",
        help="Process only the first video for debugging."
    )    
    parser.add_argument(
        "--prefix",
        type=str,
        default=None,
        help="Optional filename prefix filter (e.g., 'boston_'). Only videos starting with this prefix will be processed."
    )
    parser.add_argument(
        "--max_videos",
        type=int,
        default=None,
        help="Optional: process only the first N videos."
    )    
    return parser.parse_args()


def get_video_list(input_dir: Path, video_exts):
    video_exts = {e.lower() for e in video_exts}
    return sorted([
        p for p in input_dir.iterdir()
        if p.is_file() and p.suffix.lower() in video_exts
    ])


def extract_one_video(video_path: Path, output_root: Path, img_ext: str, every_n_frames: int, skip_existing: bool):
    video_stem = video_path.stem
    out_dir = output_root / video_stem / "image"
    out_dir.mkdir(parents=True, exist_ok=True)

    existing_frames = list(out_dir.glob(f"*{img_ext}"))
    if skip_existing and len(existing_frames) > 0:
        print(f"⏭ Skipping {video_path.name}: found {len(existing_frames)} existing frames in {out_dir}")
        return

    cap = cv2.VideoCapture(str(video_path))
    if not cap.isOpened():
        print(f"❌ Failed to open video: {video_path}")
        return

    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    fps = cap.get(cv2.CAP_PROP_FPS)

    print(f"\n📹 Processing: {video_path.name}")
    print(f"   FPS: {fps:.3f}" if fps > 0 else "   FPS: unknown")
    print(f"   Total frames reported: {total_frames}")

    frame_idx = 0
    saved_idx = 0

    pbar_total = total_frames if total_frames > 0 else None
    with tqdm(total=pbar_total, desc=video_path.name, leave=False) as pbar:
        while True:
            ret, frame = cap.read()
            if not ret:
                break

            if frame_idx % every_n_frames == 0:
                # OpenCV reads BGR, but cv2.imwrite expects BGR too, so no conversion needed
                out_name = f"{saved_idx:06d}{img_ext}"
                out_path = out_dir / out_name
                cv2.imwrite(str(out_path), frame)
                saved_idx += 1

            frame_idx += 1
            pbar.update(1)

    cap.release()
    print(f"✅ Saved {saved_idx} frames to {out_dir}")


def main():
    args = parse_args()

    input_dir = Path(args.input_dir)
    output_dir = Path(args.output_dir)

    if not input_dir.exists() or not input_dir.is_dir():
        raise ValueError(f"Input directory does not exist or is not a directory: {input_dir}")

    output_dir.mkdir(parents=True, exist_ok=True)

    video_list = get_video_list(input_dir, args.video_exts)
    if args.prefix is not None:
        video_list = [v for v in video_list if v.name.startswith(args.prefix)]

    if args.max_videos is not None:
        video_list = video_list[:args.max_videos]

    if args.debug_mode:
        video_list = video_list[:1]

    if len(video_list) == 0:
        raise RuntimeError(f"No videos found in {input_dir} with extensions {args.video_exts}")

    print(f"📁 Found {len(video_list)} videos in {input_dir}")
    print(f"📁 Output root: {output_dir}")

    for video_path in tqdm(video_list, desc="All videos"):
        extract_one_video(
            video_path=video_path,
            output_root=output_dir,
            img_ext=args.ext,
            every_n_frames=args.every_n_frames,
            skip_existing=args.skip_existing,
        )

    print("\n🎉 Done.")


if __name__ == "__main__":
    main()