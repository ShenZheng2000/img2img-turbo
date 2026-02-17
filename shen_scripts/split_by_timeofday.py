#!/usr/bin/env python3
"""
Copy BDD-style images into folders by timeofday, using per-image JSON labels.

Assumes:
  images/{train,val,test}/<name>.jpg
  labels/{train,val,test}/<name>.json
and JSON contains: data["attributes"]["timeofday"]  (e.g., daytime/night/dawn/dusk)

Example:
  python split_by_timeofday.py \
    --root /ssd0/shenzhen/Datasets/driving/BDD100K/100k \
    --split train val \
    --out /ssd0/shenzhen/Datasets/driving/BDD100K/100k/by_timeofday \
    --dry_run
"""

import argparse
import json
import shutil
from pathlib import Path

IMG_EXTS = [".jpg", ".jpeg", ".png"]

def find_image(images_dir: Path, stem: str) -> Path | None:
    for ext in IMG_EXTS:
        p = images_dir / f"{stem}{ext}"
        if p.exists():
            return p
    return None


def get_timeofday(label_json: dict) -> str:
    tod = (label_json.get("attributes") or {}).get("timeofday", "unknown")
    return str(tod).strip().lower().replace(" ", "_")


def resolve_paths(root: Path, split: str) -> tuple[Path, Path]:
    labels_dir = root / "labels" / split
    images_dir = root / "images" / split
    return labels_dir, images_dir


def copy_one(label_path: Path, images_dir: Path, out_root: Path, split: str, dry_run: bool) -> str | None:
    # load label
    try:
        data = json.loads(label_path.read_text())
    except Exception as e:
        print(f"[WARN] bad json: {label_path} ({e})")
        return None

    stem = data.get("name") or label_path.stem
    tod = get_timeofday(data)

    # find matching image (same basename)
    img_path = find_image(images_dir, stem) or find_image(images_dir, label_path.stem)
    if img_path is None:
        print(f"[WARN] no image for: {label_path.name}")
        return None

    # copy
    # dst_dir = out_root / tod / split
    dst_dir = out_root / f"{split}_{tod}"
    dst_dir.mkdir(parents=True, exist_ok=True)
    dst_path = dst_dir / img_path.name

    if dry_run:
        print(f"[DRY] {img_path} -> {dst_path}")
    else:
        shutil.copy2(img_path, dst_path)

    return tod

def parse_args():
    ap = argparse.ArgumentParser()
    ap.add_argument("--root", type=Path, required=True,
                    help="Dataset root containing images/ and labels/")
    ap.add_argument("--out", type=Path, required=True,
                    help="Output root; will create subfolders per timeofday")
    ap.add_argument("--split", nargs="+", default=["train", "val", "test"],
                    help="Which splits to process")
    ap.add_argument("--dry_run", action="store_true",
                    help="Print actions without copying")
    return ap.parse_args()

def main():
    args = parse_args()
    root: Path = args.root
    out_root: Path = args.out
    out_root.mkdir(parents=True, exist_ok=True)

    stats: dict[str, int] = {}

    for split in args.split:
        labels_dir, images_dir = resolve_paths(root, split)

        if not labels_dir.exists():
            print(f"[WARN] missing labels dir: {labels_dir}")
            continue
        if not images_dir.exists():
            print(f"[WARN] missing images dir: {images_dir}")
            continue

        for label_path in labels_dir.glob("*.json"):
            tod = copy_one(label_path, images_dir, out_root, split, args.dry_run)
            if tod is not None:
                stats[tod] = stats.get(tod, 0) + 1

    print("\n[Done] counts by timeofday:")
    for tod in sorted(stats):
        print(f"  {tod}: {stats[tod]}")


if __name__ == "__main__":
    main()