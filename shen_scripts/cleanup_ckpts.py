#!/usr/bin/env python3
import argparse
import os
import re
from pathlib import Path

'''
Example usage (Use --dry-run first to preview what would be deleted):
python shen_scripts/cleanup_ckpts.py output/cyclegan_turbo/bdd100k_1_20_resize_286_randomcrop_256x256_hflip
'''


CKPT_PATTERN = re.compile(r"^model_(\d+)\.pkl$")


def find_checkpoint_dirs(root: Path):
    for dirpath, dirnames, filenames in os.walk(root):
        if Path(dirpath).name == "checkpoints":
            yield Path(dirpath)


def get_checkpoint_files(ckpt_dir: Path):
    matches = []
    for p in ckpt_dir.iterdir():
        if not p.is_file():
            continue
        m = CKPT_PATTERN.match(p.name)
        if m:
            step = int(m.group(1))
            matches.append((step, p))
    return matches


def process_checkpoints_dir(ckpt_dir: Path, dry_run: bool):
    ckpts = get_checkpoint_files(ckpt_dir)

    if not ckpts:
        print(f"[SKIP] {ckpt_dir} (no matching model_*.pkl files)")
        return

    ckpts.sort(key=lambda x: x[0])
    keep_step, keep_file = ckpts[-1]
    delete_files = [p for _, p in ckpts[:-1]]

    print(f"\n[DIR] {ckpt_dir}")

    if delete_files:
        print("[DELETE]")
        for p in delete_files:
            print(f"  - {p.name}")
    else:
        print("[INFO] Nothing to delete.")

    if not dry_run and delete_files:
        for p in delete_files:
            p.unlink()
        print(f"[DONE] Deleted {len(delete_files)} file(s).")

    # Print the kept file at the end
    print(f"[KEEP] {keep_file.name} (step={keep_step})")


def main():
    parser = argparse.ArgumentParser(
        description=(
            'Recursively find folders named "checkpoints", keep only the '
            "largest-numbered model_*.pkl file in each, and delete the rest."
        )
    )
    parser.add_argument(
        "root",
        nargs="?",
        default=".",
        help="Root directory to search from (default: current directory)",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Preview what would be kept/deleted without deleting anything",
    )
    args = parser.parse_args()

    root = Path(args.root).resolve()
    found_any = False

    for ckpt_dir in find_checkpoint_dirs(root):
        found_any = True
        process_checkpoints_dir(ckpt_dir, dry_run=args.dry_run)

    if not found_any:
        print(f'No folders named "checkpoints" found under: {root}')


if __name__ == "__main__":
    main()