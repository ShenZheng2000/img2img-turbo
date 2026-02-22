import shutil
from pathlib import Path

NUM_SAMPLES = 100
SRC_ROOT = Path("/home/shenzhen/Datasets/VITON/test")
DST_ROOT = Path(f"/home/shenzhen/Datasets/VITON/test_sample_{NUM_SAMPLES}")

src_img_dir = SRC_ROOT / "image"
src_mask_dir = SRC_ROOT / "fg_masks"

dst_img_dir = DST_ROOT / "image"
dst_mask_dir = DST_ROOT / "fg_masks"

dst_img_dir.mkdir(parents=True, exist_ok=True)
dst_mask_dir.mkdir(parents=True, exist_ok=True)

# ---- numeric sort (1,2,10 instead of 1,10,2)
def numeric_key(p):
    return int(p.stem)

img_paths = sorted(
    [p for p in src_img_dir.glob("*.jpg")],
    key=numeric_key
)

img_paths = img_paths[:NUM_SAMPLES]

print(f"Copying {len(img_paths)} samples...")

for img_path in img_paths:
    mask_path = src_mask_dir / f"{img_path.stem}.png"

    if not mask_path.exists():
        print(f"⚠️ Missing mask for {img_path.name}, skipping")
        continue

    shutil.copy2(img_path, dst_img_dir / img_path.name)
    shutil.copy2(mask_path, dst_mask_dir / mask_path.name)

print("✅ Done")