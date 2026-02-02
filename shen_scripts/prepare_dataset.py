import os
import shutil
from pathlib import Path

IMG_EXTS = {".jpg", ".jpeg", ".png"}

def copy_images_recursive(src_dir, dst_dir):
    src_dir = Path(src_dir)
    dst_dir = Path(dst_dir)
    dst_dir.mkdir(parents=True, exist_ok=True)

    for path in src_dir.rglob("*"):
        if path.suffix.lower() in IMG_EXTS:
            dst_path = dst_dir / path.name

            # avoid overwrite collisions
            if dst_path.exists():
                stem = path.stem
                suffix = path.suffix
                i = 1
                while dst_path.exists():
                    dst_path = dst_dir / f"{stem}_{i}{suffix}"
                    i += 1

            shutil.copy2(path, dst_path)

def main(
    source_root,
    target_root,
    output_root,
):
    source_root = Path(source_root)
    target_root = Path(target_root)
    output_root = Path(output_root)

    # Source → A
    copy_images_recursive(
        source_root / "train",
        output_root / "train_A",
    )
    copy_images_recursive(
        source_root / "test",
        output_root / "test_A",
    )

    # Target → B
    copy_images_recursive(
        target_root / "train",
        output_root / "train_B",
    )
    copy_images_recursive(
        target_root / "test",
        output_root / "test_B",
    )

    print("✅ Copy finished.")

if __name__ == "__main__":
    main(
        source_root="/ssd0/shenzhen/Datasets/driving/cityscapes/leftImg8bit",
        target_root="/ssd0/shenzhen/Datasets/driving/dark_zurich/rgb_anon",
        output_root="/ssd0/shenzhen/Datasets/driving/cityscapes_to_dark_zurich",
    )