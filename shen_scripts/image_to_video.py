import os
import cv2

ROOT = "/scratch/shenzhen/Datasets/concat_frames"
FPS = 30


def process_video_folder(parent_dir):

    for video_id in sorted(os.listdir(parent_dir)):
        img_dir = os.path.join(parent_dir, video_id)
        if not os.path.isdir(img_dir):
            continue

        files = sorted(
            f for f in os.listdir(img_dir)
            if f.lower().endswith((".jpg", ".png"))
        )
        if not files:
            continue

        first = cv2.imread(os.path.join(img_dir, files[0]))
        h, w = first.shape[:2]

        out_path = os.path.join(parent_dir, f"{video_id}.mp4")

        writer = cv2.VideoWriter(
            out_path,
            cv2.VideoWriter_fourcc(*"mp4v"),
            FPS,
            (w, h)
        )

        for f in files:
            img = cv2.imread(os.path.join(img_dir, f))
            if img is None:
                continue
            if img.shape[:2] != (h, w):
                img = cv2.resize(img, (w, h))
            writer.write(img)

        writer.release()
        print("[Done]", out_path)


# ============================================================
# ROADWork
# ============================================================

# TODO: rerun this lateer! 
# road_root = os.path.join(ROOT, "ROADWork")

# for relight_type in os.listdir(road_root):
#     relight_dir = os.path.join(road_root, relight_type)
#     if os.path.isdir(relight_dir):
#         process_video_folder(relight_dir)


# ============================================================
# BDD100K
# ============================================================

bdd_root = os.path.join(ROOT, "bdd100k")

for task in os.listdir(bdd_root):

    img_root = os.path.join(
        bdd_root,
        task,
        "result_B_bdd100k",
        "extracted_frames"
    )

    if os.path.isdir(img_root):
        process_video_folder(img_root)

print("All done.")