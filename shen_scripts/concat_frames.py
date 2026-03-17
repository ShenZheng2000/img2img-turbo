# NOTE: concat [Input | Ours] only for better visual comparison,

import os
from PIL import Image

DATASET = "bdd100k"
RESULT_SUBDIR = f"result_B_{DATASET}"
INPUT_ROOT = f"/scratch/shenzhen/Datasets/{DATASET}/extracted_frames"
OUTPUT_ROOT = "/home/shenzhen/Relight_Projects/img2img-turbo/output/cyclegan_turbo"
SAVE_ROOT = f"/scratch/shenzhen/Datasets/concat_frames/{DATASET}"

# TODO: day2night later
JOBS = [
    {
        "visual_task": "rainy2clear",
        "model_task": "clear2rainy",
        "video_ids": [
            "cd4deee2-1d9539bd",
            "cd4deee2-9c9f6da1",
            "cd4deee2-688c8bba",
            "cd4deee2-0703d1c7",
        ],
    },
    # TODO later
    {
        "visual_task": "night2day",
        # what the model folder is actually called on disk
        "model_task": "day2night",
        "video_ids": [
            "cd3b1173-63cb9e2e",
            "cd4ac25c-61a9eb11",
            "cd4da443-da4fe8c7",
            "cd40cb21-18170d03",
        ],
    },
]


def concat_one_video(visual_task, model_task, video_id):
    src_dir = os.path.join(INPUT_ROOT, video_id, "image")

    warp_dir = os.path.join(
        OUTPUT_ROOT,
        f"{DATASET.upper()}_{model_task}_warped_128_resize_286_randomcrop_256x256_hflip",
        RESULT_SUBDIR,
        "extracted_frames",
        video_id,
        "image",
    )

    out_dir = os.path.join(
        SAVE_ROOT,
        visual_task,
        RESULT_SUBDIR,
        "extracted_frames",
        video_id,
    )
    os.makedirs(out_dir, exist_ok=True)

    if not os.path.isdir(src_dir):
        print(f"[Skip] missing src_dir: {src_dir}")
        return
    if not os.path.isdir(warp_dir):
        print(f"[Skip] missing warp_dir: {warp_dir}")
        return

    files = sorted(f for f in os.listdir(src_dir) if f.endswith(".jpg"))
    print(f"[Start] {visual_task} | {video_id} | {len(files)} frames")

    for f in files:
        stem = os.path.splitext(f)[0]

        src_path = os.path.join(src_dir, f)
        warp_path = os.path.join(warp_dir, stem + "_warp_relight_unwarp.png")
        out_path = os.path.join(out_dir, stem + ".jpg")

        if not os.path.exists(warp_path):
            print(f"[Warn] missing warp: {warp_path}")
            continue

        img_input = Image.open(src_path).convert("RGB")
        img_warp = Image.open(warp_path).convert("RGB")

        # force same size
        w, h = img_input.size
        if img_warp.size != (w, h):
            img_warp = img_warp.resize((w, h), Image.LANCZOS)

        # 2-column canvas
        canvas = Image.new("RGB", (w * 2, h))
        canvas.paste(img_input, (0, 0))
        canvas.paste(img_warp, (w, 0))

        canvas.save(out_path, "JPEG", quality=95, subsampling=0)

    print(f"[Done] {visual_task} | {video_id}")


def main():
    for job in JOBS:
        visual_task = job["visual_task"]
        model_task = job["model_task"]
        for video_id in job["video_ids"]:
            concat_one_video(visual_task, model_task, video_id)

    print("All done.")


if __name__ == "__main__":
    main()



# import os
# from PIL import Image

# DATASET = "bdd100k"
# RESULT_SUBDIR = f"result_B_{DATASET}"
# INPUT_ROOT = f"/scratch/shenzhen/Datasets/{DATASET}/extracted_frames"
# OUTPUT_ROOT = "/home/shenzhen/Relight_Projects/img2img-turbo/output/cyclegan_turbo"
# SAVE_ROOT = f"/scratch/shenzhen/Datasets/concat_frames/{DATASET}"

# JOBS = [
#     # {
#     #     # what you want to call this comparison task
#     #     "visual_task": "night2day",
#     #     # what the model folder is actually called on disk
#     #     "model_task": "day2night",
#     #     "video_ids": [
#     #         "cd3b1173-63cb9e2e",
#     #         "cd4ac25c-61a9eb11",
#     #         "cd4da443-da4fe8c7",
#     #         "cd40cb21-18170d03",
#     #     ],
#     # },
#     {
#         "visual_task": "rainy2clear",
#         "model_task": "clear2rainy",
#         "video_ids": [
#             "cd4deee2-1d9539bd",
#             "cd4deee2-9c9f6da1",
#             "cd4deee2-688c8bba",
#             "cd4deee2-0703d1c7",
#         ],
#     },
# ]


# def concat_one_video(visual_task, model_task, video_id):
#     src_dir = os.path.join(INPUT_ROOT, video_id, "image")

#     nowarp_dir = os.path.join(
#         OUTPUT_ROOT,
#         f"pretrained_{model_task}",
#         RESULT_SUBDIR,
#         "extracted_frames",
#         video_id,
#         "image",
#     )

#     warp_dir = os.path.join(
#         OUTPUT_ROOT,
#         f"{DATASET.upper()}_{model_task}_warped_128_resize_286_randomcrop_256x256_hflip",
#         RESULT_SUBDIR,
#         "extracted_frames",
#         video_id,
#         "image",
#     )

#     out_dir = os.path.join(
#         SAVE_ROOT,
#         visual_task,
#         RESULT_SUBDIR,
#         "extracted_frames",
#         video_id,
#     )
#     os.makedirs(out_dir, exist_ok=True)

#     if not os.path.isdir(src_dir):
#         print(f"[Skip] missing src_dir: {src_dir}")
#         return
#     if not os.path.isdir(nowarp_dir):
#         print(f"[Skip] missing nowarp_dir: {nowarp_dir}")
#         return
#     if not os.path.isdir(warp_dir):
#         print(f"[Skip] missing warp_dir: {warp_dir}")
#         return

#     files = sorted(f for f in os.listdir(src_dir) if f.endswith(".jpg"))
#     print(f"[Start] {visual_task} | {video_id} | {len(files)} frames")

#     for f in files:
#         stem = os.path.splitext(f)[0]

#         src_path = os.path.join(src_dir, f)
#         nowarp_path = os.path.join(nowarp_dir, stem + "_warp_relight_unwarp.png")
#         warp_path = os.path.join(warp_dir, stem + "_warp_relight_unwarp.png")
#         # out_path = os.path.join(out_dir, stem + ".png")
#         out_path = os.path.join(out_dir, stem + ".jpg")

#         if not os.path.exists(nowarp_path):
#             print(f"[Warn] missing nowarp: {nowarp_path}")
#             continue
#         if not os.path.exists(warp_path):
#             print(f"[Warn] missing warp: {warp_path}")
#             continue

#         img_input = Image.open(src_path).convert("RGB")
#         img_nowarp = Image.open(nowarp_path).convert("RGB")
#         img_warp = Image.open(warp_path).convert("RGB")

#         # force same size as input, in case generated images differ
#         w, h = img_input.size
#         if img_nowarp.size != (w, h):
#             img_nowarp = img_nowarp.resize((w, h), Image.LANCZOS)
#         if img_warp.size != (w, h):
#             img_warp = img_warp.resize((w, h), Image.LANCZOS)

#         canvas = Image.new("RGB", (w * 3, h))
#         canvas.paste(img_input, (0, 0))
#         canvas.paste(img_nowarp, (w, 0))
#         canvas.paste(img_warp, (2 * w, 0))
#         # canvas.save(out_path)
#         canvas.save(out_path, "JPEG", quality=95, subsampling=0)

#     print(f"[Done] {visual_task} | {video_id}")


# def main():
#     for job in JOBS:
#         visual_task = job["visual_task"]
#         model_task = job["model_task"]
#         for video_id in job["video_ids"]:
#             concat_one_video(visual_task, model_task, video_id)

#     print("All done.")


# if __name__ == "__main__":
#     main()