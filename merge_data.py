import os
import shutil
import json
from PIL import Image
from tqdm import tqdm
import argparse  # <-- added


def collect_pngs(folder):
    return sorted([os.path.join(folder, f) for f in os.listdir(folder) if f.endswith(".png")])

def merge_split(split_name, input_dirs, out_base, out_relight, prompt, start_idx):
    idx = start_idx
    for d in input_dirs:
        base_dir = os.path.join(d, split_name + "_A")
        relight_dir = os.path.join(d, split_name + "_B")
        if not (os.path.isdir(base_dir) and os.path.isdir(relight_dir)):
            continue

        base_files = collect_pngs(base_dir)
        relight_files = collect_pngs(relight_dir)

        for b, r in tqdm(zip(base_files, relight_files), total=len(base_files)):
            fname = f"{idx}.png"
            shutil.copyfile(b, os.path.join(out_base, fname))
            shutil.copyfile(r, os.path.join(out_relight, fname))
            idx += 1
    return idx

def parse_args():  # <-- added
    parser = argparse.ArgumentParser()
    parser.add_argument("--input_root", type=str, required=True)
    parser.add_argument("--target_prefix", type=str, required=True)
    parser.add_argument("--target_prefix_2", type=str, required=True)
    parser.add_argument("--relight_type", type=str, required=True)
    parser.add_argument("--merged_prefix", type=str, required=True)  # <-- added
    return parser.parse_args()

def main():
    args = parse_args()  # <-- added

    # ---- minimal change: build input_dirs using prefix1 and prefix2 ----
    input_dirs = [
        os.path.join(args.input_root, args.target_prefix, args.relight_type),
        os.path.join(args.input_root, args.target_prefix_2, args.relight_type),
    ]

    # ---- minimal change: auto-guess merged folder name ----
    merged_name = args.merged_prefix
    out_root = os.path.join(args.input_root, merged_name, args.relight_type)

    os.makedirs(out_root, exist_ok=True)

    out_train_A = os.path.join(out_root, "train_A")
    out_train_B = os.path.join(out_root, "train_B")
    out_test_A = os.path.join(out_root, "test_A")
    out_test_B = os.path.join(out_root, "test_B")
    for d in [out_train_A, out_train_B, out_test_A, out_test_B]:
        os.makedirs(d, exist_ok=True)

    # assume prompt is identical across datasets
    with open(os.path.join(input_dirs[0], "train_prompts.json"), "r") as f:
        prompt_json = json.load(f)
    prompt = list(prompt_json.values())[0]

    idx = 0
    idx = merge_split("train", input_dirs, out_train_A, out_train_B, prompt, idx)
    train_count = idx

    idx = merge_split("test", input_dirs, out_test_A, out_test_B, prompt, idx)
    total_count = idx

    train_json = {f"{i}.png": prompt for i in range(train_count)}
    test_json = {f"{i}.png": prompt for i in range(train_count, total_count)}

    with open(os.path.join(out_root, "train_prompts.json"), "w") as f:
        json.dump(train_json, f, indent=4)

    with open(os.path.join(out_root, "test_prompts.json"), "w") as f:
        json.dump(test_json, f, indent=4)

if __name__ == "__main__":
    main()