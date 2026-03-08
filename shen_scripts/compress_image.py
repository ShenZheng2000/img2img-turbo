from PIL import Image, ImageFilter
import os

input_paths = [
    "/home/shenzhen/Datasets/VITON/test/image/00055_00.jpg",
    "/home/shenzhen/Relight_Projects/img2img-turbo/output/pix2pix_turbo/exp_1_10_1_warped_128_eyes/golden_sunlight_1/VITON/test/image/00055_00_warp.png",
    "/home/shenzhen/Relight_Projects/img2img-turbo/output/pix2pix_turbo/exp_1_10_1_warped_128_eyes/golden_sunlight_1/VITON/test/image/00055_00_warp_relight.png",
    "/home/shenzhen/Relight_Projects/img2img-turbo/output/pix2pix_turbo/exp_1_10_1/golden_sunlight_1/VITON/test/image/00055_00_warp_relight_unwarp.png"
]

output_dir = "output_images"
os.makedirs(output_dir, exist_ok=True)

for in_path in input_paths:
    if not os.path.exists(in_path):
        print(f"Skipping: {in_path} (not found)")
        continue

    fname = os.path.splitext(os.path.basename(in_path))[0] + "_feat.png"
    out_path = os.path.join(output_dir, fname)

    with Image.open(in_path).convert("RGB") as img:
        w, h = img.size
        # 1. Resize down (Compression)
        img_small = img.resize((w // 4, h // 4), Image.LANCZOS)

        # 2. Smooth textures (The "Feature" look)
        # MedianFilter(3) or (5) removes small details like eyes/textures 
        # while keeping the shape of the body/clothes intact.
        img_smoothed = img_small.filter(ImageFilter.MedianFilter(size=3))

        # 3. Color Quantization
        # Using MAXCOVERAGE or MEDIANCUT with a low color count (8-12)
        img_feat = img_smoothed.quantize(colors=12, method=Image.MAXCOVERAGE).convert("RGB")

        img_feat.save(out_path, format="PNG")

print("Done! Feature-style images saved to:", output_dir)