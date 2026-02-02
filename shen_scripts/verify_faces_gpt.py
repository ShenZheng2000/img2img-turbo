# import base64
# import io
# from pathlib import Path
# from PIL import Image
# from tqdm import tqdm
# import argparse
# from openai import OpenAI

# NOTE: this result is terrible, skip for now.

# # ==============================================================
# # Utility: combine two images side-by-side (resize to same height)
# # ==============================================================
# def combine_side_by_side(img1_path: Path, img2_path: Path, target_height=512) -> Image.Image:
#     def resize_keep_aspect(img, target_h):
#         w, h = img.size
#         new_w = int(w * (target_h / h))
#         return img.resize((new_w, target_h), Image.LANCZOS)

#     img1 = Image.open(img1_path).convert("RGB")
#     img2 = Image.open(img2_path).convert("RGB")

#     img1 = resize_keep_aspect(img1, target_height)
#     img2 = resize_keep_aspect(img2, target_height)

#     w1, h1 = img1.size
#     w2, h2 = img2.size

#     combined = Image.new("RGB", (w1 + w2, target_height))
#     combined.paste(img1, (0, 0))
#     combined.paste(img2, (w1, 0))
#     return combined


# # ==============================================================
# # Utility: convert an Image to base64 string
# # ==============================================================
# def encode_image_base64(img: Image.Image) -> str:
#     buffered = io.BytesIO()
#     img.save(buffered, format="PNG")
#     return base64.b64encode(buffered.getvalue()).decode("utf-8")


# # ==============================================================
# # GPT query logic
# # ==============================================================
# def query_gpt_about_pair(client: OpenAI, orig_img: Path, relit_img: Path, prompt: str) -> str:
#     try:
#         combined = combine_side_by_side(orig_img, relit_img)

#         # save image to debug, and immediately exit
#         combined.save("debug_combined.png")
#         exit()

#         combined_b64 = encode_image_base64(combined)

#         response = client.chat.completions.create(
#             model="gpt-4o", # TODO: use gpt-5 if needed
#             temperature=0,
#             messages=[
#                 {
#                     "role": "user",
#                     "content": [
#                         {"type": "text", "text": prompt},
#                         {"type": "image_url", "image_url": {"url": f"data:image/png;base64,{combined_b64}"}},
#                     ],
#                 }
#             ],
#             max_tokens=200,
#         )
#         return response.choices[0].message.content.strip()

#     except Exception as e:
#         return f"⚠️ Error: {e}"


# # ==============================================================
# # Main evaluation loop
# # ==============================================================
# def evaluate_pairs(input_dir: Path, output_dir: Path, client: OpenAI, prompt: str):
#     subfolders = [p for p in sorted(output_dir.iterdir()) if p.is_dir()]
#     total, good, skipped = 0, 0, 0

#     for out_folder in tqdm(subfolders, desc="Comparing"):
#         orig_folder = input_dir / out_folder.name
#         if not orig_folder.exists():
#             continue

#         orig_imgs = sorted(orig_folder.glob("bdy_*.*"))
#         relit_imgs = sorted(out_folder.glob("bdy_*_warp_relight_unwarp.*"))
#         if not orig_imgs or not relit_imgs:
#             continue

#         orig_img = orig_imgs[0]
#         relit_img = relit_imgs[0]
#         total += 1

#         result = query_gpt_about_pair(client, orig_img, relit_img, prompt)

#         if "FINAL ANSWER" not in result:
#             skipped += 1
#             continue

#         if "**FINAL ANSWER:** Yes" in result or "FINAL ANSWER: Yes" in result:
#             good += 1

#     print(f"\n✅ Done! {good}/{total} pairs passed ({(good/total*100 if total else 0):.2f}%)")
#     print(f"⚠️ Skipped or unreadable pairs: {skipped}")


# # ==============================================================
# # Entry point
# # ==============================================================
# def main():
#     parser = argparse.ArgumentParser()
#     parser.add_argument("--input_dir", required=True, help="Original dataset path")
#     parser.add_argument("--output_dir", required=True, help="Relit output path")
#     args = parser.parse_args()

#     input_dir = Path(args.input_dir)
#     output_dir = Path(args.output_dir)
#     client = OpenAI()

#     CHECK_PROMPT = (
#         "Compare these two images: left = original, right = relit.\n"
#         "Answer Yes/No for each:\n"
#         "1. Same person?\n"
#         "Then write '**FINAL ANSWER:** Yes' only if all are Yes; else '**FINAL ANSWER:** No'."
#     )

#     print(f"\nInput root : {input_dir}")
#     print(f"Output root: {output_dir}\n")

#     evaluate_pairs(input_dir, output_dir, client, CHECK_PROMPT)


# if __name__ == "__main__":
#     main()