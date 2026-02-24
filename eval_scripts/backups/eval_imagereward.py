# # Ref: https://github.com/zai-org/ImageReward
# import os
# import glob
# import argparse
# import ImageReward as RM

# # Example usage: 
# '''
# # TODO: first verify small samples works, then merge code with eval_clip.py and try on large-samples!
# python eval_scripts/eval_imagereward.py --input /home/shenzhen/Relight_Projects/img2img-turbo/output/pix2pix_turbo/exp_1_10_1/golden_sunlight_1/VITON/test/image --caption "Relit with warm golden sunlight during the late afternoon, casting gentle directional shadows and surrounding the subject in soft amber tones to create a calm, radiant mood."
# CUDA_VISIBLE_DEVICES=1 python eval_scripts/eval_imagereward.py --input /home/shenzhen/Relight_Projects/img2img-turbo/output/pix2pix_turbo/exp_1_10_1_warped_128_eyes/golden_sunlight_1/VITON/test/image --caption "Relit with warm golden sunlight during the late afternoon, casting gentle directional shadows and surrounding the subject in soft amber tones to create a calm, radiant mood."
# CUDA_VISIBLE_DEVICES=2 python eval_scripts/eval_imagereward.py --input /home/shenzhen/Relight_Projects/img2img-turbo/output/pix2pix_turbo/exp_1_1_warped_128_eyes/golden_sunlight_1/VITON/test/image --caption "Relit with warm golden sunlight during the late afternoon, casting gentle directional shadows and surrounding the subject in soft amber tones to create a calm, radiant mood."
# '''

# def parse_args():
#     p = argparse.ArgumentParser()
#     p.add_argument("--input", required=True, help="image folder or file")
#     p.add_argument("--caption", required=True, help="prompt / caption")
#     # NOTE: default MUST be _warp_relight_unwarp!
#     p.add_argument(
#         "--suffix",
#         default="_warp_relight_unwarp.png",
#         help="image suffix filter ('' for all images)",
#     )
#     return p.parse_args()


# def collect_images(path, suffix):
#     if os.path.isfile(path):
#         return [path]
#     if suffix == "":
#         exts = ("*.png", "*.jpg", "*.jpeg", "*.webp")
#         files = []
#         for e in exts:
#             files.extend(glob.glob(os.path.join(path, e)))
#         return sorted(files)
#     return sorted(glob.glob(os.path.join(path, f"*{suffix}")))


# def main():
#     args = parse_args()

#     print("Loading ImageReward model...")
#     model = RM.load("ImageReward-v1.0", device="cuda")

#     imgs = collect_images(args.input, args.suffix)
#     if len(imgs) == 0:
#         print("No images found")
#         return

#     # batch scoring (faster than per-image loop)
#     scores = model.score(args.caption, imgs)

#     # normalize to python list of floats
#     if isinstance(scores, float):
#         scores_list = [scores]
#     elif hasattr(scores, "detach"):  # torch tensor
#         scores_list = scores.detach().cpu().flatten().tolist()
#     else:  # already list
#         scores_list = list(scores)

#     avg = sum(scores_list) / len(scores_list)
#     print(f"IMGRWD {avg:.3f}")


# if __name__ == "__main__":
#     main()