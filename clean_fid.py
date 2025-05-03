import os
import shutil
from cleanfid import fid
from natsort import natsorted  # NOTE: use this for correct natural sorting!

def compute_fid_with_temp_real_images(paths):
    """
    paths: dict with keys:
        'fdir1' : real images directory
        'fdir2' : fake/generated images directory
        'temp_real_dir' : temporary real images directory for matching
    """
    fdir1 = paths['fdir1']
    fdir2 = paths['fdir2']
    temp_real_dir = paths['temp_real_dir']

    # Clean and recreate temp_real_dir
    if os.path.exists(temp_real_dir):
        shutil.rmtree(temp_real_dir)
    os.makedirs(temp_real_dir, exist_ok=True)

    # Natural sort real and fake images
    all_real_images = [f for f in os.listdir(fdir1) if f.lower().endswith(('.png', '.jpg', '.jpeg'))]
    all_real_images = natsorted(all_real_images)

    all_fake_images = [f for f in os.listdir(fdir2) if f.lower().endswith(('.png', '.jpg', '.jpeg'))]
    all_fake_images = natsorted(all_fake_images)

    # NOTE: Restrict to first 100
    selected_real_images = all_real_images[:100]
    selected_fake_images = all_fake_images[:100]

    print(f"Selected {len(selected_real_images)} real images to match {len(selected_fake_images)} fake images.")

    # Symlink selected real images
    for img_name in selected_real_images:
        src = os.path.join(fdir1, img_name)
        dst = os.path.join(temp_real_dir, img_name)
        try:
            os.symlink(src, dst)
        except FileExistsError:
            pass  # Skip if symlink already exists

    # Create a temp fake dir (optional, for safety)
    temp_fake_dir = os.path.join(fdir2, "..", "fake_subset_temp")
    if os.path.exists(temp_fake_dir):
        shutil.rmtree(temp_fake_dir)
    os.makedirs(temp_fake_dir, exist_ok=True)

    for img_name in selected_fake_images:
        src = os.path.join(fdir2, img_name)
        dst = os.path.join(temp_fake_dir, img_name)
        try:
            os.symlink(src, dst)
        except FileExistsError:
            pass

    # Compute FID
    score = fid.compute_fid(temp_real_dir, temp_fake_dir)
    print("=================================================")
    print(f"Clean FID score: {score:.4f}")

    # Compute KID
    score = fid.compute_kid(temp_real_dir, temp_fake_dir)
    print("=================================================")
    print(f"Clean KID score: {score:.4f}")

    # Compute CLIP-FID
    score = fid.compute_fid(temp_real_dir, temp_fake_dir, mode="clean", model_name="clip_vit_b_32")
    print("=================================================")
    print(f"Clean CLIP-FID score: {score:.4f}")

    return score


# Example usage
# v1 
# paths = {
#     'fdir1': "/home/shenzhen/Relight_Projects/img2img-turbo/data/Seed_Direction_4_23/test_B",
#     'fdir2': "/home/shenzhen/Relight_Projects/img2img-turbo/output/pix2pix_turbo/Seed_Direction_4_23/eval/fid_6901",
#     'temp_real_dir': "/home/shenzhen/Relight_Projects/img2img-turbo/data/Seed_Direction_4_23/test_B_temp_subset"
# }

# # v2
# paths = {
#     'fdir1': "/home/shenzhen/Relight_Projects/img2img-turbo/data/Seed_Direction_4_24/test_B",
#     'fdir2': "/home/shenzhen/Relight_Projects/img2img-turbo/output/pix2pix_turbo/Seed_Direction_4_24/eval/fid_6901",
#     'temp_real_dir': "/home/shenzhen/Relight_Projects/img2img-turbo/data/Seed_Direction_4_24/test_B_temp_subset"
# }

# v3
# paths = {
#     'fdir1': "/home/shenzhen/Relight_Projects/img2img-turbo/data/Seed_Direction_4_24/test_B",
#     'fdir2': "/home/shenzhen/Relight_Projects/img2img-turbo/output/pix2pix_turbo/Seed_Direction_4_24_512x512/eval/fid_6901",
#     'temp_real_dir': "/home/shenzhen/Relight_Projects/img2img-turbo/data/Seed_Direction_4_24/test_B_temp_subset"
# }

# # v4
# paths = {
#     'fdir1': "/home/shenzhen/Relight_Projects/img2img-turbo/data/Seed_Direction_4_24_no_prompt/test_B",
#     'fdir2': "/home/shenzhen/Relight_Projects/img2img-turbo/output/pix2pix_turbo/Seed_Direction_4_24_no_prompt/eval/fid_6901",
#     'temp_real_dir': "/home/shenzhen/Relight_Projects/img2img-turbo/data/Seed_Direction_4_24_no_prompt/test_B_temp_subset"
# }

# # # v5
# paths = {
#     'fdir1': "/home/shenzhen/Relight_Projects/img2img-turbo/data/Seed_Direction_4_24_use_target_bg/test_B",
#     'fdir2': "/home/shenzhen/Relight_Projects/img2img-turbo/output/pix2pix_turbo/Seed_Direction_4_24_use_target_bg/eval/fid_6901",
#     'temp_real_dir': "/home/shenzhen/Relight_Projects/img2img-turbo/data/Seed_Direction_4_24_use_target_bg/test_B_temp_subset"
# }

# v5 (use original image's background)
paths = {
    'fdir1': "/home/shenzhen/Relight_Projects/img2img-turbo/data/Seed_Direction_4_24_use_target_bg/test_B",
    'fdir2': "/home/shenzhen/Relight_Projects/img2img-turbo/output/pix2pix_turbo/Seed_Direction_4_24_use_target_bg/result_A_original_bg",
    'temp_real_dir': "/home/shenzhen/Relight_Projects/img2img-turbo/data/Seed_Direction_4_24_use_target_bg/test_B_temp_subset"
}

# # v6
# paths = {
#     'fdir1': "/home/shenzhen/Relight_Projects/img2img-turbo/data/Seed_Direction_4_24_use_noise_bg/test_B",
#     'fdir2': "/home/shenzhen/Relight_Projects/img2img-turbo/output/pix2pix_turbo/Seed_Direction_4_24_use_noise_bg/eval/fid_6901",
#     'temp_real_dir': "/home/shenzhen/Relight_Projects/img2img-turbo/data/Seed_Direction_4_24_use_noise_bg/test_B_temp_subset"
# }

compute_fid_with_temp_real_images(paths)