# pip install huggingface_hub==0.25.0
# pip install peft==0.10.0
# pip install wandb
# pip install vision_aided_loss

# NOTE: change to --train_image_prep "resize_784"; --test_image_prep "resize_784", and --train_batch_size=1
# NOTE: keep --resolution=512 for standard FID evaluation
# Maybe: increase --eval_freq and --viz_freq to 500?

# v1: white background for base image                        [Clean FID score: 122.0491]
# accelerate launch src/train_pix2pix_turbo.py \
#     --pretrained_model_name_or_path="stabilityai/sd-turbo" \
#     --output_dir="output/pix2pix_turbo/Seed_Direction_4_23" \
#     --dataset_folder="data/Seed_Direction_4_23" \
#     --resolution=512 \
#     --train_batch_size=1 \
#     --enable_xformers_memory_efficient_attention --viz_freq 25 \
#     --track_val_fid \
#     --report_to "wandb" --tracker_project_name "pix2pix_turbo_Seed_Direction_4_23" \
#     --train_image_prep "resize_784" \
#     --test_image_prep "resize_784"


# v2: KEEP the background for base image                     [Clean FID score: 105.9337]
# accelerate launch src/train_pix2pix_turbo.py \
#     --pretrained_model_name_or_path="stabilityai/sd-turbo" \
#     --output_dir="output/pix2pix_turbo/Seed_Direction_4_24" \
#     --dataset_folder="data/Seed_Direction_4_24" \
#     --resolution=512 \
#     --train_batch_size=1 \
#     --enable_xformers_memory_efficient_attention --viz_freq 25 \
#     --track_val_fid \
#     --report_to "wandb" --tracker_project_name "pix2pix_turbo_Seed_Direction_4_24" \
#     --train_image_prep "resize_784" \
#     --test_image_prep "resize_784"


# v3: v2 + size of 512x512                                   [Clean FID score: 102.9583]
# accelerate launch src/train_pix2pix_turbo.py \
#     --pretrained_model_name_or_path="stabilityai/sd-turbo" \
#     --output_dir="output/pix2pix_turbo/Seed_Direction_4_24_512x512" \
#     --dataset_folder="data/Seed_Direction_4_24" \
#     --resolution=512 \
#     --train_batch_size=1 \
#     --enable_xformers_memory_efficient_attention --viz_freq 25 \
#     --track_val_fid \
#     --report_to "wandb" --tracker_project_name "pix2pix_turbo_Seed_Direction_4_24_512x512"


# # v4: v2 + no text prompt                                   [Clean FID score: 103.4222]
# accelerate launch src/train_pix2pix_turbo.py \
#     --pretrained_model_name_or_path="stabilityai/sd-turbo" \
#     --output_dir="output/pix2pix_turbo/Seed_Direction_4_24_no_prompt" \
#     --dataset_folder="data/Seed_Direction_4_24_no_prompt" \
#     --resolution=512 \
#     --train_batch_size=1 \
#     --enable_xformers_memory_efficient_attention --viz_freq 25 \
#     --track_val_fid \
#     --report_to "wandb" --tracker_project_name "pix2pix_turbo_Seed_Direction_4_24_no_prompt" \
#     --train_image_prep "resize_784" \
#     --test_image_prep "resize_784"

# v5: v2 + replace base image's background with target image's background          [Clean FID score: 69.6481]
# accelerate launch src/train_pix2pix_turbo.py \
#     --pretrained_model_name_or_path="stabilityai/sd-turbo" \
#     --output_dir="output/pix2pix_turbo/Seed_Direction_4_24_use_target_bg" \
#     --dataset_folder="data/Seed_Direction_4_24_use_target_bg" \
#     --resolution=512 \
#     --train_batch_size=1 \
#     --enable_xformers_memory_efficient_attention --viz_freq 25 \
#     --track_val_fid \
#     --report_to "wandb" --tracker_project_name "pix2pix_turbo_Seed_Direction_4_24_use_target_bg" \
#     --train_image_prep "resize_784" \
#     --test_image_prep "resize_784"

# # v6: v2 + replace base image's background with gaussian noise
# accelerate launch src/train_pix2pix_turbo.py \
#     --pretrained_model_name_or_path="stabilityai/sd-turbo" \
#     --output_dir="output/pix2pix_turbo/Seed_Direction_4_24_use_noise_bg" \
#     --dataset_folder="data/Seed_Direction_4_24_use_noise_bg" \
#     --resolution=512 \
#     --train_batch_size=1 \
#     --enable_xformers_memory_efficient_attention --viz_freq 25 \
#     --track_val_fid \
#     --report_to "wandb" --tracker_project_name "pix2pix_turbo_Seed_Direction_4_24_use_noise_bg" \
#     --train_image_prep "resize_784" \
#     --test_image_prep "resize_784"


# # v6: use original background
# python src/inference_paired_folder.py --model_path "output/pix2pix_turbo/Seed_Direction_4_24_use_target_bg/checkpoints/model_6501.pkl" \
#     --input_dir "data/Seed_Direction_4_24/test_A" \
#     --prompt "Relit with golden-hour sunlight from behind, softly outlining the subject in amber tones, casting long, fading shadows, and creating a calm, atmospheric glow." \
#     --output_dir "output/pix2pix_turbo/Seed_Direction_4_24_use_target_bg/result_A_original_bg"

# # # v6: use target background 
# python src/inference_paired_folder.py --model_path "output/pix2pix_turbo/Seed_Direction_4_24_use_target_bg/checkpoints/model_6501.pkl" \
#     --input_dir "data/Seed_Direction_4_24_use_target_bg/test_A" \
#     --prompt "Relit with golden-hour sunlight from behind, softly outlining the subject in amber tones, casting long, fading shadows, and creating a calm, atmospheric glow." \
#     --output_dir "output/pix2pix_turbo/Seed_Direction_4_24_use_target_bg/result_A"

# NOTE: place data in /scratch, not local folder, or data3/ (will slow down)
# candlelight_1_may4 (same settings as v2, but different dataset)
# NOTE: no_resize same as resize_784 (since the original image is 784x784)
accelerate launch src/train_pix2pix_turbo.py \
    --pretrained_model_name_or_path="stabilityai/sd-turbo" \
    --output_dir="/scratch1/shenzhen/img2img-turbo/output/pix2pix_turbo/candlelight_1_may4" \
    --dataset_folder="/scratch1/shenzhen/img2img-turbo/data/candlelight_1_may4" \
    --resolution=512 \
    --train_batch_size=1 \
    --enable_xformers_memory_efficient_attention --viz_freq 25 \
    --track_val_fid \
    --report_to "wandb" --tracker_project_name "pix2pix_turbo_candlelight_1_may4" \
    --train_image_prep "no_resize" \
    --test_image_prep "no_resize"