# Installations
# pip install torch==2.1.0 torchvision==0.16.0 torchaudio==2.1.0 --index-url https://download.pytorch.org/whl/cu121
# pip install xformers>=0.0.20
# pip install --upgrade transformers diffusers

# NOTE: use `img2img-turbo2` for repo
# NOTE: use 512x512 for now, if OOM, reduce batch size
# NOTE: use `--validation_image_prep` instead of `--test_image_prep` for validation
# NOTE: remove `--eval_freq`

# TODO: encode_prompt not defined. Wait for github repo's author to fix it

accelerate launch src/train_pix2pix_turbo.py \
    --dataset_folder /scratch1/shenzhen/img2img-turbo/data/candlelight_1_may4 \
    --train_image_prep resized_crop_512 \
    --validation_image_prep resized_crop_512 \
    --pretrained_model_name_or_path stabilityai/stable-diffusion-xl-base-1.0 \
    --model_type sdxl \
    --num_inputs 2 \
    --output_dir /scratch1/shenzhen/img2img-turbo2/output/pix2pix_turbo/candlelight_1_may4 \
    --max_train_steps 10000 \
    --train_batch_size 4 \
    --viz_freq 500 \
    --track_val_fid \
    --checkpointing_steps 2000 \
    --gradient_accumulation_steps 2 \
    --learning_rate 5e-6 \
    --enable_xformers_memory_efficient_attention