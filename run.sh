# pip install huggingface_hub==0.25.0
# pip install peft==0.10.0
# pip install wandb
# pip install vision_aided_loss

# NOTE: --max_train_steps is doubled (10k->20k) with fewer gpus (8->4)!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!
# NOTE: hardcode DATASET_NAME and EXP to be the same. 
# NOTE: adjust config if needed: /home/shenzhen/.cache/huggingface/accelerate/default_config.yaml
train_pix2pix() {
    DATASET_NAME=$1   # e.g. exp_10_16/candlelight_1; exp_10_16_warped/candlelight_1
    EXP="$DATASET_NAME"  # exactly the same

    DATASET="/home/shenzhen/Datasets/relighting/${DATASET_NAME}"
    OUTPUT="output/pix2pix_turbo/${EXP}"
    PROJECT="${EXP//\//_}"  # for wandb

    accelerate launch src/train_pix2pix_turbo.py \
        --pretrained_model_name_or_path="stabilityai/sd-turbo" \
        --dataset_folder="$DATASET" \
        --output_dir="$OUTPUT" \
        --tracker_project_name="$PROJECT" \
        --resolution=512 \
        --train_batch_size=1 \
        --enable_xformers_memory_efficient_attention \
        --viz_freq 25 \
        --track_val_fid \
        --report_to "wandb" \
        --train_image_prep "no_resize" \
        --test_image_prep "no_resize" \
        --max_train_steps 20000
}

# DONE: let's continue experiments for noon_sunlight. 
# train_pix2pix "exp_10_11_warped_128/candlelight_1"
# train_pix2pix "exp_10_11_warped_512/candlelight_1"
# train_pix2pix "exp_10_11/candlelight_1" 
train_pix2pix "exp_10_16_warped_64/candlelight_1"

# DONE: noon_sunlight experiments (warped)
# train_pix2pix "exp_10_16_warped_512/noon_sunlight_1"
# train_pix2pix "exp_10_16_warped_128/noon_sunlight_1" 
# train_pix2pix "exp_10_16/noon_sunlight_1"
# train_pix2pix "exp_10_16_warped_64/noon_sunlight_1"

# DONE: bw of 192
# train_pix2pix "exp_10_16_warped_192/candlelight_1"

# DONE: bw of 320
# train_pix2pix "exp_10_16_warped_320/candlelight_1"

# DONE: bw of 384.
# train_pix2pix "exp_10_16_warped_384/candlelight_1"

# DONE: bw of 128. 
# train_pix2pix "exp_10_16_warped_128/candlelight_1"

# DONE: warp image with full body (100 big face, smaller bw)
# train_pix2pix "exp_10_16_warped_256/candlelight_1"

# DONE: warp image with full body (100 big face)
# train_pix2pix "exp_10_16_warped_512/candlelight_1"

# DONE: try upon warped images on half body 
# train_pix2pix "exp_10_9_warped/candlelight_1"

# DONE: try noon_sunlight_1 training on half body
# train_pix2pix "exp_10_9/noon_sunlight_1"

# DONE: try candlelight_1 training on half body
# train_pix2pix "exp_10_9/candlelight_1"