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

    DATASET="/ssd1/shenzhen/relighting/${DATASET_NAME}"
    OUTPUT="/ssd1/shenzhen/img2img-turbo/output/pix2pix_turbo/${EXP}"
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

# DOING: warp image with full body (100 big face, smaller bw)
train_pix2pix "exp_10_16_warped_256/candlelight_1"

# DOING: warp image with full body (100 big face)
# train_pix2pix "exp_10_16_warped_512/candlelight_1"

# DONE: try upon warped images on half body 
# train_pix2pix "exp_10_9_warped/candlelight_1"

# DONE: try noon_sunlight_1 training on half body
# train_pix2pix "exp_10_9/noon_sunlight_1"

# DATASET_NAME="exp_10_9/candlelight_1"
# DATASET_NAME="exp_10_11/candlelight_1"
# DATASET_NAME="exp_10_16/candlelight_1"

# # baseline (full-body), top 100 images x 100 seeds, big head
# train_pix2pix "$DATASET_NAME" "exp_10_16/candlelight_1"

# baseline (full-body), 1000 images x 10 seeds, big head
# train_pix2pix "$DATASET_NAME" "exp_10_11/candlelight_1"

# baseline (upper-body), 1000 images x 10 seeds
# train_pix2pix "$DATASET_NAME" "exp_10_11/candlelight_1"