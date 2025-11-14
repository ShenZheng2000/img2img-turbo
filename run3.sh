train_cyclegan() {
    DATASET_NAME="$1"
    TRAIN_PREP="$2"

    DATASET="/home/shenzhen/Datasets/BDD100K/${DATASET_NAME}"
    OUTPUT="output/cyclegan_turbo/${DATASET_NAME}_${TRAIN_PREP}"
    PROJECT="${DATASET_NAME//\//_}_${TRAIN_PREP}"  # for wandb tracking
    
    # NOTE: --max_train_steps assumes 8 GPUs!!!!!
    # NOTE: adjust config: /home/shenzhen/.cache/huggingface/accelerate/default_config.yaml
    accelerate launch --main_process_port 29501 src/train_cyclegan_turbo.py \
        --pretrained_model_name_or_path="stabilityai/sd-turbo" \
        --dataset_folder "$DATASET" \
        --output_dir="$OUTPUT" \
        --tracker_project_name="$PROJECT" \
        --learning_rate="1e-5" \
        --train_batch_size=1 \
        --enable_xformers_memory_efficient_attention \
        --report_to "wandb" \
        --train_img_prep "$TRAIN_PREP" \
        --val_img_prep "no_resize" \
        --validation_steps 5000 \
        --max_train_steps 25000
}

# train_cyclegan "bdd100k_7_19_night" "resize_286_randomcrop_256x256_hflip"
train_cyclegan "bdd100k_7_19_night_warped_128" "resize_286_randomcrop_256x256_hflip"

# TODO: how to unwarp correctly, with randomcrop and hflip? 