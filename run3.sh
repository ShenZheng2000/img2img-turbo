train_cyclegan() {
    DATASET_NAME="$1"

    DATASET="/home/shenzhen/Datasets/BDD100K/${DATASET_NAME}"
    OUTPUT="output/cyclegan_turbo/${DATASET_NAME}"
    PROJECT="${DATASET_NAME//\//_}"  # for wandb
    
    # NOTE: --max_train_steps assumes 8 GPUs
    # TODO: maybe increase
    # TODO: maybe use different train_img_prep (since later we want to warp! )
    # TODO: retun validation with longer steps. 
    accelerate launch --main_process_port 29501 src/train_cyclegan_turbo.py \
        --pretrained_model_name_or_path="stabilityai/sd-turbo" \
        --dataset_folder "$DATASET" \
        --output_dir="$OUTPUT" \
        --tracker_project_name="$PROJECT" \
        --learning_rate="1e-5" \
        --train_batch_size=1 \
        --enable_xformers_memory_efficient_attention \
        --report_to "wandb" \
        --train_img_prep "resize_286_randomcrop_256x256_hflip" \
        --val_img_prep "no_resize" \
        --validation_steps 5000 \
        --max_train_steps 25000
}

train_cyclegan bdd100k_7_19_night