# # NOTE: use resize_256 for train and val images (because OOM error)
# NOTE: 480 is the highest resolution we can use (512 throws OOM error)
export NCCL_P2P_DISABLE=1
accelerate launch --main_process_port 29501 src/train_cyclegan_turbo.py \
    --pretrained_model_name_or_path="stabilityai/sd-turbo" \
    --output_dir="output/cyclegan_turbo/Seed_Direction_4_24_no_prompt_480" \
    --dataset_folder "data/Seed_Direction_4_24_no_prompt" \
    --train_img_prep "resize_480" --val_img_prep "resize_480" \
    --learning_rate="1e-5" --max_train_steps=25000 \
    --train_batch_size=1 --gradient_accumulation_steps=1 \
    --report_to "wandb" --tracker_project_name "cycleghan_turbo_Seed_Direction_4_24_no_prompt" \
    --enable_xformers_memory_efficient_attention --validation_steps 250 \
    --lambda_gan 0.5 --lambda_idt 1 --lambda_cycle 1
