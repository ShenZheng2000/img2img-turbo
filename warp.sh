warp_relight() {
    local mode="$1"         # (can keep or remove, no longer needed)
    local exp_name="$2"     # MUST match yaml name
    local relight_type="$3"

    local input_root="/home/shenzhen/Datasets/relighting"
    local gpu=1

    # NOTE: strip warp from exp_name to get correct image for target_prefix. 
    CUDA_VISIBLE_DEVICES=$gpu python warp_dataset.py \
        --exp_config configs/${exp_name}.yaml \
        --input_root "$input_root" \
        --target_prefix "${exp_name%%_warped*}" \
        --relight_type "$relight_type"
}


# Example usage:
warp_relight drive 2_24_drive_v2_warped_128 golden_sunlight_1
# warp_relight human exp_1_10_1_warped_128_eyes golden_sunlight_1




##### Old code (for backup reference now)
# # Driving Relight
# input_root="/ssd0/shenzhen/Datasets/relighting"
# exp_name="2_24_drive_v2"
# relight_type="foggy_1"
# bw=128

# # NOTE: REMOVE --include-eyes for now! 
# # NOTE: hardcode --use-yoloworld for now!
# CUDA_VISIBLE_DEVICES=1 python warp_dataset.py \
#     --input_root $input_root \
#     --target_prefix $exp_name \
#     --relight_type $relight_type \
#     --bw $bw \
#     --use-yoloworld \
#     --yolo-model-path yolov8x-world.pt


# # Human Relight
# input_root="/ssd0/shenzhen/Datasets/relighting"
# exp_name="1_10_1_warped_128_eyes"
# relight_type="foggy_1"
# bw=128

# CUDA_VISIBLE_DEVICES=1 python warp_dataset.py \
#     --input_root $input_root \
#     --target_prefix $exp_name \
#     --relight_type $relight_type \
#     --bw $bw \
#     --include-eyes