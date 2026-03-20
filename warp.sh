warp_relight() {
    local exp_name="$1"     # must match yaml name (without .yaml)
    local relight_type="$2"
    local gpu="${3:-1}"

    local input_root
    case "$exp_name" in
        exp_*|2_24_*)
            input_root="/home/shenzhen/Datasets/relighting"
            ;;
        cityscapes_*|BDD100K_*)
            input_root="/ssd0/shenzhen/Datasets/driving"
            ;;
        *)
            echo "Cannot infer input_root from exp_name: $exp_name"
            return 1
            ;;
    esac

    CUDA_VISIBLE_DEVICES=$gpu python warp_dataset.py \
        --exp_config "configs/${exp_name}.yaml" \
        --input_root "$input_root" \
        --target_prefix "${exp_name%%_warped*}" \
        --relight_type "$relight_type"
}

# Examples
# warp_relight 2_24_drive_v2_warped_128 golden_sunlight_1
# warp_relight exp_1_10_1_warped_128_eyes golden_sunlight_1
warp_relight cityscapes_to_acdc_fog_warped_128 



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


# # # # # Example usage with input_root (different), warp-subfolders (all), and bbox-json (gt)
# # target_prefix="cityscapes_to_dark_zurich"
# # target_prefix="cityscapes_to_dense_fog"
# # target_prefix="cityscapes_to_acdc_fog"
# # target_prefix="BDD100K_day2night"
# # target_prefix="BDD100K_clear2rainy"

# # NOTE: use bw 128 for now.
# # NOTE: must use train json to warp, and check debug vis bbox to make sure warping works! 
# python warp_dataset.py \
#   --input_root /ssd0/shenzhen/Datasets/driving \
#   --target_prefix $target_prefix \
#   --bw 128 \
#   --warp-subfolders train_A train_B \
#   --bbox-json \
#   /ssd0/shenzhen/Datasets/driving/cityscapes/gt_detection/instancesonly_filtered_gtFine_train_poly.json \
#   /ssd0/shenzhen/Datasets/driving/acdc/gt_detection/instancesonly_train_gt_detection.json