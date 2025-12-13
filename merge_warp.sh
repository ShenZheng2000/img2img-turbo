input_root="/home/shenzhen/Datasets/relighting"
target_prefix="exp_10_16"
target_prefix_2="exp_12_7"
merged_prefix="${target_prefix}_${target_prefix_2}_merged"
relight_type="noon_sunlight_1"
bw=128

# # # merge dataset (optional, NOTE: merge and then warp)
# python merge_data.py \
#     --input_root $input_root \
#     --target_prefix $target_prefix \
#     --target_prefix_2 $target_prefix_2 \
#     --merged_prefix $merged_prefix \
#     --relight_type $relight_type

# python warp_dataset.py \
#     --input_root $input_root \
#     --target_prefix $merged_prefix \
#     --relight_type $relight_type \
#     --bw $bw \
#     --include-eyes