input_root="/home/shenzhen/Datasets/relighting"
target_prefix="exp_10_16"
relight_type="foggy_1"
bw=64

CUDA_VISIBLE_DEVICES=6 python warp_dataset.py \
    --input_root $input_root \
    --target_prefix $target_prefix \
    --relight_type $relight_type \
    --bw $bw \
    --include-eyes