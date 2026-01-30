input_root="/ssd0/shenzhen/Datasets/relighting"
target_prefix="exp_1_10_1"
relight_type="foggy_1"
bw=128

CUDA_VISIBLE_DEVIVCES=7 python warp_dataset.py \
    --input_root $input_root \
    --target_prefix $target_prefix \
    --relight_type $relight_type \
    --bw $bw \
    --include-eyes