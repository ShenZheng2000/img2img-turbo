input_root="/ssd0/shenzhen/Datasets/relighting"
target_prefix="2_24_drive_v2"
relight_type="foggy_1"
bw=64

# NOTE: (above) reduce bw from 128 to 64 for more intense warping
# NOTE: REMOVE --include-eyes for now! 
# NOTE: hardcode --use-yoloworld for now!
CUDA_VISIBLE_DEVICES=1 python warp_dataset.py \
    --input_root $input_root \
    --target_prefix $target_prefix \
    --relight_type $relight_type \
    --bw $bw \
    --use-yoloworld