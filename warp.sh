# # Example usage: 
python warp_dataset.py \
    --input_root /home/shenzhen/Datasets/relighting \
    --target_prefix exp_10_16 \
    --relight_type candlelight_1 \
    --bw 128 \
    --include-eyes

# # Example usage with input_root (different), warp-subfolders (all), and bbox-json (gt)
# target_prefix="bdd100k_7_19_night"
# target_prefix="bdd100k_1_20"

# python warp_dataset.py \
#   --target_prefix $target_prefix \
#   --bw 128 \
#   --input_root /home/shenzhen/Datasets/BDD100K \
#   --train-bbox-json /home/shenzhen/Datasets/BDD100K/100k/coco_labels/bdd100k_train_scale_0p5.json \
#   --val-bbox-json /home/shenzhen/Datasets/BDD100K/100k/coco_labels/bdd100k_val_scale_0p5.json