# # # # Example usage with input_root (different), warp-subfolders (all), and bbox-json (gt)
# # target_prefix="bdd100k_7_19_night"
# target_prefix="bdd100k_1_20"
# # target_prefix="bdd100k_boreas"

# # NOTE: use these new output height and width to debug now. 
# python warp_dataset.py \
#   --input_root /home/shenzhen/Datasets/BDD100K \
#   --target_prefix $target_prefix \
#   --bw 64 \
#   --warp-subfolders train_A train_B \
#   --train-bbox-json /home/shenzhen/Datasets/BDD100K/100k/coco_labels/bdd100k_train_scale_0p5.json \
#   --val-bbox-json /home/shenzhen/Datasets/BDD100K/100k/coco_labels/bdd100k_val_scale_0p5.json \
#   --out-h 286 \
#   --out-w 286