# # Example usage: 
python warp_dataset.py \
    --target_prefix exp_10_16 \
    --relight_type candlelight_1 \
    --bw 128 \
    --include-eyes

# TODO: check if curretn change will break face detector logic! 
# TODO: later training should be able to handle inv.pth in both folders
# TODO: think about if resize-crop will screw up the warped images. 
# TODO: change inference (unpaired) later; 

# Example usage with input_root (different), warp-subfolders (all), and bbox-json (gt)
# CUDA_VISIBLE_DEVICES=7 python warp_dataset.py \
#   --target_prefix bdd100k_7_19_night \
#   --bw 128 \
#   --input_root /home/shenzhen/Datasets/BDD100K \
#   --train-bbox-json /home/shenzhen/Datasets/BDD100K/100k/coco_labels/bdd100k_train_scale_0p5.json \
#   --val-bbox-json /home/shenzhen/Datasets/BDD100K/100k/coco_labels/bdd100k_val_scale_0p5.json