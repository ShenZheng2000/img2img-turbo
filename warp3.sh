# # # # Example usage with input_root (different), warp-subfolders (all), and bbox-json (gt)
target_prefix="cityscapes_to_dark_zurich"

# TODO: think about warp to resize or not at all? can use this one --out-h 286 --out-w 286
# NOTE: use bw 128 for now.
python warp_dataset.py \
  --input_root /ssd0/shenzhen/Datasets/driving \
  --target_prefix $target_prefix \
  --bw 128 \
  --warp-subfolders train_A train_B \
  --train-bbox-json /ssd0/shenzhen/Datasets/driving/cityscapes/gt_detection/instancesonly_filtered_gtFine_train_poly.json \
  --val-bbox-json /ssd0/shenzhen/Datasets/driving/cityscapes/gt_detection/instancesonly_filtered_gtFine_val_poly.json