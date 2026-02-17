# # # # Example usage with input_root (different), warp-subfolders (all), and bbox-json (gt)
# target_prefix="cityscapes_to_dark_zurich"
# target_prefix="cityscapes_to_foggy_cityscapes_beta_002"
# target_prefix="cityscapes_to_dense_fog"
# target_prefix="cityscapes_to_acdc_fog"
# target_prefix="BDD100K_day2night"
target_prefix="BDD100K_clear2rainy"

# TODO: think about warp to resize or not at all? can use this one --out-h 286 --out-w 286
# NOTE: use bw 128 for now.
python warp_dataset.py \
  --input_root /ssd0/shenzhen/Datasets/driving \
  --target_prefix $target_prefix \
  --bw 128 \
  --warp-subfolders train_A train_B \
  --bbox-json \
  /ssd0/shenzhen/Datasets/driving/BDD100K/100k/coco_labels/det_train_coco.json