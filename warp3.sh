# # # # Example usage with input_root (different), warp-subfolders (all), and bbox-json (gt)
# target_prefix="cityscapes_to_dark_zurich"
# target_prefix="cityscapes_to_foggy_cityscapes_beta_002"
# target_prefix="cityscapes_to_dense_fog"
# target_prefix="cityscapes_to_acdc_fog"
# target_prefix="BDD100K_day2night"
# target_prefix="BDD100K_clear2rainy"
target_prefix="cityscapes_to_acdc_rain"

# NOTE: use bw 128 for now.
# NOTE: must use train json to warp, and check debug vis bbox to make sure warping works! 
python warp_dataset.py \
  --input_root /ssd0/shenzhen/Datasets/driving \
  --target_prefix $target_prefix \
  --bw 128 \
  --warp-subfolders train_A train_B \
  --bbox-json \
  /ssd0/shenzhen/Datasets/driving/cityscapes/gt_detection/instancesonly_filtered_gtFine_train_poly.json \
  /ssd0/shenzhen/Datasets/driving/acdc/gt_detection/instancesonly_train_gt_detection.json