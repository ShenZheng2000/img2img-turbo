# NOTE: use img2img-turbo2 conda env! 
BDD_DAY_DIR=/ssd0/shenzhen/Datasets/driving/BDD100K_day2night/test_A
BDD_NIGHT_DIR=/ssd0/shenzhen/Datasets/driving/BDD100K_day2night/test_B
BDD_RAINY_DIR=/ssd0/shenzhen/Datasets/driving/BDD100K_clear2rainy/test_B
CS_DAY_DIR=/ssd0/shenzhen/Datasets/driving/cityscapes_to_dark_zurich/test_A
DZ_NIGHT_DIR=/ssd0/shenzhen/Datasets/driving/cityscapes_to_dark_zurich/test_B
ACDC_FOG_DIR=/ssd0/shenzhen/Datasets/driving/cityscapes_to_acdc_fog/test_B
DENSE_FOG_DIR=/ssd0/shenzhen/Datasets/driving/cityscapes_to_dense_fog/test_B
ACDC_RAIN_DIR=/ssd0/shenzhen/Datasets/driving/cityscapes_to_acdc_rain/test_B


run_cleanfid_dino() {
  GPU="$1"
  REAL_A="$2"
  REAL_B="$3"
  RES_A="$4"
  RES_B="$5"

  CUDA_VISIBLE_DEVICES="$GPU" \
  python "eval_scripts/eval_fid_dino.py" \
    --real_A "$REAL_A" \
    --real_B "$REAL_B" \
    --result_A "$RES_A" \
    --result_B "$RES_B" \
    --mode clean \
    --gen_suffix "warp_relight_unwarp" \
    --eval_resize 0 \
    --max_pairs -1
}


# # # cityscapes_to_acdc_rain
# echo "=====================>Evaluating cityscapes_to_acdc_rain..."
# run_cleanfid_dino 0 "$CS_DAY_DIR" "$ACDC_RAIN_DIR" \
#   output/cyclegan_turbo/cityscapes_to_acdc_rain_resize_286_randomcrop_256x256_hflip/result_A \
#   output/cyclegan_turbo/cityscapes_to_acdc_rain_resize_286_randomcrop_256x256_hflip/result_B

# # cityscapes_to_acdc_rain_warped_128
# echo "=====================>Evaluating cityscapes_to_acdc_rain_warped_128..."
# run_cleanfid_dino 1 "$CS_DAY_DIR" "$ACDC_RAIN_DIR" \
#   output/cyclegan_turbo/cityscapes_to_acdc_rain_warped_128_resize_286_randomcrop_256x256_hflip/result_A \
#   output/cyclegan_turbo/cityscapes_to_acdc_rain_warped_128_resize_286_randomcrop_256x256_hflip/result_B


# # BDD100K_clear2rainy_resize_286_randomcrop_256x256_hflip
# echo "=====================>Evaluating BDD100K_clear2rainy_resize_286_randomcrop_256x256_hflip..."
# run_cleanfid_dino 0 "$BDD_DAY_DIR" "$BDD_RAINY_DIR" \
#   output/cyclegan_turbo/BDD100K_clear2rainy_resize_286_randomcrop_256x256_hflip/result_A \
#   output/cyclegan_turbo/BDD100K_clear2rainy_resize_286_randomcrop_256x256_hflip/result_B

# # BDD100K_clear2rainy_warped_128_resize_286_randomcrop_256x256_hflip
# echo "=====================>Evaluating BDD100K_clear2rainy_warped_128_resize_286_randomcrop_256x256_hflip..."
# run_cleanfid_dino 1 "$BDD_DAY_DIR" "$BDD_RAINY_DIR" \
#   output/cyclegan_turbo/BDD100K_clear2rainy_warped_128_resize_286_randomcrop_256x256_hflip/result_A \
#   output/cyclegan_turbo/BDD100K_clear2rainy_warped_128_resize_286_randomcrop_256x256_hflip/result_B


# # # BDD100K_day2night_resize_286_randomcrop_256x256_hflip
# echo "=====================>Evaluating BDD100K_day2night_resize_286_randomcrop_256x256_hflip..."
# run_cleanfid_dino 0 "$BDD_DAY_DIR" "$BDD_NIGHT_DIR" \
#   output/cyclegan_turbo/BDD100K_day2night_resize_286_randomcrop_256x256_hflip/result_A \
#   output/cyclegan_turbo/BDD100K_day2night_resize_286_randomcrop_256x256_hflip/result_B


# # # # BDD100K_day2night_warped_128_resize_286_randomcrop_256x256_hflip
# echo "=====================>Evaluating BDD100K_day2night_warped_128_resize_286_randomcrop_256x256_hflip..."
# run_cleanfid_dino 0 "$BDD_DAY_DIR" "$BDD_NIGHT_DIR" \
#   output/cyclegan_turbo/BDD100K_day2night_warped_128_resize_286_randomcrop_256x256_hflip/result_A \
#   output/cyclegan_turbo/BDD100K_day2night_warped_128_resize_286_randomcrop_256x256_hflip/result_B



# cityscapes_to_dark_zurich
# echo "=====================>Evaluating cityscapes_to_dark_zurich..."
# run_cleanfid_dino 0 "$CS_DAY_DIR" "$DZ_NIGHT_DIR" \
#   output/cyclegan_turbo/cityscapes_to_dark_zurich_resize_286_randomcrop_256x256_hflip/result_A \
#   output/cyclegan_turbo/cityscapes_to_dark_zurich_resize_286_randomcrop_256x256_hflip/result_B

# cityscapes_to_dark_zurich_warped_128
# echo "=====================>Evaluating cityscapes_to_dark_zurich_warped_128..."
# run_cleanfid_dino 0 "$CS_DAY_DIR" "$DZ_NIGHT_DIR" \
#   output/cyclegan_turbo/cityscapes_to_dark_zurich_warped_128_resize_286_randomcrop_256x256_hflip/result_A \
#   output/cyclegan_turbo/cityscapes_to_dark_zurich_warped_128_resize_286_randomcrop_256x256_hflip/result_B

# # cityscapes_to_acdc_fog
# echo "=====================>Evaluating cityscapes_to_acdc_fog..."
# run_cleanfid_dino 0 "$CS_DAY_DIR" "$ACDC_FOG_DIR" \
#   output/cyclegan_turbo/cityscapes_to_acdc_fog_resize_286_randomcrop_256x256_hflip/result_A \
#   output/cyclegan_turbo/cityscapes_to_acdc_fog_resize_286_randomcrop_256x256_hflip/result_B

# # cityscapes_to_acdc_fog_warped_128
# echo "=====================>Evaluating cityscapes_to_acdc_fog_warped_128..."
# run_cleanfid_dino 0 "$CS_DAY_DIR" "$ACDC_FOG_DIR" \
#   output/cyclegan_turbo/cityscapes_to_acdc_fog_warped_128_resize_286_randomcrop_256x256_hflip/result_A \
#   output/cyclegan_turbo/cityscapes_to_acdc_fog_warped_128_resize_286_randomcrop_256x256_hflip/result_B

# # # cityscapes_to_dense_fog
# echo "=====================>Evaluating cityscapes_to_dense_fog..."
# run_cleanfid_dino 0 "$CS_DAY_DIR" "$DENSE_FOG_DIR" \
#     output/cyclegan_turbo/cityscapes_to_dense_fog_resize_286_randomcrop_256x256_hflip/result_A \
#     output/cyclegan_turbo/cityscapes_to_dense_fog_resize_286_randomcrop_256x256_hflip/result_B

# # cityscapes_to_dense_fog_warped_128
# echo "=====================>Evaluating cityscapes_to_dense_fog_warped_128..."
# run_cleanfid_dino 0 "$CS_DAY_DIR" "$DENSE_FOG_DIR" \
#     output/cyclegan_turbo/cityscapes_to_dense_fog_warped_128_resize_286_randomcrop_256x256_hflip/result_A \
#     output/cyclegan_turbo/cityscapes_to_dense_fog_warped_128_resize_286_randomcrop_256x256_hflip/result_B