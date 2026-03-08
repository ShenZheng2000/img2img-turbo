# NOTE: use img2img-turbo2 conda env! 
REAL_A="/home/shenzhen/Datasets/depth/workzone_segm/boston/image_cc784"

IC_Light_GS="/home/shenzhen/Relight_Projects/IC-Light/outputs/512x512/golden_sunlight_1/depth/workzone_segm/boston"
DreamLight_GS="/home/shenzhen/Relight_Projects/DreamLight/outputs/512x512/golden_sunlight_1/depth/workzone_segm/boston"
Ours_NoWarp_GS="/home/shenzhen/Relight_Projects/img2img-turbo/output/pix2pix_turbo/2_24_drive_v2/golden_sunlight_1/depth/workzone_segm/boston/image_cr512/image"
Ours_GS="/home/shenzhen/Relight_Projects/img2img-turbo/output/pix2pix_turbo/2_24_drive_v2_warped_128/golden_sunlight_1/depth/workzone_segm/boston/image/image"

IC_Light_FOGGY="/home/shenzhen/Relight_Projects/IC-Light/outputs/512x512/foggy_1/depth/workzone_segm/boston"
DreamLight_FOGGY="/home/shenzhen/Relight_Projects/DreamLight/outputs/512x512/foggy_1/depth/workzone_segm/boston"
Ours_NoWarp_FOGGY="/home/shenzhen/Relight_Projects/img2img-turbo/output/pix2pix_turbo/2_24_drive_v2/foggy_1/depth/workzone_segm/boston/image_cr512/image"
Ours_FOGGY="/home/shenzhen/Relight_Projects/img2img-turbo/output/pix2pix_turbo/2_24_drive_v2_warped_128/foggy_1/depth/workzone_segm/boston/image/image"


run_dino_only() {
  GPU="$1"
  REAL_A="$2"
  RES_A="$3"
  SUFFIX="$4"

  CUDA_VISIBLE_DEVICES="$GPU" \
  python -m eval_scripts.eval_dino \
    --real_A "$REAL_A" \
    --result_A "$RES_A" \
    --gen_suffix "$SUFFIX" \
    --eval_resize 0 \
    --max_pairs -1
}

# run_dino_only 1 "$REAL_A" "$IC_Light_GS" ""
# run_dino_only 1 "$REAL_A" "$DreamLight_GS" ""
# run_dino_only 1 "$REAL_A" "$Ours_NoWarp_GS" "warp_relight_unwarp"
# run_dino_only 1 "$REAL_A" "$Ours_GS" "warp_relight_unwarp"

# run_dino_only 1 "$REAL_A" "$IC_Light_FOGGY" ""
# run_dino_only 1 "$REAL_A" "$DreamLight_FOGGY" ""
# run_dino_only 1 "$REAL_A" "$Ours_NoWarp_FOGGY" "warp_relight_unwarp"
run_dino_only 1 "$REAL_A" "$Ours_FOGGY" "warp_relight_unwarp"

