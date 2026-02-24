# NOTE: use base conda env! 
BDD_DAY_DIR=/ssd0/shenzhen/Datasets/driving/BDD100K_day2night/test_A
BDD_NIGHT_DIR=/ssd0/shenzhen/Datasets/driving/BDD100K_day2night/test_B
BDD_RAINY_DIR=/ssd0/shenzhen/Datasets/driving/BDD100K_clear2rainy/test_B
CS_DAY_DIR=/ssd0/shenzhen/Datasets/driving/cityscapes_to_dark_zurich/test_A
DZ_NIGHT_DIR=/ssd0/shenzhen/Datasets/driving/cityscapes_to_dark_zurich/test_B
ACDC_FOG_DIR=/ssd0/shenzhen/Datasets/driving/cityscapes_to_acdc_fog/test_B
DENSE_FOG_DIR=/ssd0/shenzhen/Datasets/driving/cityscapes_to_dense_fog/test_B
ACDC_RAIN_DIR=/ssd0/shenzhen/Datasets/driving/cityscapes_to_acdc_rain/test_B

run_fidelity_unwarp() {
  GPU="$1"
  INDIR="$2"
  REF="$3"

  TMP_DIR="$(mktemp -d /tmp/fid_unwarp.XXXXXX)"
  trap 'rm -rf "$TMP_DIR"' RETURN

  # collect generated images
  find "$INDIR" -maxdepth 1 -type f \
    \( -iname "*warp_relight_unwarp*.png" -o -iname "*warp_relight_unwarp*.jpg" -o -iname "*warp_relight_unwarp*.jpeg" \) \
    -print0 | while IFS= read -r -d '' f; do
      ln -sf "$(realpath "$f")" "$TMP_DIR/"
    done

  # counts
  N1=$(ls -1 "$TMP_DIR" | wc -l)
  N2=$(find "$REF" -maxdepth 1 -type f \( -iname '*.png' -o -iname '*.jpg' -o -iname '*.jpeg' \) | wc -l)

  MIN=$(( N1 < N2 ? N1 : N2 ))
  KID_S=$(( MIN < 1000 ? MIN : 1000 ))

  # Pass counts (n1, n2, kids) directly to awk. No separate echo needed.
  PYTHONWARNINGS=ignore fidelity \
    --gpu "$GPU" \
    --fid --kid \
    --kid-subset-size "$KID_S" \
    --rng-seed 0 \
    --input1 "$TMP_DIR" \
    --input2 "$REF" \
    --silent | awk -v n1="$N1" -v n2="$N2" -v ks="$KID_S" '
      # 1. Capture FID
      /frechet_inception_distance/ {
        for(i=1; i<=NF; i++) {
          if ($i ~ /[0-9]+\.[0-9]+/) fid = $i
        }
      }

      # 2. Capture KID and Scale (*1000)
      /kernel_inception_distance_mean/ {
        for(i=1; i<=NF; i++) {
          if ($i ~ /[0-9]+\.[0-9]+/) kid = $i * 1000
        }
      }

      # 3. Print the Clean Table Block
      END {
        print "################################################################################"
        printf "| Generated: %-9s | Reference: %-9s | Subset Size: %-9s |\n", n1, n2, ks
        print "|------------------------------------------------------------------------------|"
        printf "| FID:      %-10.4f | KID (*1k): %-10.4f                                  |\n", fid, kid
        print "################################################################################"
        print "" 
      }
    '
}


# # # # cityscapes_to_acdc_rain
# echo "=====================>Evaluating cityscapes_to_acdc_rain..."
# run_fidelity_unwarp 1 output/cyclegan_turbo/cityscapes_to_acdc_rain_resize_286_randomcrop_256x256_hflip/result_A "$ACDC_RAIN_DIR"
# run_fidelity_unwarp 1 output/cyclegan_turbo/cityscapes_to_acdc_rain_resize_286_randomcrop_256x256_hflip/result_B "$CS_DAY_DIR"

# # # # cityscapes_to_acdc_rain_warped_128
# echo "=====================>Evaluating cityscapes_to_acdc_rain_warped_128..."
# run_fidelity_unwarp 2 output/cyclegan_turbo/cityscapes_to_acdc_rain_warped_128_resize_286_randomcrop_256x256_hflip/result_A "$ACDC_RAIN_DIR"
# run_fidelity_unwarp 2 output/cyclegan_turbo/cityscapes_to_acdc_rain_warped_128_resize_286_randomcrop_256x256_hflip/result_B "$CS_DAY_DIR"


# # BDD100K_clear2rainy_resize_286_randomcrop_256x256_hflip
# echo "=====================>Evaluating BDD100K_clear2rainy_resize_286_randomcrop_256x256_hflip..."
# run_fidelity_unwarp 1 output/cyclegan_turbo/BDD100K_clear2rainy_resize_286_randomcrop_256x256_hflip/result_A "$BDD_RAINY_DIR"
# run_fidelity_unwarp 1 output/cyclegan_turbo/BDD100K_clear2rainy_resize_286_randomcrop_256x256_hflip/result_B "$BDD_DAY_DIR"

# # BDD100K_clear2rainy_warped_128_resize_286_randomcrop_256x256_hflip
# echo "=====================>Evaluating BDD100K_clear2rainy_warped_128_resize_286_randomcrop_256x256_hflip..."
# run_fidelity_unwarp 1 output/cyclegan_turbo/BDD100K_clear2rainy_warped_128_resize_286_randomcrop_256x256_hflip/result_A "$BDD_RAINY_DIR"
# run_fidelity_unwarp 1 output/cyclegan_turbo/BDD100K_clear2rainy_warped_128_resize_286_randomcrop_256x256_hflip/result_B "$BDD_DAY_DIR"


# # BDD100K_day2night_resize_286_randomcrop_256x256_hflip
# echo "=====================>Evaluating BDD100K_day2night_resize_286_randomcrop_256x256_hflip..."
# run_fidelity_unwarp 1 output/cyclegan_turbo/BDD100K_day2night_resize_286_randomcrop_256x256_hflip/result_A "$BDD_NIGHT_DIR"
# run_fidelity_unwarp 1 output/cyclegan_turbo/BDD100K_day2night_resize_286_randomcrop_256x256_hflip/result_B "$BDD_DAY_DIR"

# # BDD100K_day2night_warped_128_resize_286_randomcrop_256x256_hflip
# echo "=====================>Evaluating BDD100K_day2night_warped_128_resize_286_randomcrop_256x256_hflip..."
# run_fidelity_unwarp 1 output/cyclegan_turbo/BDD100K_day2night_warped_128_resize_286_randomcrop_256x256_hflip/result_A "$BDD_NIGHT_DIR"
# run_fidelity_unwarp 1 output/cyclegan_turbo/BDD100K_day2night_warped_128_resize_286_randomcrop_256x256_hflip/result_B "$BDD_DAY_DIR"


# # # cityscapes_to_dark_zurich
# echo "=====================>Evaluating cityscapes_to_dark_zurich..."
# run_fidelity_unwarp 1 output/cyclegan_turbo/cityscapes_to_dark_zurich_resize_286_randomcrop_256x256_hflip/result_A "$DZ_NIGHT_DIR"
# run_fidelity_unwarp 1 output/cyclegan_turbo/cityscapes_to_dark_zurich_resize_286_randomcrop_256x256_hflip/result_B "$CS_DAY_DIR"

# # # # cityscapes_to_dark_zurich_warped_128
# echo "=====================>Evaluating cityscapes_to_dark_zurich_warped_128..."
# run_fidelity_unwarp 1 output/cyclegan_turbo/cityscapes_to_dark_zurich_warped_128_resize_286_randomcrop_256x256_hflip/result_A "$DZ_NIGHT_DIR"
# run_fidelity_unwarp 1 output/cyclegan_turbo/cityscapes_to_dark_zurich_warped_128_resize_286_randomcrop_256x256_hflip/result_B "$CS_DAY_DIR"


# # # # cityscapes_to_dense_fog
# echo "=====================>Evaluating cityscapes_to_dense_fog..."
# run_fidelity_unwarp 1 output/cyclegan_turbo/cityscapes_to_dense_fog_resize_286_randomcrop_256x256_hflip/result_A "$DENSE_FOG_DIR"
# run_fidelity_unwarp 1 output/cyclegan_turbo/cityscapes_to_dense_fog_resize_286_randomcrop_256x256_hflip/result_B "$CS_DAY_DIR"

# # # # cityscapes_to_dense_fog_warped_128
# # # echo "=====================>Evaluating cityscapes_to_dense_fog_warped_128..."
# run_fidelity_unwarp 2 output/cyclegan_turbo/cityscapes_to_dense_fog_warped_128_resize_286_randomcrop_256x256_hflip/result_A "$DENSE_FOG_DIR"
# run_fidelity_unwarp 2 output/cyclegan_turbo/cityscapes_to_dense_fog_warped_128_resize_286_randomcrop_256x256_hflip/result_B "$CS_DAY_DIR"

# # # # cityscapes_to_acdc_fog
# echo "=====================>Evaluating cityscapes_to_acdc_fog..."
# run_fidelity_unwarp 1 output/cyclegan_turbo/cityscapes_to_acdc_fog_resize_286_randomcrop_256x256_hflip/result_A "$ACDC_FOG_DIR"
# run_fidelity_unwarp 1 output/cyclegan_turbo/cityscapes_to_acdc_fog_resize_286_randomcrop_256x256_hflip/result_B "$CS_DAY_DIR"

# # # # cityscapes_to_acdc_fog_warped_128
# echo "=====================>Evaluating cityscapes_to_acdc_fog_warped_128..."
# run_fidelity_unwarp 1 output/cyclegan_turbo/cityscapes_to_acdc_fog_warped_128_resize_286_randomcrop_256x256_hflip/result_A "$ACDC_FOG_DIR"
# run_fidelity_unwarp 1 output/cyclegan_turbo/cityscapes_to_acdc_fog_warped_128_resize_286_randomcrop_256x256_hflip/result_B "$CS_DAY_DIR"
