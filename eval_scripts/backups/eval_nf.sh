# conda create -n pyiqa python=3.10 -y
# conda activate pyiqa
# pip install pyiqa
# pip install "setuptools<82"


#!/usr/bin/env bash
set -e

NGPU="${NGPU:-8}"   # run up to NGPU evals at once (GPU 0..NGPU-1)

# Edit these if your paths differ
PIX2PIX_ROOT="/home/shenzhen/Relight_Projects/img2img-turbo/output/pix2pix_turbo"
ICLIGHT_ROOT="/home/shenzhen/Relight_Projects/IC-Light/outputs/512x512"
DREAMLIGHT_ROOT="/home/shenzhen/Relight_Projects/DreamLight/outputs/512x512"

# Which NR metrics to run (comma-separated)
METRICS_CSV="${METRICS_CSV:-musiq,clipiqa,dbcnn,nima}"

run_eval_iqa() {
    METHOD_IN="$1"   # pix2pix_turbo | iclight | dreamlight
    NAME="$2"        # pix2pix_turbo: exp_xxx/.../golden_sunlight_1 ; iclight/dreamlight: just golden_sunlight_1
    DATASET="$3"     # e.g. VITON/test
    GPU_ID="${4:-0}"

    # infer LIGHT from NAME (works for both exp paths and light-only)
    if [[ "$NAME" == *"noon_sunlight_1"* ]]; then
        LIGHT="noon_sunlight_1"
    elif [[ "$NAME" == *"golden_sunlight_1"* ]]; then
        LIGHT="golden_sunlight_1"
    elif [[ "$NAME" == *"foggy_1"* ]]; then
        LIGHT="foggy_1"
    elif [[ "$NAME" == *"moonlight_1"* ]]; then
        LIGHT="moonlight_1"
    else
        echo "Unknown lighting type in name: $NAME"
        exit 1
    fi

    # output folder by method
    if [[ "$METHOD_IN" == "pix2pix_turbo" ]]; then
        OUT_DIR="${PIX2PIX_ROOT}/${NAME}/${DATASET}/image"
        SUFFIX_ARG="warp_relight_unwarp.png"   # pix2pix outputs this suffix
    elif [[ "$METHOD_IN" == "iclight" ]]; then
        OUT_DIR="${ICLIGHT_ROOT}/${LIGHT}/${DATASET}/image"
        SUFFIX_ARG=""                          # iclight images are normal names
    elif [[ "$METHOD_IN" == "dreamlight" ]]; then
        OUT_DIR="${DREAMLIGHT_ROOT}/${LIGHT}/${DATASET}/image"
        SUFFIX_ARG=""
    else
        echo "Unknown METHOD: $METHOD_IN"
        exit 1
    fi

    # run eval
    score=$(CUDA_VISIBLE_DEVICES=$GPU_ID python eval_scripts/eval_nf.py \
        --input "$OUT_DIR" \
        --suffix "$SUFFIX_ARG" \
        --device cuda \
        --metrics "$METRICS_CSV"
    )

    echo "----------------------------------------"
    echo "[GPU $GPU_ID][$METHOD_IN][$DATASET][$NAME]"
    echo "$score"
    echo
}

throttle() {
  while (( $(jobs -pr | wc -l) >= NGPU )); do
    sleep 0.5
  done
}

launch_eval() {
  local METHOD="$1"
  local NAME="$2"
  local DATASET="$3"
  local GPU_ID="$4"
  run_eval_iqa "$METHOD" "$NAME" "$DATASET" "$GPU_ID" &
}

# TASK FORMAT:
#   "METHOD|NAME|DATASET"
TASKS=(
  # pix2pix_turbo examples (NAME is exp/.../light)
  "pix2pix_turbo|exp_1_1_warped_128_eyes/golden_sunlight_1|VITON/test"
  "pix2pix_turbo|exp_1_1_warped_128_eyes/noon_sunlight_1|VITON/test"
  "pix2pix_turbo|exp_1_1_warped_128_eyes/moonlight_1|VITON/test"
  "pix2pix_turbo|exp_1_1_warped_128_eyes/foggy_1|VITON/test"

  "pix2pix_turbo|exp_1_10_1/golden_sunlight_1|VITON/test"
  "pix2pix_turbo|exp_1_10_1/noon_sunlight_1|VITON/test"
  "pix2pix_turbo|exp_1_10_1/moonlight_1|VITON/test"
  "pix2pix_turbo|exp_1_10_1/foggy_1|VITON/test"

  "pix2pix_turbo|exp_1_10_1_warped_128_eyes/golden_sunlight_1|VITON/test"
  "pix2pix_turbo|exp_1_10_1_exp_1_10_1_v2_merged_warped_128_eyes/noon_sunlight_1|VITON/test"
    "pix2pix_turbo|exp_1_10_1_warped_128_eyes/moonlight_1|VITON/test"
    "pix2pix_turbo|exp_1_10_1_warped_128_eyes/foggy_1|VITON/test"




  "pix2pix_turbo|exp_1_1_warped_128_eyes/golden_sunlight_1|street_tryon/validation"
  "pix2pix_turbo|exp_1_1_warped_128_eyes/noon_sunlight_1|street_tryon/validation"
  "pix2pix_turbo|exp_1_1_warped_128_eyes/moonlight_1|street_tryon/validation"
  "pix2pix_turbo|exp_1_1_warped_128_eyes/foggy_1|street_tryon/validation"

  "pix2pix_turbo|exp_1_10_1/golden_sunlight_1|street_tryon/validation"
  "pix2pix_turbo|exp_1_10_1/noon_sunlight_1|street_tryon/validation"
  "pix2pix_turbo|exp_1_10_1/moonlight_1|street_tryon/validation"
  "pix2pix_turbo|exp_1_10_1/foggy_1|street_tryon/validation"

  "pix2pix_turbo|exp_1_10_1_warped_128_eyes/golden_sunlight_1|street_tryon/validation"
  "pix2pix_turbo|exp_1_10_1_exp_1_10_1_v2_merged_warped_128_eyes/noon_sunlight_1|street_tryon/validation"
    "pix2pix_turbo|exp_1_10_1_warped_128_eyes/moonlight_1|street_tryon/validation"
    "pix2pix_turbo|exp_1_10_1_warped_128_eyes/foggy_1|street_tryon/validation"





  # iclight
  "iclight|golden_sunlight_1|VITON/test"
  "iclight|noon_sunlight_1|VITON/test"
  "iclight|moonlight_1|VITON/test"
  "iclight|foggy_1|VITON/test"

  "iclight|golden_sunlight_1|street_tryon/validation"
  "iclight|noon_sunlight_1|street_tryon/validation"
  "iclight|moonlight_1|street_tryon/validation"
  "iclight|foggy_1|street_tryon/validation"



  # dreamlight
  "dreamlight|golden_sunlight_1|VITON/test"
  "dreamlight|noon_sunlight_1|VITON/test"
  "dreamlight|moonlight_1|VITON/test"
  "dreamlight|foggy_1|VITON/test"

  "dreamlight|golden_sunlight_1|street_tryon/validation"
  "dreamlight|noon_sunlight_1|street_tryon/validation"
  "dreamlight|moonlight_1|street_tryon/validation"
  "dreamlight|foggy_1|street_tryon/validation"
)

gpu=0
for t in "${TASKS[@]}"; do
  METHOD="${t%%|*}"
  rest="${t#*|}"
  NAME="${rest%%|*}"
  DATASET="${rest#*|}"

  throttle
  launch_eval "$METHOD" "$NAME" "$DATASET" "$gpu"
  gpu=$(( (gpu + 1) % NGPU ))
done

wait
echo "All evals done."