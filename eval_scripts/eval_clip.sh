## Installation
# conda create -n clip_metric python=3.10 -y
# conda activate clip_metric
# pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu121
# pip install torchmetrics
# pip install "transformers<5" "tokenizers<0.20"
# pip install open_clip_torch
# pip install image-reward
# pip install git+https://github.com/openai/CLIP.git

set -e

NGPU="${NGPU:-8}"   # run up to NGPU evals at once (GPU 0..NGPU-1)

run_eval_clip() {
    METHOD_IN="$1"   # pix2pix_turbo | iclight | dreamlight
    NAME="$2"        # pix2pix_turbo: exp_xxx/.../golden_sunlight_1 ; iclight/dreamlight: just golden_sunlight_1
    DATASET="$3"     # e.g. VITON/test
    GPU_ID="${4:-0}"


    # infer LIGHT + PROMPT from NAME (works for both exp paths and light-only)
    if [[ "$NAME" == *"noon_sunlight_1"* ]]; then
        PROMPT="Relit with bright noon sunlight in a clear outdoor setting, casting soft natural shadows and surrounding the subject in crisp white light to create a clean, vibrant daytime mood."
        LIGHT="noon_sunlight_1"
    elif [[ "$NAME" == *"golden_sunlight_1"* ]]; then
        PROMPT="Relit with warm golden sunlight during the late afternoon, casting gentle directional shadows and surrounding the subject in soft amber tones to create a calm, radiant mood."
        LIGHT="golden_sunlight_1"
    elif [[ "$NAME" == *"foggy_1"* ]]; then
        PROMPT="Relit with dense fog in a muted outdoor setting, casting soft diffused shadows and surrounding the subject in pale gray light to create a quiet, atmospheric mood."
        LIGHT="foggy_1"
    elif [[ "$NAME" == *"moonlight_1"* ]]; then
        PROMPT="Relit with cold moonlight in a minimalist nighttime scene, casting crisp soft shadows and bathing the subject in icy blue highlights to create a tranquil, distant mood."
        LIGHT="moonlight_1"
    else
        echo "Unknown lighting type in name: $NAME"
        exit 1
    fi

    FINAL_PROMPT=${PROMPT}

    # output folder by method
    if [[ "$METHOD_IN" == "pix2ppix2pix_turboix_turbo" ]]; then
        OUT_DIR="output//${NAME}/${DATASET}/image"
    elif [[ "$METHOD_IN" == "iclight" ]]; then
        OUT_DIR="/home/shenzhen/Relight_Projects/IC-Light/outputs/512x512/${LIGHT}/${DATASET}/image"
    elif [[ "$METHOD_IN" == "dreamlight" ]]; then
        OUT_DIR="/home/shenzhen/Relight_Projects/DreamLight/outputs/512x512/${LIGHT}/${DATASET}/image"
    else
        echo "Unknown METHOD: $METHOD_IN"
        exit 1
    fi

    # NOTE: assume pix2pix_turbo outputs _warp_relight_unwarp.png as surfix
    SUFFIX_ARG="_warp_relight_unwarp.png"
    if [[ "$METHOD_IN" != "pix2pix_turbo" ]]; then
    SUFFIX_ARG=""
    fi

    score=$(CUDA_VISIBLE_DEVICES=$GPU_ID python eval_scripts/eval_clip.py \
        --input "$OUT_DIR" \
        --caption "$FINAL_PROMPT" \
        --suffix "$SUFFIX_ARG" \
        --metric "imagereward"
        )

    echo "----------------------------------------"
    echo "[GPU $GPU_ID][$DATASET][$NAME][$METHOD_IN]"
    echo "DIR: $OUT_DIR"
    echo "Score: $score"
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
  run_eval_clip "$METHOD" "$NAME" "$DATASET" "$GPU_ID" &
}

# TASK FORMAT:
#   "METHOD|NAME|DATASET"
#   - pix2pix_turbo: NAME is exp path (many exps)
#   - iclight/dreamlight: NAME is just the light (few entries)


TASKS=(
  # pix2pix_turbo
#   "pix2pix_turbo|exp_1_1_warped_128_eyes/golden_sunlight_1|VITON/test"
#   "pix2pix_turbo|exp_1_1_warped_128_eyes/noon_sunlight_1|VITON/test"
#   "pix2pix_turbo|exp_1_1_warped_128_eyes/moonlight_1|VITON/test"
#   "pix2pix_turbo|exp_1_1_warped_128_eyes/foggy_1|VITON/test"

#   "pix2pix_turbo|exp_1_10_1/golden_sunlight_1|VITON/test"
#   "pix2pix_turbo|exp_1_10_1/noon_sunlight_1|VITON/test"
#   "pix2pix_turbo|exp_1_10_1/moonlight_1|VITON/test"
#   "pix2pix_turbo|exp_1_10_1/foggy_1|VITON/test"

#   "pix2pix_turbo|exp_1_10_1_warped_128_eyes/golden_sunlight_1|VITON/test"
#   "pix2pix_turbo|exp_1_10_1_exp_1_10_1_v2_merged_warped_128_eyes/noon_sunlight_1|VITON/test"
#   "pix2pix_turbo|exp_1_10_1_warped_128_eyes/moonlight_1|VITON/test"
#   "pix2pix_turbo|exp_1_10_1_warped_128_eyes/foggy_1|VITON/test"




#   "pix2pix_turbo|exp_1_1_warped_128_eyes/golden_sunlight_1|street_tryon/validation"
#   "pix2pix_turbo|exp_1_1_warped_128_eyes/noon_sunlight_1|street_tryon/validation"
#   "pix2pix_turbo|exp_1_1_warped_128_eyes/moonlight_1|street_tryon/validation"
#   "pix2pix_turbo|exp_1_1_warped_128_eyes/foggy_1|street_tryon/validation"

#   "pix2pix_turbo|exp_1_10_1/golden_sunlight_1|street_tryon/validation"
#   "pix2pix_turbo|exp_1_10_1/noon_sunlight_1|street_tryon/validation"
#   "pix2pix_turbo|exp_1_10_1/moonlight_1|street_tryon/validation"
#   "pix2pix_turbo|exp_1_10_1/foggy_1|street_tryon/validation"

#   "pix2pix_turbo|exp_1_10_1_warped_128_eyes/golden_sunlight_1|street_tryon/validation"
#   "pix2pix_turbo|exp_1_10_1_exp_1_10_1_v2_merged_warped_128_eyes/noon_sunlight_1|street_tryon/validation"
#   "pix2pix_turbo|exp_1_10_1_warped_128_eyes/moonlight_1|street_tryon/validation"
#   "pix2pix_turbo|exp_1_10_1_warped_128_eyes/foggy_1|street_tryon/validation"




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
