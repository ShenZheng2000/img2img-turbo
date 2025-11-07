#!/bin/bash
# ===============================================================
# Universal inference launcher for Pix2Pix-Turbo + warp-unwarp
# Auto-detects prompt and bandwidth from experiment name
# ===============================================================

# NOTE: text prompt has little (almost no impact) during test-time. 
# TODO: Current pipeline is: crop_to_fg -> resize to (target_size, target_size) -> warp -> relight -> unwarp -> resize_longest_side_to_target_size.
# Need to think if this is good? 

run_inference() {
    EXP="$1"        # e.g. exp_10_16_warped_128/candlelight_1
    DATASET="$2"    # e.g. dataset_with_garment_bigface_100
    INTENSITY="${3:-medium}"  # light / medium / strong (default = medium)

    # -------------------------------
    # Auto-detect bandwidth (BW)
    # -------------------------------
    if [[ "$EXP" == *"warped_"* ]]; then
        BW=$(echo "$EXP" | sed -E 's/.*warped_([0-9]+).*/\1/')
    else
        BW=0
    fi
    echo "Detected bandwidth = $BW"

    # -------------------------------
    # Select lighting prompt by type + intensity
    # -------------------------------
    if [[ "$EXP" == *"candlelight_1"* ]]; then
        declare -A PROMPTS=(
            [light]="Relit with a very gentle candlelight glow, faintly illuminating the subject with subtle amber tones and minimal shadow contrast."
            [medium]="Relit with warm candlelight in a dimly lit indoor setting, casting soft, flickering shadows and enveloping the subject in golden-orange tones to create a cozy, nostalgic mood."
            [strong]="Relit with bright, close candlelight that casts strong golden highlights and deep soft shadows, creating a vivid and dramatic warmth."
        )
    elif [[ "$EXP" == *"noon_sunlight_1"* ]]; then
        declare -A PROMPTS=(
            [light]="Relit with soft midday sunlight filtered through light haze, giving gentle contrast and a slightly muted brightness."
            [medium]="Relit with bright noon sunlight in a clear outdoor setting, casting soft natural shadows and surrounding the subject in crisp white light to create a clean, vibrant daytime mood."
            [strong]="Relit with strong, direct noon sunlight producing bright highlights and high contrast, evoking a clear, radiant outdoor atmosphere."
        )
    else
        echo "Unknown lighting type in experiment name: $EXP"
        exit 1
    fi

    PROMPT="${PROMPTS[$INTENSITY]}"
    echo "Using intensity '$INTENSITY': ${PROMPT:0:60}..."

    # -------------------------------
    # Setup paths
    # -------------------------------
    MODEL_PATH="output/pix2pix_turbo/${EXP}/checkpoints/model_18501.pkl"
    INPUT_DIR="/home/shenzhen/Datasets/${DATASET}"
    OUT_DIR="output/pix2pix_turbo/${EXP}/result_A_${DATASET}_${INTENSITY}"

    mkdir -p "$OUT_DIR"

    # -------------------------------
    # Run inference
    # -------------------------------
    echo "Launching inference on $EXP ($INTENSITY intensity)..."
    CUDA_VISIBLE_DEVICES=3 python src/inference_paired_folder.py \
        --model_path "$MODEL_PATH" \
        --input_dir "$INPUT_DIR" \
        --prompt "$PROMPT" \
        --output_dir "$OUT_DIR" \
        --bw "$BW"
}

# ===============================================================
# Examples
# ===============================================================

# NOTE: now using one body mask and skip folders with multiple body masks

run_inference "exp_10_16/candlelight_1" "dataset_with_garment_bigface_100" "light"
run_inference "exp_10_16/candlelight_1" "dataset_with_garment_bigface_100" "medium"
run_inference "exp_10_16/candlelight_1" "dataset_with_garment_bigface_100" "strong"

run_inference "exp_10_16_warped_512/candlelight_1" "dataset_with_garment_bigface_100" "light"
run_inference "exp_10_16_warped_512/candlelight_1" "dataset_with_garment_bigface_100" "medium"
run_inference "exp_10_16_warped_512/candlelight_1" "dataset_with_garment_bigface_100" "strong"

run_inference "exp_10_16_warped_128/candlelight_1" "dataset_with_garment_bigface_100" "light"
run_inference "exp_10_16_warped_128/candlelight_1" "dataset_with_garment_bigface_100" "medium"
run_inference "exp_10_16_warped_128/candlelight_1" "dataset_with_garment_bigface_100" "strong"


# run_inference "exp_10_16_warped_128/noon_sunlight_1" "dataset_with_garment_bigface_100" # TODO: this need a trained model 