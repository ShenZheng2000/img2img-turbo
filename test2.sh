#!/bin/bash
# ===============================================================
# Universal inference launcher for Pix2Pix-Turbo + warp-unwarp
# Auto-detects prompt and bandwidth from experiment name
# ===============================================================

run_inference() {
    EXP="$1"        # e.g. exp_10_16_warped_128/candlelight_1
    DATASET="$2"    # e.g. dataset_with_garment_bigface_100

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
    # Auto-detect lighting prompt
    # -------------------------------
    if [[ "$EXP" == *"candlelight_1"* ]]; then
        PROMPT="Relit with warm candlelight in a dimly lit indoor setting, casting soft, flickering shadows and enveloping the subject in golden-orange tones to create a cozy, nostalgic mood."
    elif [[ "$EXP" == *"noon_sunlight_1"* ]]; then
        PROMPT="Relit with bright noon sunlight in a clear outdoor setting, casting soft natural shadows and surrounding the subject in crisp white light to create a clean, vibrant daytime mood."
    else
        echo "Unknown lighting type in experiment name: $EXP"
        exit 1
    fi
    echo "Using prompt: ${PROMPT:0:60}..."

    # -------------------------------
    # Setup paths
    # -------------------------------
    MODEL_PATH="output/pix2pix_turbo/${EXP}/checkpoints/model_18501.pkl"
    INPUT_DIR="/home/shenzhen/Datasets/${DATASET}"
    OUT_DIR="output/pix2pix_turbo/${EXP}/result_A_${DATASET}"

    mkdir -p "$OUT_DIR"

    # -------------------------------
    # Run inference
    # -------------------------------
    echo "Launching inference on $EXP ..."
    CUDA_VISIBLE_DEVICES=7 python src/inference_paired_folder.py \
        --model_path "$MODEL_PATH" \
        --input_dir "$INPUT_DIR" \
        --prompt "$PROMPT" \
        --output_dir "$OUT_DIR" \
        --bw "$BW"
}

# ===============================================================
# Examples
# ===============================================================

run_inference "exp_10_16/candlelight_1" "dataset_with_garment_bigface_100"
# run_inference "exp_10_16_warped_128/candlelight_1" "dataset_with_garment_bigface_100"
# run_inference "exp_10_16_warped_512/candlelight_1" "dataset_with_garment_bigface_100"

# run_inference "exp_10_16/noon_sunlight_1" "dataset_with_garment_bigface_100"
# run_inference "exp_10_16_warped_512/noon_sunlight_1" "dataset_with_garment_bigface_100"