#!/bin/bash
# ===============================================================
# Universal inference launcher for Pix2Pix-Turbo + warp-unwarp
# Auto-detects prompt and bandwidth from experiment name
# ===============================================================

run_inference() {
    EXP="$1"        # e.g. exp_10_16_warped_128_eyes/candlelight_1
    DATASET="$2"    # e.g. dataset_with_garment_bigface_100
    GPU_ID="${3:-0}"  # e.g. 0, 1, 2... (Defaults to 0 if not provided)

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
    # Auto-detect include_eyes
    # -------------------------------
    EYE_FLAG=""
    if [[ "$EXP" == *"_eyes"* ]]; then
        EYE_FLAG="--include-eyes"
        echo "Detected: include-eyes = True"
    fi

    # -------------------------------
    # Auto-detect lighting prompt
    # -------------------------------
    if [[ "$EXP" == *"noon_sunlight_1"* ]]; then
        PROMPT="Relit with bright noon sunlight in a clear outdoor setting, casting soft natural shadows and surrounding the subject in crisp white light to create a clean, vibrant daytime mood."

    elif [[ "$EXP" == *"golden_sunlight_1"* ]]; then
        PROMPT="Relit with warm golden sunlight during the late afternoon, casting gentle directional shadows and surrounding the subject in soft amber tones to create a calm, radiant mood."

    elif [[ "$EXP" == *"foggy_1"* ]]; then
        PROMPT="Relit with dense fog in a muted outdoor setting, casting soft diffused shadows and surrounding the subject in pale gray light to create a quiet, atmospheric mood."

    elif [[ "$EXP" == *"moonlight_1"* ]]; then
        PROMPT="Relit with cold moonlight in a minimalist nighttime scene, casting crisp soft shadows and bathing the subject in icy blue highlights to create a tranquil, distant mood."

    else
        echo "Unknown lighting type in experiment name: $EXP"
        exit 1
    fi

    echo "Using prompt: ${PROMPT:0:60}..."

    # -------------------------------
    # Setup paths
    # -------------------------------
    CKPT_DIR="output/pix2pix_turbo/${EXP}/checkpoints"
    MODEL_PATH=$(ls -1 ${CKPT_DIR}/model_*.pkl | sort -V | tail -n 1)
    INPUT_DIR="/home/shenzhen/Datasets/${DATASET}"
    OUT_DIR="output/pix2pix_turbo/${EXP}/result_A_${DATASET}"

    mkdir -p "$OUT_DIR"

    # -------------------------------
    # Run inference
    # -------------------------------
    echo "[GPU $GPU_ID] Launching inference on $EXP ..."

    CUDA_VISIBLE_DEVICES=$GPU_ID python src/inference_paired_folder.py \
        --model_path "$MODEL_PATH" \
        --input_dir "$INPUT_DIR" \
        --prompt "$PROMPT" \
        --output_dir "$OUT_DIR" \
        --bw "$BW" \
        $EYE_FLAG
}

# =====================================
# Model Clarifications
# =====================================
# exp_1_10_1_warped_128_eyes                        -> final model
# exp_1_10_1_exp_1_10_1_v2_merged_warped_128_eyes   -> final model (train 2x time for better noon sunlight results)
# exp_1_1_warped_128_eyes                           -> no bg prompt injection
# exp_1_10_1                                        -> no warp-unwarp


# # # =====================================
# # # street_tryon/validation
# # # =====================================
run_inference "exp_1_10_1_warped_128_eyes/golden_sunlight_1" "street_tryon/validation_big_sample_100" 1
run_inference "exp_1_10_1_warped_128_eyes/moonlight_1" "street_tryon/validation_big_sample_100" 1
run_inference "exp_1_10_1_warped_128_eyes/foggy_1" "street_tryon/validation_big_sample_100" 1
run_inference "exp_1_10_1_warped_128_eyes/noon_sunlight_1" "street_tryon/validation_big_sample_100" 1

# NOTE: only noon sunlight needed for this one 
run_inference "exp_1_10_1_exp_1_10_1_v2_merged_warped_128_eyes/noon_sunlight_1" "street_tryon/validation_big_sample_100" 1

run_inference "exp_1_10_1/golden_sunlight_1" "street_tryon/validation_big_sample_100" 1
run_inference "exp_1_10_1/moonlight_1" "street_tryon/validation_big_sample_100" 1
run_inference "exp_1_10_1/foggy_1" "street_tryon/validation_big_sample_100" 1
run_inference "exp_1_10_1/noon_sunlight_1" "street_tryon/validation_big_sample_100" 1

run_inference "exp_1_1_warped_128_eyes/golden_sunlight_1" "street_tryon/validation_big_sample_100" 1
run_inference "exp_1_1_warped_128_eyes/moonlight_1" "street_tryon/validation_big_sample_100" 1
run_inference "exp_1_1_warped_128_eyes/foggy_1" "street_tryon/validation_big_sample_100" 1
run_inference "exp_1_1_warped_128_eyes/noon_sunlight_1" "street_tryon/validation_big_sample_100" 1