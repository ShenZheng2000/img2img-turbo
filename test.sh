#!/bin/bash
# ===============================================================
# Universal inference launcher for Pix2Pix-Turbo + warp-unwarp
# Auto-detects prompt and bandwidth from experiment name
# ===============================================================

run_inference() {
    EXP="$1"        # e.g. exp_10_16_warped_128_eyes/candlelight_1
    DATASET="$2"    # e.g. dataset_with_garment_bigface_100
    # MAX_STEPS=${3:-18501}    # optional arg; default = 18501

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
    # MODEL_PATH="output/pix2pix_turbo/${EXP}/checkpoints/model_${MAX_STEPS}.pkl"
    CKPT_DIR="output/pix2pix_turbo/${EXP}/checkpoints"
    MODEL_PATH=$(ls -1 ${CKPT_DIR}/model_*.pkl | sort -V | tail -n 1)
    INPUT_DIR="/home/shenzhen/Datasets/${DATASET}"
    OUT_DIR="output/pix2pix_turbo/${EXP}/result_A_${DATASET}"

    mkdir -p "$OUT_DIR"

    # -------------------------------
    # Run inference
    # -------------------------------
    echo "Launching inference on $EXP ..."
    CUDA_VISIBLE_DEVICES=0 python src/inference_paired_folder.py \
        --model_path "$MODEL_PATH" \
        --input_dir "$INPUT_DIR" \
        --prompt "$PROMPT" \
        --output_dir "$OUT_DIR" \
        --bw "$BW" \
        $EYE_FLAG
}
# ===============================================================
# Examples
# ===============================================================

# relight_type=golden_sunlight_1
# relight_type=moonlight_1
relight_type=foggy_1
# relight_type=noon_sunlight_1
# relight_type=dusk_backlit_1

# =====================================
# dataset_with_garment_bigface_100
# =====================================
# run_inference "exp_10_16/${relight_type}" "dataset_with_garment_bigface_100"
# run_inference "exp_10_16_warped_128_eyes/${relight_type}" "dataset_with_garment_bigface_100"
# run_inference "exp_10_16_exp_12_7_merged/${relight_type}" "dataset_with_garment_bigface_100"
# run_inference "exp_10_16_exp_12_7_merged_warped_128_eyes/${relight_type}" "dataset_with_garment_bigface_100"

# run_inference "exp_10_16_v2/${relight_type}" "dataset_with_garment_bigface_100"
# run_inference "exp_10_16_v2_warped_128_eyes/${relight_type}" "dataset_with_garment_bigface_100"

# run_inference "exp_10_17/${relight_type}" "dataset_with_garment_bigface_100"
# run_inference "exp_10_17_warped_128_eyes/${relight_type}" "dataset_with_garment_bigface_100"

# =====================================
# VITON/test
# =====================================
# run_inference "exp_10_16/${relight_type}" "VITON/test"
# run_inference "exp_10_16_warped_128_eyes/${relight_type}" "VITON/test"
# run_inference "exp_10_16_exp_12_7_merged/${relight_type}" "VITON/test"
# run_inference "exp_10_16_exp_12_7_merged_warped_128_eyes/${relight_type}" "VITON/test"

# run_inference "exp_10_16_v2/${relight_type}" "VITON/test"
# run_inference "exp_10_16_v2_warped_128_eyes/${relight_type}" "VITON/test"

# run_inference "exp_1_1_warped_128_eyes/${relight_type}" "VITON/test"
# run_inference "exp_1_1/${relight_type}" "VITON/test"

run_inference "exp_1_10_1_warped_128_eyes/${relight_type}" "VITON/test"
# run_inference "exp_1_10_1/${relight_type}" "VITON/test"

# =====================================
# street_tryon/validation
# =====================================
# run_inference "exp_10_16/${relight_type}" "street_tryon/validation"
# run_inference "exp_10_16_warped_128_eyes/${relight_type}" "street_tryon/validation"
# run_inference "exp_10_16_exp_12_7_merged/${relight_type}" "street_tryon/validation"
# run_inference "exp_10_16_exp_12_7_merged_warped_128_eyes/${relight_type}" "street_tryon/validation"

# run_inference "exp_10_16_v2/${relight_type}" "street_tryon/validation"
# run_inference "exp_10_16_v2_warped_128_eyes/${relight_type}" "street_tryon/validation"

# run_inference "exp_1_6_warped_128_eyes/${relight_type}" "street_tryon/validation"
# run_inference "exp_1_6/${relight_type}" "street_tryon/validation"

# run_inference "exp_1_6_warped_128_eyes/${relight_type}" "street_tryon/validation_big_sample_100"
# run_inference "exp_1_6/${relight_type}" "street_tryon/validation_big_sample_100"

# =====================================
# dataset_with_garment_bigface_start_100_end_200 (no need to test the base model)
# =====================================
# run_inference "exp_10_16_exp_12_7_merged/${relight_type}" "dataset_with_garment_bigface_start_100_end_200"
# run_inference "exp_10_16_exp_12_7_merged_warped_128_eyes/${relight_type}" "dataset_with_garment_bigface_start_100_end_200"