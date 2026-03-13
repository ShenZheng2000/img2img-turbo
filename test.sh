#!/bin/bash
# ===============================================================
# Universal inference launcher for Pix2Pix-Turbo + warp-unwarp
# Auto-detects prompt and bandwidth from experiment name
# ===============================================================

run_inference() {
    EXP="$1"
    DATASET="$2"
    GPU_ID="${3:-0}"

    # NEW: optional crop_resize_size (e.g. 512). Default: empty = do nothing.
    CROP_RESIZE_SIZE="${4:-}"

    TARGET_SIZE="${5:-784}"

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

    CENTER_CROP_FLAG=""
    if [[ "$EXP" == *"drive"* ]]; then
        CENTER_CROP_FLAG="--center_crop"
        echo "Detected: center_crop = True (driving experiment)"
    fi

    # -------------------------------
    # Auto-detect YOLOWorld usage
    # -------------------------------
    YOLO_FLAG=""
    if [[ "$EXP" == *"drive"* && "$EXP" == *"warped"* ]]; then
        YOLO_FLAG="--use-yoloworld"
        echo "Detected: use_yoloworld = True (drive + warped)"
    fi

    # -------------------------------
    # NEW: crop_resize flag + OUT_DIR suffix
    # -------------------------------
    CROP_RESIZE_FLAG=""
    OUT_SUFFIX=""
    if [[ -n "$CROP_RESIZE_SIZE" ]]; then
        CROP_RESIZE_FLAG="--crop_resize_size $CROP_RESIZE_SIZE"
        OUT_SUFFIX="_cr${CROP_RESIZE_SIZE}"
        echo "Detected: crop_resize_size = $CROP_RESIZE_SIZE"
    fi


    # -------------------------------
    # target_size flag + optional OUT_DIR suffix
    # -------------------------------
    TARGET_FLAG="--target_size $TARGET_SIZE"

    # Only append suffix if user explicitly passed arg5
    if [[ -n "$5" ]]; then
        OUT_SUFFIX="${OUT_SUFFIX}_ts${TARGET_SIZE}"
        echo "Detected: target_size override = $TARGET_SIZE"
    else
        echo "Using default target_size = 784"
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

    INPUT_DIR="/ssd0/shenzhen/Datasets/${DATASET}"
    OUT_DIR="output/pix2pix_turbo/${EXP}/${DATASET}${OUT_SUFFIX}"

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
        $EYE_FLAG \
        $CENTER_CROP_FLAG \
        $YOLO_FLAG \
        $CROP_RESIZE_FLAG \
        $TARGET_FLAG
}

# TODO: fix code bug, now we are putting image/ fodler inside image/ foldeer!!!!!!!!

# # =====================================
# # roadwork driving images (NOTE: use 512x512 as crop_resize_size for better vis differences)
# # =====================================


# run_inference "2_24_drive_v2_warped_128/golden_sunlight_1" "depth/workzone_segm/boston" 1 512 1080

# run_inference "2_24_drive_v2/golden_sunlight_1" "depth/workzone_segm/boston" 1 512 1080
# run_inference "2_24_drive_v2/golden_sunlight_1" "depth/workzone_segm/boston" 1 512
# run_inference "2_24_drive_v2/foggy_1" "depth/workzone_segm/boston" 2 512
# run_inference "2_24_drive_v2_warped_128/golden_sunlight_1" "depth/workzone_segm/boston" 0 512
# run_inference "2_24_drive_v2_warped_128/foggy_1" "depth/workzone_segm/boston" 2 512

# # =====================================
# # VITON/test
# # =====================================
# run_inference "exp_1_10_1_warped_128_eyes/golden_sunlight_1" "VITON/test" 0
# run_inference "exp_1_10_1_warped_128_eyes/moonlight_1" "VITON/test" 1
# run_inference "exp_1_10_1_warped_128_eyes/foggy_1" "VITON/test" 2
# run_inference "exp_1_10_1_exp_1_10_1_v2_merged_warped_128_eyes/noon_sunlight_1" "VITON/test" 3

# run_inference "exp_1_10_1/noon_sunlight_1" "VITON/test_sample_100" 7
# run_inference "exp_1_10_1/golden_sunlight_1" "VITON/test"
# run_inference "exp_1_10_1/moonlight_1" "VITON/test" 6
# run_inference "exp_1_10_1/foggy_1" "VITON/test"
# run_inference "exp_1_10_1/noon_sunlight_1" "VITON/test"

# TODO: rerun this later! 
# run_inference "exp_1_1_warped_128_eyes/golden_sunlight_1" "VITON/test"
# run_inference "exp_1_1_warped_128_eyes/moonlight_1" "VITON/test"
# run_inference "exp_1_1_warped_128_eyes/foggy_1" "VITON/test"
# run_inference "exp_1_1_warped_128_eyes/noon_sunlight_1" "VITON/test"


# # # =====================================
# # # street_tryon/validation
# # # =====================================
# run_inference "exp_1_10_1_warped_128_eyes/golden_sunlight_1" "street_tryon/validation"
# run_inference "exp_1_10_1_warped_128_eyes/moonlight_1" "street_tryon/validation"
# run_inference "exp_1_10_1_warped_128_eyes/foggy_1" "street_tryon/validation"
# run_inference "exp_1_10_1_exp_1_10_1_v2_merged_warped_128_eyes/noon_sunlight_1" "street_tryon/validation"

# run_inference "exp_1_10_1/golden_sunlight_1" "street_tryon/validation"
# run_inference "exp_1_10_1/moonlight_1" "street_tryon/validation"
# run_inference "exp_1_10_1/foggy_1" "street_tryon/validation"
# run_inference "exp_1_10_1/noon_sunlight_1" "street_tryon/validation"

# run_inference "exp_1_1_warped_128_eyes/golden_sunlight_1" "street_tryon/validation"
# run_inference "exp_1_1_warped_128_eyes/moonlight_1" "street_tryon/validation"
# run_inference "exp_1_1_warped_128_eyes/foggy_1" "street_tryon/validation"
# run_inference "exp_1_1_warped_128_eyes/noon_sunlight_1" "street_tryon/validation"