run_inference() {
    DATASET_NAME=$1   # e.g. exp_10_9/noon_sunlight_1
    EXP="$DATASET_NAME"

    # --- prompt routing based on substring (NOTE: hardcode for now, since very few prompts) ---
    if [[ "$DATASET_NAME" == *"candlelight_1"* ]]; then
        PROMPT="Relit with warm candlelight in a dimly lit indoor setting, casting soft, flickering shadows and enveloping the subject in golden-orange tones to create a cozy, nostalgic mood."
    elif [[ "$DATASET_NAME" == *"noon_sunlight_1"* ]]; then
        PROMPT="Relit with bright noon sunlight in a clear outdoor setting, casting soft natural shadows and surrounding the subject in crisp white light to create a clean, vibrant daytime mood."
    else
        echo "❌ Unknown dataset lighting type: ${DATASET_NAME}"
        exit 1
    fi

    OUT_DIR="output/pix2pix_turbo/${EXP}/result_A"

    # # NOTE: use model_19001.pkl (or 18501.pkl) for inferencec 
    python src/inference_paired_folder.py \
        --model_path "output/pix2pix_turbo/${EXP}/checkpoints/model_18501.pkl" \
        --input_dir "/data3/shenzhen/Datasets/relighting/${DATASET_NAME}/test_A" \
        --prompt "$PROMPT" \
        --output_dir "$OUT_DIR"

    echo "Evaluating LPIPS / SSIM / PSNR ..."
    python eval_relight_metrics.py \
        --gt_dir  "/data3/shenzhen/Datasets/relighting/${DATASET_NAME}/test_B" \
        --pred_dir "$OUT_DIR"
}

# TODO: inference on source REAL images.
# run_inference "exp_10_16_warped_512/noon_sunlight_1"
# run_inference "exp_10_16/noon_sunlight_1"

# run_inference "exp_10_16_warped_192/candlelight_1"
# run_inference "exp_10_16_warped_320/candlelight_1"
# run_inference "exp_10_16_warped_384/candlelight_1"
# run_inference "exp_10_16_warped_128/candlelight_1"
# run_inference "exp_10_16_warped_256/candlelight_1"
# run_inference "exp_10_16_warped_512/candlelight_1"
# run_inference "exp_10_16/candlelight_1"
# run_inference "exp_10_9_warped_512/candlelight_1"
# run_inference "exp_10_9/candlelight_1"