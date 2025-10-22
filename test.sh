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

    OUT_DIR="/ssd1/shenzhen/img2img-turbo/output/pix2pix_turbo/${EXP}/result_A"

    # NOTE: use model_19001.pkl (or 18501.pkl) for inferencec 
    python src/inference_paired_folder.py \
        --model_path "/ssd1/shenzhen/img2img-turbo/output/pix2pix_turbo/${EXP}/checkpoints/model_18501.pkl" \
        --input_dir "/ssd1/shenzhen/relighting/${DATASET_NAME}/test_A" \
        --prompt "$PROMPT" \
        --output_dir "$OUT_DIR"

    echo "Evaluating LPIPS / SSIM / PSNR ..."
    python eval_relight_metrics.py \
        --gt_dir  "/ssd1/shenzhen/relighting/${DATASET_NAME}/test_B" \
        --pred_dir "$OUT_DIR"
}

# run_inference "exp_10_16_warped_512/candlelight_1"
# run_inference "exp_10_9_warped_512/candlelight_1"
run_inference "exp_10_16/candlelight_1"