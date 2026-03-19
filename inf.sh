run_infer() {
    exp_name="$1"
    relight_type="$2"
    data_name="$3"
    gpu_id="${4:-0}"

    # NOTE: hardcode here! 
    input_root=/ssd0/shenzhen/Datasets
    output_root=output/pix2pix_turbo

    case "$relight_type" in
      noon_sunlight_1)
        prompt="Relit with bright noon sunlight in a clear outdoor setting, casting soft natural shadows and surrounding the subject in crisp white light to create a clean, vibrant daytime mood."
        ;;
      golden_sunlight_1)
        prompt="Relit with warm golden sunlight during the late afternoon, casting gentle directional shadows and surrounding the subject in soft amber tones to create a calm, radiant mood."
        ;;
      foggy_1)
        prompt="Relit with dense fog in a muted outdoor setting, casting soft diffused shadows and surrounding the subject in pale gray light to create a quiet, atmospheric mood."
        ;;
      moonlight_1)
        prompt="Relit with cold moonlight in a minimalist nighttime scene, casting crisp soft shadows and bathing the subject in icy blue highlights to create a tranquil, distant mood."
        ;;
      *)
        echo "Unknown relight_type: $relight_type"
        return 1
        ;;
    esac

    # ✅ NEW: auto pick latest checkpoint
    ckpt_dir=${output_root}/${exp_name}/${relight_type}/checkpoints
    model_path=$(ls -1 ${ckpt_dir}/model_*.pkl | sort -V | tail -n 1)

    CUDA_VISIBLE_DEVICES=$gpu_id python src/inference_paired_folder.py \
        --exp_config configs/${exp_name}.yaml \
        --input_dir ${input_root}/${data_name} \
        --output_dir ${output_root}/${exp_name}/${relight_type}/${data_name} \
        --prompt "$prompt" \
        --model_path "$model_path"
}

# Example usage: 
# run_infer 2_24_drive_v2_warped_128 golden_sunlight_1 depth/workzone_segm/boston
# run_infer exp_1_10_1_warped_128_eyes golden_sunlight_1 VITON/test_sample_100 1
# run_infer exp_1_10_1 golden_sunlight_1 VITON/test_sample_100
