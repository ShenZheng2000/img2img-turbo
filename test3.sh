# # # google street view -> boreas snowy
# python src/inference_unpaired_folder.py \
#     --prompt "driving in heavy snow" \
#     --direction "a2b" \
#     --model_path "output/cyclegan_turbo/boreas_1_26_resize_286_randomcrop_256x256_hflip/checkpoints/model_25001.pkl" \
#     --image_prep "no_resize" \
#     --input_dir "/ssd0/shenzhen/Datasets/driving/boreas_1_26/test_A" \
#     --output_dir "output/cyclegan_turbo/boreas_1_26_resize_286_randomcrop_256x256_hflip/result_A"

# # # google street view -> boreas snowy
# python src/inference_unpaired_folder.py \
#     --prompt "driving in heavy snow" \
#     --direction "a2b" \
#     --model_path "output/cyclegan_turbo/boreas_1_26_v2_resize_286_randomcrop_256x256_hflip/checkpoints/model_25001.pkl" \
#     --image_prep "no_resize" \
#     --input_dir "/ssd0/shenzhen/Datasets/driving/boreas_1_26_v2/test_A" \
#     --output_dir "output/cyclegan_turbo/boreas_1_26_v2_resize_286_randomcrop_256x256_hflip/result_A"


# # # #  boreas snowy -> google street view
# CUDA_VISIBLE_DEVICES=4 python src/inference_unpaired_folder.py \
#     --prompt "driving in the day" \
#     --direction "b2a" \
#     --model_path "output/cyclegan_turbo/boreas_1_26_resize_286_randomcrop_256x256_hflip/checkpoints/model_25001.pkl" \
#     --image_prep "no_resize" \
#     --input_dir "/ssd0/shenzhen/Datasets/driving/boreas_1_26/test_B" \
#     --output_dir "output/cyclegan_turbo/boreas_1_26_resize_286_randomcrop_256x256_hflip/result_B"


# TODO: also do the opposite way later! 

# day2night (base)
# python src/inference_unpaired_folder.py \
#     --prompt "driving in the night" \
#     --direction "a2b" \
#     --model_path "output/cyclegan_turbo/bdd100k_7_19_night_resize_286_randomcrop_256x256_hflip/checkpoints/model_25001.pkl" \
#     --image_prep "no_resize" \
#     --input_dir "/home/shenzhen/Datasets/BDD100K/bdd100k_7_19_night/test_A" \
#     --output_dir "output/cyclegan_turbo/bdd100k_7_19_night_resize_286_randomcrop_256x256_hflip/result_A"

# day2night (warped 128)
# python src/inference_unpaired_folder.py \
#     --prompt "driving in the night" \
#     --direction "a2b" \
#     --model_path "output/cyclegan_turbo/bdd100k_7_19_night_warped_128_resize_286_randomcrop_256x256_hflip/checkpoints/model_25001.pkl" \
#     --image_prep "no_resize" \
#     --input_dir "/home/shenzhen/Datasets/BDD100K/bdd100k_7_19_night/test_A" \
#     --output_dir "output/cyclegan_turbo/bdd100k_7_19_night_warped_128_resize_286_randomcrop_256x256_hflip/result_A" \
#     --bw 128 \
#     --train-bbox-json /home/shenzhen/Datasets/BDD100K/100k/coco_labels/bdd100k_train_scale_0p5.json \
#     --val-bbox-json /home/shenzhen/Datasets/BDD100K/100k/coco_labels/bdd100k_val_scale_0p5.json

# day2night (warped 64)
# python src/inference_unpaired_folder.py \
#     --prompt "driving in the night" \
#     --direction "a2b" \
#     --model_path "output/cyclegan_turbo/bdd100k_7_19_night_warped_64_resize_286_randomcrop_256x256_hflip/checkpoints/model_25001.pkl" \
#     --image_prep "no_resize" \
#     --input_dir "/home/shenzhen/Datasets/BDD100K/bdd100k_7_19_night/test_A" \
#     --output_dir "output/cyclegan_turbo/bdd100k_7_19_night_warped_64_resize_286_randomcrop_256x256_hflip/result_A" \
#     --bw 64 \
#     --train-bbox-json /home/shenzhen/Datasets/BDD100K/100k/coco_labels/bdd100k_train_scale_0p5.json \
#     --val-bbox-json /home/shenzhen/Datasets/BDD100K/100k/coco_labels/bdd100k_val_scale_0p5.json

# # day2night (warped 64; warp instead of resize) # NOTE: use 24501 since 25001 get OOM
# python src/inference_unpaired_folder.py \
#     --prompt "driving in the night" \
#     --direction "a2b" \
#     --model_path "output/cyclegan_turbo/bdd100k_7_19_night_warped_64_286x286_randomcrop_256x256_hflip/checkpoints/model_24501.pkl" \
#     --image_prep "no_resize" \
#     --input_dir "/home/shenzhen/Datasets/BDD100K/bdd100k_7_19_night/test_A" \
#     --output_dir "output/cyclegan_turbo/bdd100k_7_19_night_warped_64_286x286_randomcrop_256x256_hflip/result_A" \
#     --bw 64 \
#     --train-bbox-json /home/shenzhen/Datasets/BDD100K/100k/coco_labels/bdd100k_train_scale_0p5.json \
#     --val-bbox-json /home/shenzhen/Datasets/BDD100K/100k/coco_labels/bdd100k_val_scale_0p5.json    

# night2day (base)
# CUDA_VISIBLE_DEVICES=6 python src/inference_unpaired_folder.py \
#     --prompt "driving in the day" \
#     --direction "b2a" \
#     --model_path "output/cyclegan_turbo/bdd100k_7_19_night_resize_286_randomcrop_256x256_hflip/checkpoints/model_25001.pkl" \
#     --image_prep "no_resize" \
#     --input_dir "/home/shenzhen/Datasets/BDD100K/bdd100k_7_19_night/test_B" \
#     --output_dir "output/cyclegan_turbo/bdd100k_7_19_night_resize_286_randomcrop_256x256_hflip/result_B"

# # night2day (warped 128)
# CUDA_VISIBLE_DEVICES=7 python src/inference_unpaired_folder.py \
#     --prompt "driving in the day" \
#     --direction "b2a" \
#     --model_path "output/cyclegan_turbo/bdd100k_7_19_night_warped_128_resize_286_randomcrop_256x256_hflip/checkpoints/model_25001.pkl" \
#     --image_prep "no_resize" \
#     --input_dir "/home/shenzhen/Datasets/BDD100K/bdd100k_7_19_night/test_B" \
#     --output_dir "output/cyclegan_turbo/bdd100k_7_19_night_warped_128_resize_286_randomcrop_256x256_hflip/result_B" \
#     --bw 128 \
#     --train-bbox-json /home/shenzhen/Datasets/BDD100K/100k/coco_labels/bdd100k_train_scale_0p5.json \
#     --val-bbox-json /home/shenzhen/Datasets/BDD100K/100k/coco_labels/bdd100k_val_scale_0p5.json    

# night2day (warped 64)
# CUDA_VISIBLE_DEVICES=7 python src/inference_unpaired_folder.py \
#     --prompt "driving in the day" \
#     --direction "b2a" \
#     --model_path "output/cyclegan_turbo/bdd100k_7_19_night_warped_64_resize_286_randomcrop_256x256_hflip/checkpoints/model_25001.pkl" \
#     --image_prep "no_resize" \
#     --input_dir "/home/shenzhen/Datasets/BDD100K/bdd100k_7_19_night/test_B" \
#     --output_dir "output/cyclegan_turbo/bdd100k_7_19_night_warped_64_resize_286_randomcrop_256x256_hflip/result_B" \
#     --bw 64 \
#     --train-bbox-json /home/shenzhen/Datasets/BDD100K/100k/coco_labels/bdd100k_train_scale_0p5.json \
#     --val-bbox-json /home/shenzhen/Datasets/BDD100K/100k/coco_labels/bdd100k_val_scale_0p5.json

# # night2day (warped 64; warp instead of resize) # NOTE: use 24501 since 25001 get OOM
# CUDA_VISIBLE_DEVICES=7 python src/inference_unpaired_folder.py \
#     --prompt "driving in the day" \
#     --direction "b2a" \
#     --model_path "output/cyclegan_turbo/bdd100k_7_19_night_warped_64_286x286_randomcrop_256x256_hflip/checkpoints/model_24501.pkl" \
#     --image_prep "no_resize" \
#     --input_dir "/home/shenzhen/Datasets/BDD100K/bdd100k_7_19_night/test_B" \
#     --output_dir "output/cyclegan_turbo/bdd100k_7_19_night_warped_64_286x286_randomcrop_256x256_hflip/result_B" \
#     --bw 64 \
#     --train-bbox-json /home/shenzhen/Datasets/BDD100K/100k/coco_labels/bdd100k_train_scale_0p5.json \
#     --val-bbox-json /home/shenzhen/Datasets/BDD100K/100k/coco_labels/bdd100k_val_scale_0p5.json

# # clear2rainy (base)
# CUDA_VISIBLE_DEVICES=1 python src/inference_unpaired_folder.py \
#     --prompt "driving in heavy rain" \
#     --direction "a2b" \
#     --model_path "output/cyclegan_turbo/bdd100k_1_20_resize_286_randomcrop_256x256_hflip/checkpoints/model_25001.pkl" \
#     --image_prep "no_resize" \
#     --input_dir "/home/shenzhen/Datasets/BDD100K/bdd100k_1_20/test_A" \
#     --output_dir "output/cyclegan_turbo/bdd100k_1_20_resize_286_randomcrop_256x256_hflip/result_A"

# # clear2rainy (warped 128)
# CUDA_VISIBLE_DEVICES=2 python src/inference_unpaired_folder.py \
#     --prompt "driving in heavy rain" \
#     --direction "a2b" \
#     --model_path "output/cyclegan_turbo/bdd100k_1_20_warped_128_resize_286_randomcrop_256x256_hflip/checkpoints/model_25001.pkl" \
#     --image_prep "no_resize" \
#     --input_dir "/home/shenzhen/Datasets/BDD100K/bdd100k_1_20/test_A" \
#     --output_dir "output/cyclegan_turbo/bdd100k_1_20_warped_128_resize_286_randomcrop_256x256_hflip/result_A" \
#     --bw 128 \
#     --train-bbox-json /home/shenzhen/Datasets/BDD100K/100k/coco_labels/bdd100k_train_scale_0p5.json \
#     --val-bbox-json /home/shenzhen/Datasets/BDD100K/100k/coco_labels/bdd100k_val_scale_0p5.json

# # clear2rainy (warped 64)
# CUDA_VISIBLE_DEVICES=2 python src/inference_unpaired_folder.py \
#     --prompt "driving in heavy rain" \
#     --direction "a2b" \
#     --model_path "output/cyclegan_turbo/bdd100k_1_20_warped_64_resize_286_randomcrop_256x256_hflip/checkpoints/model_25001.pkl" \
#     --image_prep "no_resize" \
#     --input_dir "/home/shenzhen/Datasets/BDD100K/bdd100k_1_20/test_A" \
#     --output_dir "output/cyclegan_turbo/bdd100k_1_20_warped_64_resize_286_randomcrop_256x256_hflip/result_A" \
#     --bw 64 \
#     --train-bbox-json /home/shenzhen/Datasets/BDD100K/100k/coco_labels/bdd100k_train_scale_0p5.json \
#     --val-bbox-json /home/shenzhen/Datasets/BDD100K/100k/coco_labels/bdd100k_val_scale_0p5.json

# clear2rainy (warped 64; warp instead of resize) # NOTE: use 24501 since 25001 get OOM
# CUDA_VISIBLE_DEVICES=2 python src/inference_unpaired_folder.py \
#     --prompt "driving in heavy rain" \
#     --direction "a2b" \
#     --model_path "output/cyclegan_turbo/bdd100k_1_20_warped_64_286x286_randomcrop_256x256_hflip/checkpoints/model_25001.pkl" \
#     --image_prep "no_resize" \
#     --input_dir "/home/shenzhen/Datasets/BDD100K/bdd100k_1_20/test_A" \
#     --output_dir "output/cyclegan_turbo/bdd100k_1_20_warped_64_286x286_randomcrop_256x256_hflip/result_A" \
#     --bw 64 \
#     --train-bbox-json /home/shenzhen/Datasets/BDD100K/100k/coco_labels/bdd100k_train_scale_0p5.json \
#     --val-bbox-json /home/shenzhen/Datasets/BDD100K/100k/coco_labels/bdd100k_val_scale_0p5.json


# # rainy2clear (base)
# CUDA_VISIBLE_DEVICES=5 python src/inference_unpaired_folder.py \
#     --prompt "driving in the day" \
#     --direction "b2a" \
#     --model_path "output/cyclegan_turbo/bdd100k_1_20_resize_286_randomcrop_256x256_hflip/checkpoints/model_25001.pkl" \
#     --image_prep "no_resize" \
#     --input_dir "/home/shenzhen/Datasets/BDD100K/bdd100k_1_20/test_B" \
#     --output_dir "output/cyclegan_turbo/bdd100k_1_20_resize_286_randomcrop_256x256_hflip/result_B"

# # rainy2clear (warped 128)
# CUDA_VISIBLE_DEVICES=6 python src/inference_unpaired_folder.py \
#     --prompt "driving in the day" \
#     --direction "b2a" \
#     --model_path "output/cyclegan_turbo/bdd100k_1_20_warped_128_resize_286_randomcrop_256x256_hflip/checkpoints/model_25001.pkl" \
#     --image_prep "no_resize" \
#     --input_dir "/home/shenzhen/Datasets/BDD100K/bdd100k_1_20/test_B" \
#     --output_dir "output/cyclegan_turbo/bdd100k_1_20_warped_128_resize_286_randomcrop_256x256_hflip/result_B" \
#     --bw 128 \
#     --train-bbox-json /home/shenzhen/Datasets/BDD100K/100k/coco_labels/bdd100k_train_scale_0p5.json \
#     --val-bbox-json /home/shenzhen/Datasets/BDD100K/100k/coco_labels/bdd100k_val_scale_0p5.json

# rainy2clear (warped 64)
# CUDA_VISIBLE_DEVICES=6 python src/inference_unpaired_folder.py \
#     --prompt "driving in the day" \
#     --direction "b2a" \
#     --model_path "output/cyclegan_turbo/bdd100k_1_20_warped_64_resize_286_randomcrop_256x256_hflip/checkpoints/model_25001.pkl" \
#     --image_prep "no_resize" \
#     --input_dir "/home/shenzhen/Datasets/BDD100K/bdd100k_1_20/test_B" \
#     --output_dir "output/cyclegan_turbo/bdd100k_1_20_warped_64_resize_286_randomcrop_256x256_hflip/result_B" \
#     --bw 64 \
#     --train-bbox-json /home/shenzhen/Datasets/BDD100K/100k/coco_labels/bdd100k_train_scale_0p5.json \
#     --val-bbox-json /home/shenzhen/Datasets/BDD100K/100k/coco_labels/bdd100k_val_scale_0p5.json

# # rainy2clear (warped 64; warp instead of resize) # NOTE: use 24501 since 25001 get OOM
# CUDA_VISIBLE_DEVICES=6 python src/inference_unpaired_folder.py \
#     --prompt "driving in the day" \
#     --direction "b2a" \
#     --model_path "output/cyclegan_turbo/bdd100k_1_20_warped_64_286x286_randomcrop_256x256_hflip/checkpoints/model_25001.pkl" \
#     --image_prep "no_resize" \
#     --input_dir "/home/shenzhen/Datasets/BDD100K/bdd100k_1_20/test_B" \
#     --output_dir "output/cyclegan_turbo/bdd100k_1_20_warped_64_286x286_randomcrop_256x256_hflip/result_B" \
#     --bw 64 \
#     --train-bbox-json /home/shenzhen/Datasets/BDD100K/100k/coco_labels/bdd100k_train_scale_0p5.json \
#     --val-bbox-json /home/shenzhen/Datasets/BDD100K/100k/coco_labels/bdd100k_val_scale_0p5.json

# bdd -> boreas snow (base)
# CUDA_VISIBLE_DEVICES=1 python src/inference_unpaired_folder.py \
#     --prompt "driving in heavy snow" \
#     --direction "a2b" \
#     --model_path "output/cyclegan_turbo/bdd100k_boreas_resize_286_randomcrop_256x256_hflip/checkpoints/model_25001.pkl" \
#     --image_prep "no_resize" \
#     --input_dir "/home/shenzhen/Datasets/BDD100K/bdd100k_boreas/test_A" \
#     --output_dir "output/cyclegan_turbo/bdd100k_boreas_resize_286_randomcrop_256x256_hflip/result_A"

# # boreas snow -> bdd (base)
# CUDA_VISIBLE_DEVICES=4 python src/inference_unpaired_folder.py \
#     --prompt "driving in the day" \
#     --direction "b2a" \
#     --model_path "output/cyclegan_turbo/bdd100k_boreas_resize_286_randomcrop_256x256_hflip/checkpoints/model_25001.pkl" \
#     --image_prep "no_resize" \
#     --input_dir "/home/shenzhen/Datasets/BDD100K/bdd100k_boreas/test_B" \
#     --output_dir "output/cyclegan_turbo/bdd100k_boreas_resize_286_randomcrop_256x256_hflip/result_B"

# # bdd -> boreas snow (warped 128)
# CUDA_VISIBLE_DEVICES=3 python src/inference_unpaired_folder.py \
#     --prompt "driving in heavy snow" \
#     --direction "a2b" \
#     --model_path "output/cyclegan_turbo/bdd100k_boreas_warped_128_resize_286_randomcrop_256x256_hflip/checkpoints/model_25001.pkl" \
#     --image_prep "no_resize" \
#     --input_dir "/home/shenzhen/Datasets/BDD100K/bdd100k_boreas/test_A" \
#     --output_dir "output/cyclegan_turbo/bdd100k_boreas_warped_128_resize_286_randomcrop_256x256_hflip/result_A" \
#     --bw 128 \
#     --train-bbox-json /home/shenzhen/Datasets/BDD100K/100k/coco_labels/bdd100k_train_scale_0p5.json \
#     --val-bbox-json /home/shenzhen/Datasets/BDD100K/100k/coco_labels/bdd100k_val_scale_0p5.json

# # boreas snow -> bdd (warped 128)
# CUDA_VISIBLE_DEVICES=5 python src/inference_unpaired_folder.py \
#     --prompt "driving in the day" \
#     --direction "b2a" \
#     --model_path "output/cyclegan_turbo/bdd100k_boreas_warped_128_resize_286_randomcrop_256x256_hflip/checkpoints/model_25001.pkl" \
#     --image_prep "no_resize" \
#     --input_dir "/home/shenzhen/Datasets/BDD100K/bdd100k_boreas/test_B" \
#     --output_dir "output/cyclegan_turbo/bdd100k_boreas_warped_128_resize_286_randomcrop_256x256_hflip/result_B" \
#     --bw 128 \
#     --train-bbox-json /home/shenzhen/Datasets/BDD100K/100k/coco_labels/bdd100k_train_scale_0p5.json \
#     --val-bbox-json /home/shenzhen/Datasets/BDD100K/100k/coco_labels/bdd100k_val_scale_0p5.json