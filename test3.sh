# TODO: write a shell function to reduce code duplication. 

# python src/inference_unpaired_folder.py \
#     --prompt "driving in the night" \
#     --direction "a2b" \
#     --model_path "output/cyclegan_turbo/bdd100k_7_19_night_resize_286_randomcrop_256x256_hflip/checkpoints/model_25001.pkl" \
#     --image_prep "no_resize" \
#     --input_dir "/home/shenzhen/Datasets/BDD100K/bdd100k_7_19_night/test_A" \
#     --output_dir "output/cyclegan_turbo/bdd100k_7_19_night/result_A"

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

# python src/inference_unpaired_folder.py \
#     --prompt "driving in the night" \
#     --direction "b2a" \
#     --model_path "output/cyclegan_turbo/bdd100k_7_19_night/checkpoints/model_25001.pkl" \
#     --image_prep "no_resize" \
#     --input_dir "/home/shenzhen/Datasets/BDD100K/bdd100k_7_19_night/test_B" \
#     --output_dir "output/cyclegan_turbo/bdd100k_7_19_night/result_B"

# python src/inference_unpaired_folder.py \
#     --prompt "driving in the night" \
#     --direction "b2a" \
#     --model_path "output/cyclegan_turbo/bdd100k_7_19_night_warped_128_resize_286_randomcrop_256x256_hflip/checkpoints/model_25001.pkl" \
#     --image_prep "no_resize" \
#     --input_dir "/home/shenzhen/Datasets/BDD100K/bdd100k_7_19_night/test_B" \
#     --output_dir "output/cyclegan_turbo/bdd100k_7_19_night_warped_128_resize_286_randomcrop_256x256_hflip/result_B" \
#     --bw 128 \
#     --train-bbox-json /home/shenzhen/Datasets/BDD100K/100k/coco_labels/bdd100k_train_scale_0p5.json \
#     --val-bbox-json /home/shenzhen/Datasets/BDD100K/100k/coco_labels/bdd100k_val_scale_0p5.json    