# Train Pix2Pix-Turbo with Synthesized Images

## 1. Setup Repo and Install Env
```
git clone https://github.com/ShenZheng2000/img2img-turbo
cd img2img-turbo
conda env create -f environment.yaml
conda activate img2img-turbo

pip install huggingface_hub==0.25.0
pip install peft==0.10.0
pip install wandb
pip install vision_aided_loss

pip install insightface==0.7.3
pip install opencv-python pillow
pip install numpy==1.26.4
pip install onnxruntime-gpu==1.17.1
```


## 2. Image Warping on Detected Face Regions

Run the following commands inside the `img2img-turbo` repository.

NOTE: A bandwidth (`--bw`) of 128 is recommended, and using `--include-eyes` improves face and eye detail quality.

Run in terminal:
```
python warp_dataset.py \
    --input_root /home/shenzhen/Datasets/relighting \
    --target_prefix exp_10_16 \
    --relight_type golden_sunlight_1 \
    --bw 128 \
    --include-eyes
```

### Example Dataset Structure (Warped Images)
```
/home/shenzhen/Datasets/relighting/exp_10_16_warped_128/golden_sunlight_1
├── train_A
│ ├── 0.png
│ ├── 0.inv.pth
│ ├── 1.png
│ ├── 1.inv.pth
│ └── ...
├── train_B
│ ├── 0.png
│ ├── 0.inv.pth
│ ├── 1.png
│ ├── 1.inv.pth
│ └── ...
├── test_A
│ ├── 0.png
│ ├── 0.inv.pth
│ ├── 1.png
│ ├── 1.inv.pth
│ └── ...
├── test_B
│ ├── 0.png
│ ├── 0.inv.pth
│ ├── 1.png
│ ├── 1.inv.pth
│ └── ...
├── train_prompts.json
└── test_prompts.json
```


## 3. Model Training 

For training details, see the [official guide](https://github.com/GaParmar/img2img-turbo/blob/main/docs/training_pix2pix_turbo.md)

Example training with 4 GPUs, each at least having 48GB of memory

Run in terminal
```
bash run.sh
```

## 4. Model Testing

Example testing with 1 GPU
```
bash test2.sh
```