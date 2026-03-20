# Train img2img-Turbo with Synthesized Images

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

pip install ultralytics==8.4.19
pip install omegaconf
```


## 2. Warp Images on Salient Regions

See `warp.sh` for example commands. 

### Example Dataset Structure (Warped Images)
```
/home/shenzhen/Datasets/relighting/exp_10_16_warped_128/golden_sunlight_1
├── train_A
│ ├── 0.png
│ ├── 0.inv.pth
│ └── ...
├── train_B
│ ├── 0.png
│ ├── 0.inv.pth
│ └── ...
├── test_A
│ ├── 0.png
│ ├── 0.inv.pth
│ └── ...
├── test_B
│ ├── 0.png
│ ├── 0.inv.pth
│ └── ...
├── train_prompts.json
└── test_prompts.json
```


## 3. Model Training 

NOTE: Requires ~40GB GPU memory.

Configure GPUs via `accelerate config`:

- Pix2Pix-Turbo (relighting): 4 GPUs → see `train.sh`
- CycleGAN-Turbo (I2I): 8 GPUs → see `train3.sh`


## 4. Model Testing

See `inf.sh` for inference commands. 