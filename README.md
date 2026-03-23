🚀 Train img2img-Turbo with Synthesized Images

# ⚙️ 1. Setup Repo and Install Env
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

# 📥 2. Download Datasets

For **BDD100K**, download `100K Images` and `Labels` from [here](http://bdd-data.berkeley.edu/download.html) and `coco_labels` from [here](https://drive.google.com/drive/folders/1Hqf1S_I2Q_PG77wD8GGgRN0Z7h8ocqCc?usp=drive_link)

For **Cityscapes**, download `leftImg8bit_trainvaltest.zip` and `gtFine_trainvaltest.zip` from [here](https://www.cityscapes-dataset.com/downloads/), and `gt_detection` from [here](https://drive.google.com/drive/folders/1yYBRz96Xf_Hld9DWu-4I-DvQuHtMVvUd?usp=drive_link).

For **Dark Zurich**, download [`Dark_Zurich_train_anon.zip`](https://data.vision.ee.ethz.ch/csakarid/shared/GCMA_UIoU/Dark_Zurich_train_anon.zip), and [`Dark_Zurich_val_anno.zip`](https://drive.google.com/file/d/1cM6X0WpqUdOceGRILvlPUdASVdA-zyQi/view?usp=drive_link). 

For **ACDC**, download `rgb_anon_trainvaltest.zip` and `gt_trainval.zip` from [here](https://acdc.vision.ee.ethz.ch/download), and `gt_detection` from [here](https://drive.google.com/drive/folders/1LwJwM3heHy-U9u9bfpNEl8h0f9_5yJw4?usp=drive_link).

For **VITON-HD**, download from [here](https://drive.google.com/file/d/1tLx8LRp-sxDp0EcYmYoV_vXdSc-jJ79w/view).

For **ROADWork** (boston), download from [here](https://drive.google.com/file/d/11weqyiw3ODjwyG1aWklqYhVNk-hsOPwY/view?usp=drive_link)


<details><summary><strong>📂 Dataset Structure</strong></summary>

```
Datasets/
├── relighting/
│   ├── VITON/
│   │   └── test/
│   │       └── image/
│   └── workzone_segm/
│       └── boston/
│           └── image/
└── driving/
    ├── BDD100K/
    │   └── 100k/
    │       ├── coco_labels/
    │       ├── images/
    │       └── labels/
    ├── cityscapes/
    │   ├── gt_detection/
    │   └── leftImg8bit/
    ├── dark_zurich/
    │   └── rgb_anon/
    └── acdc/
        ├── gt_detection/
        └── rgb_anon/
```

</details>
<br>


# 🔄 3. Split and Convert datasets

For splitting BDD100K (day/night) and (clear/rainy), and converting to img2img-turbo format, see `shen_scripts/prepare_bdd100k_dataset.py`.

For converting other datasets, see `shen_scripts/prepare_driving_dataset.py`.
<details><summary><strong>📂 Dataset Structure</strong></summary>
```
/home/shenzhen/Datasets/driving/BDD100K_clear2rainy
├── train_A
│ ├── 0.png
│ └── ...
├── train_B
│ ├── 0.png
│ └── ...
├── test_A
│ ├── 0.png
│ └── ...
├── test_B
│ ├── 0.png
│ └── ...
├── train_prompts.json
└── test_prompts.json
```

</details>


# 🧩 4. Warp Images on Salient Regions

See `warp.sh` for example commands. 
<details><summary><strong>📂 Dataset Structure</strong></summary>

```
/home/shenzhen/Datasets/driving/BDD100K_clear2rainy
├── train_A
│ ├── 0.png
│ ├── 0.inv.pth  <-- NEW: Inverse grid files added here
│ └── ...
├── train_B
│ ├── 0.png
│ ├── 0.inv.pth  <-- NEW: Inverse grid files added here
│ └── ...
├── test_A
│ ├── 0.png
│ └── ...
├── test_B
│ ├── 0.png
│ └── ...
├── train_prompts.json
└── test_prompts.json
```

</details>


# 🧠 5. Model Training 

NOTE: Requires ~40GB GPU memory.

Configure GPUs via `accelerate config`:

- Pix2Pix-Turbo (relighting): 4 GPUs → see `train.sh`
- CycleGAN-Turbo (I2I): 8 GPUs → see `train3.sh`


# 🔍 6. Model Testing

See `inf.sh` for inference commands. 


# 🧩 7. Understand Warp-UnWarp

Most warp–unwarp related code insertions are marked with ✅. Searching for these markers is the quickest way to locate where the functionality is integrated.

The main insertion points are:

- **Data loading**: `PairedDataset` and `UnpairedDataset` in `src/my_utils/training_utils.py`

- **Paired training (Pix2Pix-Turbo)**: `src/train_pix2pix_turbo.py`

- **Unpaired training (CycleGAN-Turbo)**: `src/train_cyclegan_turbo.py`


In contrast, the underlying warp–unwarp implementations are defined in utility functions under `src/warp_utils/`