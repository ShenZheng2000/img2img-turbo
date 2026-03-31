# 🚀 Overview

We extend **pix2pix-Turbo** and **CycleGAN-Turbo** with a plug-and-play **warp–unwarp strategy**.

Specifically, this framework supports three tasks:
* Human relighting
* Driving scene relighting
* Driving scene translation (weather & time-of-day)


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


## 2.1 Unpaired Data
For **BDD100K**, download `100K Images` and `Labels` from [here](http://bdd-data.berkeley.edu/download.html) and `coco_labels` from [here](https://drive.google.com/drive/folders/1Hqf1S_I2Q_PG77wD8GGgRN0Z7h8ocqCc?usp=drive_link).

For **Cityscapes**, download `leftImg8bit_trainvaltest.zip` and `gtFine_trainvaltest.zip` from [here](https://www.cityscapes-dataset.com/downloads/), and `gt_detection` from [here](https://drive.google.com/drive/folders/1yYBRz96Xf_Hld9DWu-4I-DvQuHtMVvUd?usp=drive_link).

For **Dark Zurich**, download [`Dark_Zurich_train_anon.zip`](https://data.vision.ee.ethz.ch/csakarid/shared/GCMA_UIoU/Dark_Zurich_train_anon.zip), and [`Dark_Zurich_val_anno.zip`](https://drive.google.com/file/d/1cM6X0WpqUdOceGRILvlPUdASVdA-zyQi/view?usp=drive_link). 

For **ACDC**, download `rgb_anon_trainvaltest.zip` and `gt_trainval.zip` from [here](https://acdc.vision.ee.ethz.ch/download), and `gt_detection` from [here](https://drive.google.com/drive/folders/1LwJwM3heHy-U9u9bfpNEl8h0f9_5yJw4?usp=drive_link).


## 2.2 Paired Test Data
For **VITON-HD**, download test images from [here](https://drive.google.com/file/d/1tLx8LRp-sxDp0EcYmYoV_vXdSc-jJ79w/view).

For **ROADWork** (boston), download test images from [here](https://drive.google.com/file/d/11weqyiw3ODjwyG1aWklqYhVNk-hsOPwY/view?usp=drive_link).

## 2.3 Paired Train Data (Synthetic)
Paired training data for **pix2pix-Turbo** model is generated using our relighting pipeline: https://github.com/ShenZheng2000/relighting. 


<details><summary><strong>📂 Dataset Structure</strong></summary>

```
Datasets/
├── relighting/
│   ├── VITON/
│   │   ├── test/
│   │   │   └── image/
│   │   └── train_debug_100/
│   │       └── image/
│   └── workzone_segm/
│       ├── boston/
│       │   └── image/
│       └── pittsburgh/
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
<br>


# 🌀 4. Warp Images on Salient Regions

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
<br>


# 🧠 5. Model Training 

NOTE: Requires ~40GB GPU memory.

Configure GPUs via `accelerate config`:

- Pix2Pix-Turbo (relighting): 4 GPUs → see `train.sh`
- CycleGAN-Turbo (I2I): 8 GPUs → see `train3.sh`


# 🔍 6. Model Testing

## 6.1 Pretrained Models (img2img-Turbo)
- [BDD100K day2night](https://www.cs.cmu.edu/~img2img-turbo/models/day2night.pkl)
- [BDD100K night2day](https://www.cs.cmu.edu/~img2img-turbo/models/night2day.pkl) *(optional, can reuse day2night in reverse)*
- [BDD100K clear2rainy](https://www.cs.cmu.edu/~img2img-turbo/models/clear2rainy.pkl)
- [BDD100K rainy2clear](https://www.cs.cmu.edu/~img2img-turbo/models/rainy2clear.pkl) *(optional, can reuse clear2rainy in reverse)*


## 6.2 Our Model (Warp–Unwarp)
Our final model (with warping) is available [here](https://drive.google.com/drive/folders/136eVrXWOI6cSOnFRVYiGWqlERP_jLJ76?dmr=1&ec=wgc-drive-%5Bmodule%5D-goto).

## 6.3 Inference
See `inf.sh` for example inference commands.


# 📍 7. Warp–Unwarp Integration

Key warp–unwarp code insertions are marked with **✅**.

The main insertion points are:
- Data loading → `src/my_utils/training_utils.py` (`PairedDataset`, `UnpairedDataset`)
- Pix2Pix-Turbo → `src/train_pix2pix_turbo.py`
- CycleGAN-Turbo → `src/train_cyclegan_turbo.py`
- Warp utilities → `src/warp_utils/`


# 📌 8. TODO Lists

- [ ] Add Gradio demo
- [ ] Add arXiv link


# 🙏 9. Acknowledgement

This project builds upon several excellent open-source works, including: 
- [img2img-turbo](https://github.com/GaParmar/img2img-turbo)
- [Instance-Warp](https://github.com/ShenZheng2000/Instance-Warp)
- [Two-Plane Prior](https://github.com/geometriczoom/two-plane-prior)

We appreciate the authors for making their code publicly available.