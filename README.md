## 🚀 Overview

We extend **pix2pix-Turbo** and **CycleGAN-Turbo** with a plug-and-play **warp–unwarp strategy**.

Specifically, this framework supports three tasks:
* Human relighting
* Driving scene relighting
* Driving scene translation (weather & time-of-day)
<br>

## 📋 Table of Contents
- [Installation](#installation)
- [Dataset Preparation](#dataset-preparation)
- [Training and Testing](#training-and-testing)
- [Warp–Unwarp Integration](#warpunwarp-integration)
- [TODO Lists](#todo-lists)
- [Acknowledgement](#acknowledgement)


## ⚙️ Installation

Setup and environment installation is in [install.md](docs/install.md).
<br>


## 📥 Dataset Preparation

Dataset download, preparation and warping instructions are in [dataset.md](docs/dataset.md).
<br>


## ⚡ Training and Testing

Model training and testing instructions are in [train_test.md](docs/train_test.md).
<br>


## 🔌 Warp–Unwarp Integration

Key warp–unwarp code insertions are marked with **✅**.

The main insertion points are:
- Data loading → `src/my_utils/training_utils.py` (`PairedDataset`, `UnpairedDataset`)
- Pix2Pix-Turbo → `src/train_pix2pix_turbo.py`
- CycleGAN-Turbo → `src/train_cyclegan_turbo.py`
- Warp utilities → `src/warp_utils/`

<br>

## 📌 TODO Lists

- [ ] Add Gradio demo
- [ ] Add arXiv link

<br>

## 🙏 Acknowledgement

This project builds upon several excellent open-source works, including: 
- [img2img-turbo](https://github.com/GaParmar/img2img-turbo)
- [Instance-Warp](https://github.com/ShenZheng2000/Instance-Warp)
- [Two-Plane Prior](https://github.com/geometriczoom/two-plane-prior)

We appreciate the authors for making their code publicly available.