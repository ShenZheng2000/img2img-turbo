## 🚀 Overview

We extend **pix2pix-Turbo** and **CycleGAN-Turbo** with a plug-and-play **warp–unwarp strategy**.

Specifically, this framework supports three tasks:
* Human relighting
* Driving scene relighting
* Driving scene translation (weather & time-of-day)


## ⚙️ Installation

Setup and environment installation is in [install.md](docs/install.md).



## 📥 Dataset Preparation

Dataset download, preparation and warping instructions are in [dataset.md](docs/dataset.md).



## ⚡ Training and Testing

Model training and testing instructions are in [train_test.md](docs/train_test.md).



## 🔌 Warp–Unwarp Integration

Key warp–unwarp code insertions are marked with **✅**.

The main insertion points are:
- Data loading → `src/my_utils/training_utils.py` (`PairedDataset`, `UnpairedDataset`)
- Pix2Pix-Turbo → `src/train_pix2pix_turbo.py`
- CycleGAN-Turbo → `src/train_cyclegan_turbo.py`
- Warp utilities → `src/warp_utils/`



## 📌 TODO Lists

- [ ] Add Gradio demo
- [ ] Add arXiv link



## 🙏 Acknowledgement

This project builds upon the following excellent open-source works:
- [img2img-turbo](https://github.com/GaParmar/img2img-turbo)
- [Instance-Warp](https://github.com/ShenZheng2000/Instance-Warp)
- [Two-Plane Prior](https://github.com/geometriczoom/two-plane-prior)