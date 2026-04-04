
# Training and Testing

## 1. Model Training

NOTE: Requires ~40GB GPU memory.

Configure GPUs via `accelerate config`:

- pix2pix-Turbo (relighting): 4 GPUs → see `train.sh`
- CycleGAN-Turbo (I2I): 8 GPUs → see `train3.sh`

<br>

## 2. Model Testing

## 2.1 Pretrained Models (img2img-Turbo)
- [BDD100K day2night](https://www.cs.cmu.edu/~img2img-turbo/models/day2night.pkl)
- [BDD100K night2day](https://www.cs.cmu.edu/~img2img-turbo/models/night2day.pkl) *(optional, can reuse day2night in reverse)*
- [BDD100K clear2rainy](https://www.cs.cmu.edu/~img2img-turbo/models/clear2rainy.pkl)
- [BDD100K rainy2clear](https://www.cs.cmu.edu/~img2img-turbo/models/rainy2clear.pkl) *(optional, can reuse clear2rainy in reverse)*


## 2.2 Our Model (Warp–Unwarp)
Our final model (with warping) is available [here](https://drive.google.com/drive/folders/136eVrXWOI6cSOnFRVYiGWqlERP_jLJ76?dmr=1&ec=wgc-drive-%5Bmodule%5D-goto).

## 2.3 Inference
See `inf_paired.sh` for human relighting and driving scene relighting with pix2pix-Turbo.

See `inf_unpaired.sh` for driving scene weather and time-of-day translation with CycleGAN-Turbo. 
