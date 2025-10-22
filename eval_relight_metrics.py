import os
import torch
import lpips
import numpy as np
from PIL import Image
from torchvision import transforms
from tqdm import tqdm
from torchmetrics.image import StructuralSimilarityIndexMeasure, PeakSignalNoiseRatio

'''
Example usage:
python eval_relighting_metrics.py \
  --gt_dir "data/candlelight_1_may4/test_B" \
  --pred_dir "/scratch/shenzhen/img2img-turbo/output/pix2pix_turbo/exp_10_2/candlelight_1_half_on_full/result_A"
'''

device = "cuda" if torch.cuda.is_available() else "cpu"
loss_fn = lpips.LPIPS(net="vgg").to(device)
ssim = StructuralSimilarityIndexMeasure(data_range=1.0).to(device)
psnr = PeakSignalNoiseRatio(data_range=1.0).to(device)

def load_image(path):
    img = Image.open(path).convert("RGB")
    return transforms.ToTensor()(img).unsqueeze(0).to(device)

def compute_metrics(gt_dir, pred_dir):
    files = sorted(os.listdir(gt_dir))
    lpips_scores, ssim_scores, psnr_scores = [], [], []

    for f in tqdm(files, desc="Evaluating"):
        gt_path = os.path.join(gt_dir, f)
        pred_path = os.path.join(pred_dir, f)
        if not (os.path.exists(gt_path) and os.path.exists(pred_path)):
            continue

        img_gt = load_image(gt_path)
        img_pred = load_image(pred_path)

        if img_gt.shape != img_pred.shape:
            _, _, h, w = img_gt.shape
            img_pred = torch.nn.functional.interpolate(img_pred, size=(h, w), mode='bilinear', align_corners=False)

        lp = loss_fn(img_gt, img_pred).item()
        ss = ssim(img_gt, img_pred).item()
        ps = psnr(img_gt, img_pred).item()

        lpips_scores.append(lp)
        ssim_scores.append(ss)
        psnr_scores.append(ps)

    print(f"\n✅ Results on {len(lpips_scores)} images:")
    print(f"  LPIPS (↓): {np.mean(lpips_scores):.4f}")
    print(f"  SSIM  (↑): {np.mean(ssim_scores):.4f}")
    print(f"  PSNR  (↑): {np.mean(psnr_scores):.2f} dB")

if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--gt_dir", required=True, help="Ground truth candlelight images (test_B)")
    parser.add_argument("--pred_dir", required=True, help="Model outputs (result_A)")
    args = parser.parse_args()

    compute_metrics(args.gt_dir, args.pred_dir)