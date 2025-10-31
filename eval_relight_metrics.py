import os
os.environ["INSIGHTFACE_FORCE_TORCH"] = "1"
import cv2
import torch
import lpips
import numpy as np
from PIL import Image
from torchvision import transforms
from tqdm import tqdm
from torchmetrics.image import StructuralSimilarityIndexMeasure, PeakSignalNoiseRatio
from insightface.app import FaceAnalysis

device = "cuda" if torch.cuda.is_available() else "cpu"
loss_fn = lpips.LPIPS(net="vgg").to(device)
ssim = StructuralSimilarityIndexMeasure(data_range=1.0).to(device)
psnr = PeakSignalNoiseRatio(data_range=1.0).to(device)

# --------------------------------------
# Initialize InsightFace once
# --------------------------------------
face_app = FaceAnalysis(name='buffalo_l')
face_app.prepare(ctx_id=0)   # ctx_id=0 → GPU:0

def load_image(path):
    img = Image.open(path).convert("RGB")
    return transforms.ToTensor()(img).unsqueeze(0).to(device)

def compute_metrics(gt_dir, pred_dir):
    files = sorted(os.listdir(gt_dir))
    lpips_global, ssim_global, psnr_global = [], [], []
    lpips_face, ssim_face, psnr_face = [], [], []

    for f in tqdm(files, desc="Evaluating"):
        gt_path = os.path.join(gt_dir, f)
        pred_path = os.path.join(pred_dir, f)
        if not (os.path.exists(gt_path) and os.path.exists(pred_path)):
            continue

        # Load images
        img_gt = load_image(gt_path)        # torch [1,3,H,W]
        img_pred = load_image(pred_path)

        # Resize pred if mismatch
        if img_gt.shape != img_pred.shape:
            _, _, h, w = img_gt.shape
            img_pred = torch.nn.functional.interpolate(
                img_pred, size=(h, w), mode='bilinear', align_corners=False
            )

        # ------------------------
        # GLOBAL metrics
        # ------------------------
        lp = loss_fn(img_gt, img_pred).item()
        ss = ssim(img_gt, img_pred).item()
        ps = psnr(img_gt, img_pred).item()

        lpips_global.append(lp)
        ssim_global.append(ss)
        psnr_global.append(ps)

        # ------------------------
        # FACE REGION metrics
        # ------------------------
        # Detect face on GT image (cv2 BGR)
        img_gt_cv2 = cv2.imread(gt_path)
        faces = face_app.get(img_gt_cv2)

        if len(faces) > 0:
            # pick the largest face
            faces.sort(key=lambda f: (f.bbox[2]-f.bbox[0])*(f.bbox[3]-f.bbox[1]), reverse=True)
            x1, y1, x2, y2 = map(int, faces[0].bbox)

            # crop GT and PRED using IDENTICAL box
            crop_gt   = img_gt[:, :, y1:y2, x1:x2]
            crop_pred = img_pred[:, :, y1:y2, x1:x2]

            # safety check for empty / invalid cropping
            if crop_gt.numel() > 0 and crop_pred.numel() > 0:
                
                _, _, ch, cw = crop_gt.shape
                # skip tiny faces (< 32px) to avoid LPIPS pooling collapse
                if ch < 32 or cw < 32:
                    continue

                lp_f = loss_fn(crop_gt, crop_pred).item()
                ss_f = ssim(crop_gt, crop_pred).item()
                ps_f = psnr(crop_gt, crop_pred).item()

                lpips_face.append(lp_f)
                ssim_face.append(ss_f)
                psnr_face.append(ps_f)

    # ------------------------
    # PRINT RESULTS
    # ------------------------
    print(f"\n✅ Global metrics on {len(lpips_global)} images:")
    print(f"  LPIPS (↓): {np.mean(lpips_global):.4f}")
    print(f"  SSIM  (↑): {np.mean(ssim_global):.4f}")
    print(f"  PSNR  (↑): {np.mean(psnr_global):.2f} dB")

    if len(lpips_face) > 0:
        print(f"\n✅ Face-region metrics on {len(lpips_face)} images:")
        print(f"  LPIPS_face (↓): {np.mean(lpips_face):.4f}")
        print(f"  SSIM_face  (↑): {np.mean(ssim_face):.4f}")
        print(f"  PSNR_face  (↑): {np.mean(psnr_face):.2f} dB")
    else:
        print("\n⚠️ No face detected in any image -> No face metrics computed.")


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--gt_dir", required=True, help="Ground truth candlelight images (test_B)")
    parser.add_argument("--pred_dir", required=True, help="Model outputs (result_A)")
    args = parser.parse_args()

    compute_metrics(args.gt_dir, args.pred_dir)