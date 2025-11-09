"""
warp_pipeline.py
Reusable warp–unwarp utilities for Pix2Pix-Turbo (used in both train & inference).
"""

import numpy as np
import torch
from .warping_layers import PlainKDEGrid, warp, invert_grid

# optional import (only used for face detection)
from insightface.app import FaceAnalysis
from torchvision import transforms
import torch.nn.functional as F

# ===============================================================
# ✅ Initialize face detector once
# ===============================================================
def get_face_app(local_rank=0):
    """Initialize and return a FaceAnalysis detector (explicit init, no lazy caching)."""
    _face_app = FaceAnalysis(name='buffalo_l', providers=["CUDAExecutionProvider", "CPUExecutionProvider"])
    _face_app.prepare(ctx_id=local_rank)
    return _face_app


# ===============================================================
# ✅ Face detection (returns bbox tensor [1,4])
# ===============================================================
def detect_face_bbox(image_pil, face_app=None):
    """
    Detects largest face in the image (PIL.Image) and returns bbox tensor [[x1,y1,x2,y2]].
    Falls back to full image if no face found.
    """
    img_cv2 = np.array(image_pil)[:, :, ::-1]  # PIL → BGR
    faces = face_app.get(img_cv2)
    h, w = image_pil.height, image_pil.width

    if len(faces) > 0:
        faces.sort(key=lambda f: (f.bbox[2]-f.bbox[0])*(f.bbox[3]-f.bbox[1]), reverse=True)
        x1, y1, x2, y2 = map(int, faces[0].bbox)
    else:
        x1, y1, x2, y2 = 0, 0, w, h  # fallback: full image

    return torch.tensor([[x1, y1, x2, y2]], dtype=torch.float32)


# ===============================================================
# ✅ Forward warp (returns warped tensor + warp grid)
# ===============================================================
def apply_forward_warp(image_tensor, bbox_tensor, bw, separable=True):
    """
    Applies KDE-based forward warp around bbox region.
    Returns (warped_tensor, warp_grid).
    """
    device = image_tensor.device
    bbox_tensor = bbox_tensor.to(device)
    _, _, H, W = image_tensor.shape
    grid_net = PlainKDEGrid(
        input_shape=(H, W),
        output_shape=(H, W),
        separable=separable,
        bandwidth_scale=bw,
        amplitude_scale=1.0,
    ).to(device)
    warp_grid = grid_net(image_tensor, gt_bboxes=bbox_tensor.unsqueeze(0))
    warped = warp(warp_grid, image_tensor)
    return warped, warp_grid

# ===============================================================
# ✅ Unwarp (returns restored tensor)
# ===============================================================
def apply_unwarp(warp_grid, warped_output, separable=True):
    """
    Applies inverse KDE grid to unwarp the warped output tensor.
    """
    device = warped_output.device  # make sure we stay on the same GPU
    _, _, H, W = warped_output.shape

    inv_grid = invert_grid(warp_grid.to(device), (1, 3, H, W), separable=separable)
    restored = F.grid_sample(warped_output, inv_grid.to(device), align_corners=True)
    return restored