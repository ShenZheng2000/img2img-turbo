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
        # TODO: think if we need to clamp here
        x1 = max(0, min(x1, w - 1))
        y1 = max(0, min(y1, h - 1))
        x2 = max(0, min(x2, w - 1))
        y2 = max(0, min(y2, h - 1))
    else:
        x1, y1, x2, y2 = 0, 0, w, h  # fallback: full image

    return torch.tensor([[x1, y1, x2, y2]], dtype=torch.float32)


# ===============================================================
# ✅ Forward warp (returns warped tensor + warp grid)
# ===============================================================
def apply_forward_warp(image_tensor, bbox_tensor, bw):
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
        separable=True,
        bandwidth_scale=bw,
        amplitude_scale=1.0,
    ).to(device)
    warp_grid = grid_net(image_tensor, gt_bboxes=bbox_tensor.unsqueeze(0))
    warped = warp(warp_grid, image_tensor)
    return warped, warp_grid

# ===============================================================
# ✅ Unwarp (returns restored tensor)
# ===============================================================
def apply_unwarp(warp_grid, warped_output):
    """
    Applies inverse KDE grid to unwarp the warped output tensor.
    """
    device = warped_output.device  # make sure we stay on the same GPU
    _, _, H, W = warped_output.shape

    inv_grid = invert_grid(warp_grid.to(device), (1, 3, H, W), separable=True)
    restored = F.grid_sample(warped_output, inv_grid.to(device), align_corners=True)
    return restored


# NOTE: use for training only
# NOTE: these are not working, skip for now. 
# # ===============================================================
# def warp_batch(x_src, face_app, bw):
#     """Apply forward warp to a batch of source images."""
#     if bw <= 0:
#         return x_src, [None] * x_src.size(0)

#     warped_list, warp_grid_list = [], []
#     B = x_src.size(0)
#     for b in range(B):
#         img_pil = transforms.ToPILImage()(x_src[b].cpu() * 0.5 + 0.5)
#         bbox = detect_face_bbox(img_pil, face_app)
#         warped, warp_grid = apply_forward_warp(x_src[b:b+1], bbox.to(x_src.device), bw)
#         warped_list.append(warped)
#         warp_grid_list.append(warp_grid)
#     return torch.cat(warped_list, dim=0), warp_grid_list

# # ===============================================================
# def unwarp_batch(x_pred, warp_grid_list):
#     """Unwarp model outputs using saved warp grids."""
#     if not any(warp_grid_list):
#         return x_pred
#     unwarped_list = []
#     B = x_pred.size(0)
#     for b in range(B):
#         if warp_grid_list[b] is not None:
#             x_unw = apply_unwarp(warp_grid_list[b], x_pred[b:b+1])
#         else:
#             x_unw = x_pred[b:b+1]
#         unwarped_list.append(x_unw)
#     return torch.cat(unwarped_list, dim=0)