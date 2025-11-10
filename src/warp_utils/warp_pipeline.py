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

def detect_face_bbox(image_pil, face_app=None, include_eyes=False):
    """
    Detects the largest face in the image (PIL.Image).
    Returns:
        - torch.Tensor of shape [1, 4] if include_eyes=False → face only
        - torch.Tensor of shape [3, 4] if include_eyes=True  → [face, left_eye, right_eye]
    """
    img_cv2 = np.array(image_pil)[:, :, ::-1]  # PIL → BGR
    faces = face_app.get(img_cv2)
    h, w = image_pil.height, image_pil.width

    # fallback: full image
    x1, y1, x2, y2 = 0, 0, w, h

    if faces:
        # use largest detected face
        faces.sort(key=lambda f: (f.bbox[2]-f.bbox[0])*(f.bbox[3]-f.bbox[1]), reverse=True)
        face = faces[0]
        x1, y1, x2, y2 = map(int, face.bbox)

        if include_eyes and hasattr(face, "kps") and face.kps is not None:
            left_eye_box, right_eye_box = _compute_eye_boxes(face, x1, y1, x2, y2)
            return torch.tensor([[x1, y1, x2, y2], left_eye_box, right_eye_box], dtype=torch.float32)

    # if no eyes or not requested → just return face
    return torch.tensor([[x1, y1, x2, y2]], dtype=torch.float32)


def _compute_eye_boxes(face, x1, y1, x2, y2):
    """Compute left/right eye boxes from 5-point keypoints."""
    kps = face.kps.astype(int)
    left_eye, right_eye = kps[0], kps[1]
    eye_w = int((x2 - x1) * 0.25)
    eye_h = int((y2 - y1) * 0.15) # NOTE: use 0.15 instead of 0.20 to avoid too large boxes

    lx1, ly1 = int(left_eye[0] - eye_w / 2), int(left_eye[1] - eye_h / 2)
    lx2, ly2 = int(left_eye[0] + eye_w / 2), int(left_eye[1] + eye_h / 2)
    rx1, ry1 = int(right_eye[0] - eye_w / 2), int(right_eye[1] - eye_h / 2)
    rx2, ry2 = int(right_eye[0] + eye_w / 2), int(right_eye[1] + eye_h / 2)

    return [lx1, ly1, lx2, ly2], [rx1, ry1, rx2, ry2]


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