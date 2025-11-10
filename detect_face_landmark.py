#!/usr/bin/env python3
# -*- coding: utf-8 -*-

'''
Example command lines:
python detect_face_landmark.py --input /home/shenzhen/Datasets/dataset_with_garment_bigface_100/BADRHINO_MEN_T-SHIRTS_011/bdy_1.jpg --out face_landmark
python detect_face_landmark.py --input /home/shenzhen/Datasets/dataset_with_garment_bigface_100/Andres-otalora_Women_Tops_028/bdy_1.jpg --out face_landmark
python detect_face_landmark.py --input /home/shenzhen/Datasets/dataset_with_garment_bigface_100/BananaRepublic_R2_Men_Sweatshirts_Hoodies_Joggers_027/bdy_2.webp --out face_landmark
python detect_face_landmark.py --input /home/shenzhen/Datasets/dataset_with_garment_bigface_100/bananarepublic_R2_woman_t-shirts_018/bdy_1.webp --out face_landmark
'''

import argparse, os, sys, glob
import numpy as np
import cv2
from PIL import Image
import pathlib

try:
    from insightface.app import FaceAnalysis
except Exception as e:
    print("Failed to import insightface. Install with:\n"
          "  pip install insightface onnxruntime onnxruntime-gpu opencv-python pillow numpy")
    raise

def parse_args():
    ap = argparse.ArgumentParser()
    ap.add_argument("--input", required=True, help="Path to an image file or a directory of images")
    ap.add_argument("--out", required=True, help="Output folder for visualized results")
    ap.add_argument("--cpu", action="store_true", help="Force CPU (no GPU provider)")
    ap.add_argument("--det-size", type=int, nargs=2, default=[640, 640], help="Detector input size WxH")
    return ap.parse_args()

def load_images(p):
    p = pathlib.Path(p)
    if p.is_file():
        return [str(p)]
    exts = ["*.jpg","*.jpeg","*.png","*.webp","*.bmp"]
    files = []
    for ext in exts:
        files += glob.glob(str(p / ext))
    files.sort()
    return files

def init_face_app(use_cpu=False, det_size=(640,640)):
    providers = ["CPUExecutionProvider"] if use_cpu else ["CUDAExecutionProvider","CPUExecutionProvider"]
    app = FaceAnalysis(name="buffalo_l", providers=providers)
    app.prepare(ctx_id=0 if not use_cpu else -1, det_size=tuple(det_size))
    return app

def draw_and_save(img_bgr, faces, out_path):
    vis = img_bgr.copy()
    for i, face in enumerate(faces):
        # draw face bbox
        x1, y1, x2, y2 = face.bbox.astype(int)
        cv2.rectangle(vis, (x1, y1), (x2, y2), (0, 255, 0), 2)

        # 5 keypoints (left_eye, right_eye, nose, left_mouth, right_mouth)
        if hasattr(face, "kps") and face.kps is not None:
            kps = face.kps.astype(int)
            for j, (x, y) in enumerate(kps):
                cv2.circle(vis, (x, y), 3, (0, 0, 255), -1)
                cv2.putText(vis, f"k{j}", (x + 3, y - 3),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.4, (0, 0, 255), 1, cv2.LINE_AA)

            # 🔵 draw simple blue boxes around eyes
            left_eye, right_eye = kps[0], kps[1]
            eye_w = int((x2 - x1) * 0.25)
            eye_h = int((y2 - y1) * 0.15) # NOTE: use 0.15 instead of 0.20 to avoid too large boxes
            for ex, ey in [left_eye, right_eye]:
                ex1, ey1 = int(ex - eye_w / 2), int(ey - eye_h / 2)
                ex2, ey2 = int(ex + eye_w / 2), int(ey + eye_h / 2)
                cv2.rectangle(vis, (ex1, ey1), (ex2, ey2), (255, 0, 0), 2)

        # (optional) 106 landmarks if available
        if hasattr(face, "landmark_2d_106") and face.landmark_2d_106 is not None:
            lm = face.landmark_2d_106.astype(int)
            for (x, y) in lm:
                cv2.circle(vis, (x, y), 1, (255, 0, 0), -1)

    os.makedirs(os.path.dirname(out_path), exist_ok=True)
    cv2.imwrite(out_path, vis)

def main():
    args = parse_args()
    imgs = load_images(args.input)
    if not imgs:
        print(f"No images found at {args.input}")
        sys.exit(1)

    os.makedirs(args.out, exist_ok=True)
    app = init_face_app(use_cpu=args.cpu, det_size=args.det_size)

    for path in imgs:
        img_bgr = cv2.imread(path)
        if img_bgr is None:
            print(f"[SKIP] Cannot read image: {path}")
            continue

        faces = app.get(img_bgr)

        print(f"\n=== {path} ===")
        if not faces:
            print("No face detected.")
        else:
            # sort by area (largest first)
            faces.sort(key=lambda f: (f.bbox[2]-f.bbox[0])*(f.bbox[3]-f.bbox[1]), reverse=True)
            for idx, face in enumerate(faces):
                bbox = face.bbox.tolist()
                print(f"[Face {idx}] bbox: [x1={bbox[0]:.1f}, y1={bbox[1]:.1f}, x2={bbox[2]:.1f}, y2={bbox[3]:.1f}]")

                # 5-keypoint eyes available here:
                if hasattr(face, "kps") and face.kps is not None:
                    kps = face.kps  # shape (5,2): [left_eye, right_eye, nose, left_mouth, right_mouth]
                    print(f"         kps (5):\n"
                          f"           left_eye : ({kps[0][0]:.1f}, {kps[0][1]:.1f})\n"
                          f"           right_eye: ({kps[1][0]:.1f}, {kps[1][1]:.1f})\n"
                          f"           nose     : ({kps[2][0]:.1f}, {kps[2][1]:.1f})\n"
                          f"           l_mouth  : ({kps[3][0]:.1f}, {kps[3][1]:.1f})\n"
                          f"           r_mouth  : ({kps[4][0]:.1f}, {kps[4][1]:.1f})")
                else:
                    print("         kps not available on this face object.")

                # 106 landmarks (optional; only if the model/app provides them)
                if hasattr(face, "landmark_2d_106") and face.landmark_2d_106 is not None:
                    print(f"         landmark_2d_106: {face.landmark_2d_106.shape}")
                else:
                    print("         landmark_2d_106 not available.")

        # visualize and save
        out_name = os.path.join(args.out, os.path.basename(path))
        draw_and_save(img_bgr, faces, out_name)
        print(f"Saved visualization → {out_name}")

if __name__ == "__main__":
    main()