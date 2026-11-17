"""Camera (head / bike) motion estimated from the video itself.

For each pair of consecutive sampled frames we track corner points with
Lucas-Kanade optical flow and fit a similarity transform (translation,
rotation, scale) with RANSAC. This gives:

* ``affine[k]``  2x3 matrix mapping points of frame k-1 into frame k. Used to
  remove head motion from object velocities (features/graph_builder.py).
* ``feats[k]``   6 numbers per frame, the model's ego-motion modality:
      [tx, ty, rotation, log-scale, flow magnitude, inlier ratio]
  tx/ty: image shift (head turn / pitch), rotation: roll (lean),
  log-scale: image expansion (forward-speed proxy), flow magnitude: overall
  motion (speed proxy), inlier ratio: how reliable the estimate is.

No speedometer, GPS or manual entry is needed.
"""
from __future__ import annotations

from pathlib import Path
from typing import List, Sequence, Tuple

import cv2
import numpy as np

EGO_DIM = 6
FLIP_NEGATE = (0, 2)  # tx and rotation change sign when the image is mirrored
_IDENTITY = np.array([[1.0, 0.0, 0.0], [0.0, 1.0, 0.0]], dtype=np.float32)


def _background_mask(h: int, w: int, ego_box: Sequence[float]) -> np.ndarray:
    """Use only pixels outside the rider's own bike region."""
    mask = np.full((h, w), 255, dtype=np.uint8)
    cx, cy, bw, bh = ego_box
    x0, x1 = int((cx - bw / 2) * w), int((cx + bw / 2) * w)
    y0, y1 = int((cy - bh / 2) * h), int((cy + bh / 2) * h)
    mask[max(y0, 0):min(y1, h), max(x0, 0):min(x1, w)] = 0
    return mask


def estimate_pair(prev_gray: np.ndarray, gray: np.ndarray, mask: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
    h, w = gray.shape
    feats = np.zeros(EGO_DIM, dtype=np.float32)
    pts = cv2.goodFeaturesToTrack(prev_gray, maxCorners=400, qualityLevel=0.01, minDistance=8, mask=mask)
    if pts is None or len(pts) < 12:
        return _IDENTITY.copy(), feats
    nxt, status, _ = cv2.calcOpticalFlowPyrLK(prev_gray, gray, pts, None, winSize=(31, 31), maxLevel=4)
    good = status.ravel() == 1
    if good.sum() < 12:
        return _IDENTITY.copy(), feats
    p0, p1 = pts[good].reshape(-1, 2), nxt[good].reshape(-1, 2)
    A, inliers = cv2.estimateAffinePartial2D(p0, p1, method=cv2.RANSAC, ransacReprojThreshold=3.0)
    if A is None:
        return _IDENTITY.copy(), feats

    centre = np.array([w / 2.0, h / 2.0, 1.0])
    shift = A @ centre - centre[:2]
    scale = float(np.hypot(A[0, 0], A[1, 0]))
    rot = float(np.arctan2(A[1, 0], A[0, 0]))
    flow = float(np.median(np.linalg.norm(p1 - p0, axis=1)))
    inl = float(inliers.sum()) / max(len(p0), 1) if inliers is not None else 0.0
    feats[:] = [
        10.0 * shift[0] / w, 10.0 * shift[1] / h, 10.0 * rot,
        10.0 * np.log(max(scale, 1e-3)), 10.0 * flow / w, inl,
    ]
    return A.astype(np.float32), np.clip(feats, -5.0, 5.0)


def estimate_video(frame_paths: List[Path], ego_box: Sequence[float]) -> Tuple[np.ndarray, np.ndarray]:
    """Returns (feats [N, 6], affine [N, 2, 3]). Row 0 is zeros / identity."""
    n = len(frame_paths)
    feats = np.zeros((n, EGO_DIM), dtype=np.float32)
    affine = np.tile(_IDENTITY, (n, 1, 1))
    prev, mask = None, None
    for k, path in enumerate(frame_paths):
        gray = cv2.imdecode(np.frombuffer(path.read_bytes(), np.uint8), cv2.IMREAD_GRAYSCALE)
        if gray is None:
            prev = None
            continue
        if mask is None:
            mask = _background_mask(gray.shape[0], gray.shape[1], ego_box)
        if prev is not None:
            affine[k], feats[k] = estimate_pair(prev, gray, mask)
        prev = gray
    return feats, affine
