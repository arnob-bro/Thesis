"""Per-track object features.

Node feature contract (MD implementation_contracts_specification.md):

    [x_position, y_position, width, height,
     velocity_x, velocity_y, acceleration, object_type]
"""
from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Dict, List, Sequence

import numpy as np


@dataclass
class TrackPoint:
    frame_id: int
    track_id: int
    class_id: int
    bbox: Sequence[float]  # [x, y, w, h] absolute pixel coordinates


@dataclass
class ObjectFeature:
    track_id: int
    class_id: int
    x: float
    y: float
    w: float
    h: float
    vx: float
    vy: float
    acceleration: float

    def as_vector(self) -> List[float]:
        return [
            self.x, self.y, self.w, self.h,
            self.vx, self.vy,
            self.acceleration, float(self.class_id),
        ]


def _centre(box: Sequence[float]) -> np.ndarray:
    x, y, w, h = box
    return np.asarray([x + w / 2.0, y + h / 2.0], dtype=np.float64)


def compute_object_features(
    tracks_by_frame: Dict[int, List[TrackPoint]],
    fps: float,
    clamp_velocity: float = 50.0,
    clamp_acceleration: float = 20.0,
) -> Dict[int, List[ObjectFeature]]:
    """Compute per-frame per-track object features.

    Velocity and acceleration are estimated via finite differences across the
    *previous two frames* using the centre of the bounding box. Both are
    clamped (MD constraint: numerical stability).
    """
    dt = 1.0 / max(fps, 1e-6)
    frame_ids = sorted(tracks_by_frame.keys())
    out: Dict[int, List[ObjectFeature]] = {fid: [] for fid in frame_ids}

    # Cache previous centres per track for velocity, plus prev velocity for accel.
    prev_centre: Dict[int, np.ndarray] = {}
    prev_velocity: Dict[int, float] = {}

    for fid in frame_ids:
        # Sort tracks by ID to ensure deterministic node ordering
        sorted_tracks = sorted(tracks_by_frame[fid], key=lambda t: t.track_id)
        for trk in sorted_tracks:
            centre = _centre(trk.bbox)
            if trk.track_id in prev_centre:
                disp = centre - prev_centre[trk.track_id]
                vel_vec = disp / dt
                velocity = float(np.linalg.norm(vel_vec))
                vx = float(np.clip(vel_vec[0], -clamp_velocity, clamp_velocity))
                vy = float(np.clip(vel_vec[1], -clamp_velocity, clamp_velocity))
            else:
                velocity = 0.0
                vx = 0.0
                vy = 0.0

            if trk.track_id in prev_velocity:
                accel = (velocity - prev_velocity[trk.track_id]) / dt
            else:
                accel = 0.0

            accel = float(np.clip(accel, -clamp_acceleration, clamp_acceleration))

            out[fid].append(
                ObjectFeature(
                    track_id=trk.track_id,
                    class_id=trk.class_id,
                    x=float(centre[0]), y=float(centre[1]),
                    w=float(trk.bbox[2]), h=float(trk.bbox[3]),
                    vx=vx,
                    vy=vy,
                    acceleration=accel,
                )
            )

            prev_centre[trk.track_id] = centre
            prev_velocity[trk.track_id] = velocity

    return out


def normalize_node_features(
    features: List[ObjectFeature],
    image_size: Sequence[int],
    num_classes: int,
) -> np.ndarray:
    """Normalise to roughly [0,1] / standard ranges before feeding the GNN."""
    if not features:
        return np.zeros((0, 8), dtype=np.float32)

    W, H = float(image_size[0]), float(image_size[1])
    rows = []
    for f in features:
        rows.append([
            f.x / W,
            f.y / H,
            f.w / W,
            f.h / H,
            f.vx / 50.0,
            f.vy / 50.0,
            f.acceleration / 20.0,
            f.class_id / max(num_classes - 1, 1),
        ])
    return np.asarray(rows, dtype=np.float32)
