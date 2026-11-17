"""Rider-centric interaction graphs, built once per video.

Node features (``node_dim = 13 + number of classes``)::

    0 cx  1 cy  2 w  3 h            box, normalised by image size
    4 vx  5 vy                      velocity, head-motion compensated, log-scaled
    6 acc                           change of speed, log-scaled
    7 dist_to_rider                 distance to the rider's position / image diagonal
    8 bearing_sin  9 bearing_cos    direction from the rider (0 = straight ahead)
    10 looming                      box-height growth rate = 1 / tau  (approach cue)
    11 tau                          looming time-to-contact, clamped and normalised
    12 is_ego                       1 for the rider node
    13.. one-hot object class

Edge features: ``[proximity, rel_velocity_long, rel_velocity_lat, ttc]``.

The rider is node 0 of every frame (if ``graph.include_ego_node``) and is
connected to every object in both directions; objects are also linked to
their k nearest neighbours.
"""
from __future__ import annotations

import math
from pathlib import Path
from typing import Dict, List, Optional, Sequence

import numpy as np
import torch
from torch_geometric.data import Data

from features.interaction_features import (
    bbox_proximity_score, build_edges_knn, geometry_ttc, relative_velocity_components,
)
from features.object_features import ObjectFeature
from utils.config import Config

NODE_BASE_DIM = 13
EDGE_DIM = 4
MAX_TRACK_GAP = 3   # frames; longer gaps restart velocity estimation


def node_dim(cfg: Config) -> int:
    return NODE_BASE_DIM + len(cfg.yolo.classes)


def symlog(v, scale: float):
    """Signed log scaling: keeps small motions linear, never saturates."""
    return np.sign(v) * np.log1p(np.abs(v) / scale) / 4.0


# ---------------------------------------------------------------------------
# Ego-vehicle filter
# ---------------------------------------------------------------------------

def is_ego_detection(det: Dict, cfg: Config, size: Sequence[int]) -> bool:
    f = cfg.ego_filter
    if not f.enabled or det["class_id"] not in f.classes:
        return False
    W, H = size
    x, y, w, h = det["bbox"]
    return (y + h) >= f.min_bottom * H and w >= f.min_width * W


def filter_ego_detections(dets: List[Dict], cfg: Config, size: Sequence[int]) -> List[Dict]:
    return [d for d in dets if not is_ego_detection(d, cfg, size)]


# ---------------------------------------------------------------------------
# Graph construction
# ---------------------------------------------------------------------------

def _pair_attr(a: ObjectFeature, b: ObjectFeature, diag: float, ttc_clamp: float) -> List[float]:
    rv_long, rv_lat = relative_velocity_components(a, b)
    ttc = geometry_ttc(a, b, b.vx - a.vx, b.vy - a.vy, clamp=ttc_clamp)
    return [bbox_proximity_score(a, b, diag), rv_long, rv_lat, ttc]


def build_video_graphs(
    tracks_by_frame: Dict[int, List[Dict]],
    n_frames: int,
    affine: Optional[np.ndarray],
    cfg: Config,
    size: Sequence[int],
) -> Dict[str, np.ndarray]:
    """Build every frame's graph and pack them into flat arrays."""
    W, H = float(size[0]), float(size[1])
    diag = math.hypot(W, H)
    dt = 1.0 / float(cfg.frame_extraction.frame_rate)
    fc = cfg.features
    n_cls = len(cfg.yolo.classes)
    D = node_dim(cfg)
    use_ego = bool(cfg.graph.include_ego_node)
    compensate = bool(fc.compensate_head_motion) and affine is not None

    ecx, ecy, ew, eh = cfg.ego_box
    ego = ObjectFeature(-1, 0, ecx * W, ecy * H, ew * W, eh * H, 0.0, 0.0, 0.0)

    state: Dict[int, Dict] = {}
    xs, eis, eas, tids = [], [], [], []
    node_ptr, edge_ptr = [0], [0]

    for k in range(n_frames):
        objs: List[ObjectFeature] = []
        rows: List[List[float]] = []
        for det in sorted(tracks_by_frame.get(k, []), key=lambda d: d["track_id"]):
            x, y, w, h = det["bbox"]
            c = np.array([x + w / 2.0, y + h / 2.0])
            tid = int(det["track_id"])
            vel, acc, loom = np.zeros(2), 0.0, 0.0
            st = state.get(tid)
            if st is not None and 0 < k - st["k"] <= MAX_TRACK_GAP:
                gap = (k - st["k"]) * dt
                p = st["c"]
                if compensate:   # where the old position would be now if only the camera moved
                    for j in range(st["k"] + 1, k + 1):
                        p = affine[j] @ np.array([p[0], p[1], 1.0])
                vel = (c - p) / gap
                if st["speed"] is not None:
                    acc = (float(np.linalg.norm(vel)) - st["speed"]) / gap
                raw = (math.log(max(h, 1.0)) - math.log(max(st["h"], 1.0))) / gap
                loom = raw if st["loom"] is None else 0.5 * raw + 0.5 * st["loom"]
                state[tid] = {"k": k, "c": c, "h": h, "speed": float(np.linalg.norm(vel)), "loom": loom}
            else:
                state[tid] = {"k": k, "c": c, "h": h, "speed": None, "loom": None}

            loom = float(np.clip(loom, -fc.loom_clamp, fc.loom_clamp))
            tau = min(1.0 / loom, fc.tau_clamp) if loom > 1e-3 else float(fc.tau_clamp)
            d = c - np.array([ego.x, ego.y])
            dist = float(np.linalg.norm(d)) + 1e-6
            cls = int(det["class_id"])
            onehot = [1.0 if i == cls else 0.0 for i in range(n_cls)]
            rows.append([
                c[0] / W, c[1] / H, w / W, h / H,
                float(symlog(vel[0], fc.velocity_scale)), float(symlog(vel[1], fc.velocity_scale)),
                float(symlog(acc, fc.accel_scale)),
                dist / diag, d[0] / dist, -d[1] / dist,
                loom / fc.loom_clamp, tau / fc.tau_clamp, 0.0, *onehot,
            ])
            objs.append(ObjectFeature(tid, cls, float(c[0]), float(c[1]), float(w), float(h),
                                      float(vel[0]), float(vel[1]), acc))

        offset = 1 if use_ego else 0
        if use_ego:
            ego_row = [ecx, ecy, ew, eh] + [0.0] * (D - 4)
            ego_row[11], ego_row[12] = 1.0, 1.0
            rows.insert(0, ego_row)
        elif not rows:
            rows.append([0.0] * D)   # placeholder node so the graph is never empty

        edges, attrs = [], []
        knn_e, knn_a = build_edges_knn(objs, int(cfg.graph.k_neighbors), diag)
        for (s, t), a in zip(knn_e, knn_a):
            edges.append((s + offset, t + offset))
            attrs.append([a[0], a[1], a[2], min(a[3], fc.ttc_clamp)])
        if use_ego:
            for i, o in enumerate(objs):
                edges.append((0, i + 1)); attrs.append(_pair_attr(ego, o, diag, fc.ttc_clamp))
                edges.append((i + 1, 0)); attrs.append(_pair_attr(o, ego, diag, fc.ttc_clamp))

        ea = np.asarray(attrs, dtype=np.float32).reshape(-1, EDGE_DIM)
        if len(ea):
            ea[:, 1] = symlog(ea[:, 1], fc.velocity_scale)
            ea[:, 2] = symlog(ea[:, 2], fc.velocity_scale)
            ea[:, 3] = ea[:, 3] / fc.ttc_clamp

        xs.append(np.asarray(rows, dtype=np.float32))
        eis.append(np.asarray(edges, dtype=np.int64).reshape(-1, 2).T)
        eas.append(ea)
        tids.append(np.asarray(([-1] if use_ego or not objs else []) + [o.track_id for o in objs], dtype=np.int64))
        node_ptr.append(node_ptr[-1] + len(rows))
        edge_ptr.append(edge_ptr[-1] + len(edges))

    return {
        "x": np.concatenate(xs), "edge_index": np.concatenate(eis, axis=1),
        "edge_attr": np.concatenate(eas), "track_id": np.concatenate(tids),
        "node_ptr": np.asarray(node_ptr, dtype=np.int64),
        "edge_ptr": np.asarray(edge_ptr, dtype=np.int64),
    }


# ---------------------------------------------------------------------------
# Loading
# ---------------------------------------------------------------------------

class VideoGraphs:
    """All frame graphs of one video; ``get(k)`` returns frame k as PyG Data."""

    def __init__(self, path: str | Path) -> None:
        z = np.load(path)
        self.x = torch.from_numpy(z["x"])
        self.edge_index = torch.from_numpy(z["edge_index"])
        self.edge_attr = torch.from_numpy(z["edge_attr"])
        self.node_ptr = z["node_ptr"]
        self.edge_ptr = z["edge_ptr"]

    def __len__(self) -> int:
        return len(self.node_ptr) - 1

    def get(self, k: int, flip: bool = False) -> Data:
        n0, n1 = self.node_ptr[k], self.node_ptr[k + 1]
        e0, e1 = self.edge_ptr[k], self.edge_ptr[k + 1]
        x = self.x[n0:n1].clone()
        ea = self.edge_attr[e0:e1].clone()
        if flip:   # mirror left-right
            x[:, 0] = 1.0 - x[:, 0]
            x[:, 4] = -x[:, 4]
            x[:, 8] = -x[:, 8]
            ea[:, 2] = -ea[:, 2]
        return Data(x=x, edge_index=self.edge_index[:, e0:e1], edge_attr=ea)


STAT_NAMES = [
    "n_objects_mean", "n_objects_max", "loom_max", "loom_mean", "tau_min",
    "dist_min", "speed_mean", "speed_max", "proximity_max", "ttc_min",
    "box_h_max", "n_objects_last", "loom_last_max", "dist_last_min",
]


def window_stats(graphs: List[Data]) -> np.ndarray:
    """Hand-crafted summary of a window (for the gradient-boosting baseline)."""
    n, loom, tau, dist, speed, prox, ttc, hmax = [], [], [], [], [], [], [], []
    per_frame = []
    for g in graphs:
        obj = g.x[g.x[:, 12] < 0.5]
        obj = obj[obj[:, 2:4].abs().sum(dim=1) > 0]      # drop placeholder nodes
        n.append(len(obj))
        per_frame.append(obj)
        if len(obj):
            loom += obj[:, 10].tolist(); tau += obj[:, 11].tolist(); dist += obj[:, 7].tolist()
            speed += obj[:, 4:6].norm(dim=1).tolist(); hmax += obj[:, 3].tolist()
        if g.edge_attr.numel():
            prox += g.edge_attr[:, 0].tolist(); ttc += g.edge_attr[:, 3].tolist()
    last = per_frame[-1]
    f = lambda v, fn, default: float(fn(v)) if len(v) else default
    return np.array([
        np.mean(n), np.max(n), f(loom, max, 0.0), f(loom, np.mean, 0.0), f(tau, min, 1.0),
        f(dist, min, 1.0), f(speed, np.mean, 0.0), f(speed, max, 0.0), f(prox, max, 0.0),
        f(ttc, min, 1.0), f(hmax, max, 0.0), float(len(last)),
        float(last[:, 10].max()) if len(last) else 0.0,
        float(last[:, 7].min()) if len(last) else 1.0,
    ], dtype=np.float32)
