"""Pairwise interaction (edge) features for the traffic interaction graph.

Edge feature contract (MD):

    [proximity, rel_velocity_long, rel_velocity_lat, ttc]

TTC philosophy (MD graph_and_ego_policies.md):
    Estimates are based on predicted rectangle overlap under constant
    velocity motion, using axis-aligned bounding box geometry.
"""
from __future__ import annotations

import math
from typing import List, Sequence, Tuple

import numpy as np

from features.object_features import ObjectFeature


def pairwise_distance(a: ObjectFeature, b: ObjectFeature) -> float:
    return float(math.hypot(a.x - b.x, a.y - b.y))


def bbox_edge_distance(a: ObjectFeature, b: ObjectFeature) -> float:
    dx = max(0.0, abs(a.x - b.x) - (a.w + b.w) / 2.0)
    dy = max(0.0, abs(a.y - b.y) - (a.h + b.h) / 2.0)
    return float(math.hypot(dx, dy))


def bbox_overlap_ratio(a: ObjectFeature, b: ObjectFeature) -> float:
    a_left = a.x - a.w / 2.0
    a_right = a.x + a.w / 2.0
    a_top = a.y - a.h / 2.0
    a_bottom = a.y + a.h / 2.0

    b_left = b.x - b.w / 2.0
    b_right = b.x + b.w / 2.0
    b_top = b.y - b.h / 2.0
    b_bottom = b.y + b.h / 2.0

    overlap_width = max(0.0, min(a_right, b_right) - max(a_left, b_left))
    overlap_height = max(0.0, min(a_bottom, b_bottom) - max(a_top, b_top))
    overlap_area = overlap_width * overlap_height
    union_area = a.w * a.h + b.w * b.h - overlap_area
    return float(overlap_area / max(union_area, 1.0))


def bbox_proximity_score(
    a: ObjectFeature, b: ObjectFeature, image_diag: float,
) -> float:
    edge_dist = bbox_edge_distance(a, b)
    base = 1.0 - min(edge_dist, image_diag) / max(image_diag, 1.0)
    overlap = bbox_overlap_ratio(a, b)
    return float(np.clip(base + overlap, 0.0, 1.0))


def bbox_separator_vector(a: ObjectFeature, b: ObjectFeature) -> np.ndarray:
    dx = b.x - a.x
    dy = b.y - a.y
    half_w = (a.w + b.w) / 2.0
    half_h = (a.h + b.h) / 2.0

    sep_x = 0.0 if abs(dx) <= half_w else dx - math.copysign(half_w, dx)
    sep_y = 0.0 if abs(dy) <= half_h else dy - math.copysign(half_h, dy)
    if sep_x == 0.0 and sep_y == 0.0:
        return np.array([dx, dy], dtype=np.float64)
    return np.array([sep_x, sep_y], dtype=np.float64)


def relative_velocity_components(
    a: ObjectFeature, b: ObjectFeature,
) -> Tuple[float, float]:
    """Relative velocity decomposed along and across the effective bbox separation."""
    direction = bbox_separator_vector(a, b)
    norm = np.linalg.norm(direction)
    if norm < 1e-6:
        rel = np.array([b.vx - a.vx, b.vy - a.vy], dtype=np.float64)
        return float(rel[0]), float(rel[1])

    direction = direction / norm
    perp = np.array([-direction[1], direction[0]], dtype=np.float64)
    rel = np.array([b.vx - a.vx, b.vy - a.vy], dtype=np.float64)
    rel_long = float(np.dot(rel, direction))
    rel_lat = float(np.dot(rel, perp))
    return rel_long, rel_lat


def _interval_collision_times(
    a_center: float,
    a_length: float,
    b_center: float,
    b_length: float,
    rel_vel: float,
    eps: float = 1e-6,
) -> Tuple[float, float]:
    """Return the entry and exit time for 1D interval overlap under constant motion."""
    a_half = a_length / 2.0
    b_half = b_length / 2.0
    a_min, a_max = -a_half, a_half
    b_min = (b_center - a_center) - b_half
    b_max = (b_center - a_center) + b_half

    if abs(rel_vel) < eps:
        if b_max < a_min or b_min > a_max:
            return float("inf"), float("-inf")
        return 0.0, float("inf")

    t_enter = (a_min - b_max) / rel_vel
    t_exit = (a_max - b_min) / rel_vel
    if t_enter > t_exit:
        t_enter, t_exit = t_exit, t_enter
    return t_enter, t_exit


def geometry_ttc(
    a: ObjectFeature,
    b: ObjectFeature,
    rel_vel_x: float,
    rel_vel_y: float,
    eps: float = 1e-6,
    clamp: float = 30.0,
) -> float:
    """Estimate TTC using constant velocity rectangle overlap."""
    # Use axis-aligned bounding boxes and image-space relative motion.
    entry_x, exit_x = _interval_collision_times(
        a_center=a.x,
        a_length=a.w,
        b_center=b.x,
        b_length=b.w,
        rel_vel=rel_vel_x,
        eps=eps,
    )
    entry_y, exit_y = _interval_collision_times(
        a_center=a.y,
        a_length=a.h,
        b_center=b.y,
        b_length=b.h,
        rel_vel=rel_vel_y,
        eps=eps,
    )

    if entry_x == float("inf") or entry_y == float("inf"):
        return clamp

    entry = max(entry_x, entry_y)
    exit = min(exit_x, exit_y)
    if exit < 0 or entry > exit:
        return clamp
    return float(np.clip(max(entry, 0.0), 0.0, clamp))


def build_edges_knn(
    features: List[ObjectFeature], k: int, image_diag: float,
) -> Tuple[List[Tuple[int, int]], List[List[float]]]:
    """Construct a sparse KNN edge set (k = 3..5 per MD).

    Returns (edge_index_list, edge_feature_list). Edges are directed; each
    object connects to its k nearest neighbours.
    """
    n = len(features)
    if n <= 1:
        return [], []

    coords = np.array([(f.x, f.y) for f in features], dtype=np.float64)
    diff = coords[:, None, :] - coords[None, :, :]
    dist_matrix = np.linalg.norm(diff, axis=-1)
    np.fill_diagonal(dist_matrix, np.inf)

    k_actual = min(k, n - 1)
    nn_idx = np.argpartition(dist_matrix, k_actual - 1, axis=1)[:, :k_actual]

    edge_index: List[Tuple[int, int]] = []
    edge_attr: List[List[float]] = []

    for src in range(n):
        for dst in nn_idx[src]:
            dst = int(dst)
            proximity = bbox_proximity_score(features[src], features[dst], image_diag)
            rv_long, rv_lat = relative_velocity_components(features[src], features[dst])
            rel_vel_x = float(features[dst].vx - features[src].vx)
            rel_vel_y = float(features[dst].vy - features[src].vy)
            ttc = geometry_ttc(
                a=features[src],
                b=features[dst],
                rel_vel_x=rel_vel_x,
                rel_vel_y=rel_vel_y,
                eps=1e-6,
                clamp=30.0,
            )
            edge_index.append((src, dst))
            edge_attr.append([proximity, rv_long, rv_lat, ttc])

    return edge_index, edge_attr


def approx_ttc(
    distance: float, rel_velocity: float, eps: float = 1e-6, clamp: float = 30.0,
) -> float:
    """Approximate Time-To-Collision for ego-relative features."""
    if rel_velocity <= 0:
        return clamp
    ttc = distance / (rel_velocity + eps)
    return float(np.clip(ttc, 0.0, clamp))


def normalize_edge_features(
    edge_attr: List[List[float]], image_diag: float,
) -> np.ndarray:
    if not edge_attr:
        return np.zeros((0, 4), dtype=np.float32)
    arr = np.asarray(edge_attr, dtype=np.float32)
    arr[:, 0] = np.clip(arr[:, 0], 0.0, 1.0)        # proximity score
    arr[:, 1] = arr[:, 1] / 50.0                   # longitudinal rel velocity
    arr[:, 2] = arr[:, 2] / 50.0                   # lateral rel velocity
    arr[:, 3] = arr[:, 3] / 30.0                   # TTC
    return arr


def compute_ego_object_interactions(
    features: List[ObjectFeature],
    ego_speed: float,
    image_size: Sequence[int],
) -> List[dict]:
    """Compute ego→object interaction features (MD graph_and_ego_policies.md).

    The ego vehicle is NOT a graph node, but we still record per-object
    ego-relative collision-risk signals for downstream fusion / debugging.
    """
    W, H = float(image_size[0]), float(image_size[1])
    ego_xy = np.array([W / 2.0, H], dtype=np.float64)   # bottom-centre = ego
    records = []
    for f in features:
        obj = np.array([f.x, f.y])
        dist = float(np.linalg.norm(obj - ego_xy))
        speed = float(np.linalg.norm([f.vx, f.vy]))
        closing_speed = max(ego_speed - speed, 0.0)
        ttc = approx_ttc(dist, closing_speed)
        records.append(
            {
                "track_id": f.track_id,
                "distance": dist,
                "closing_speed": closing_speed,
                "ttc": ttc,
            }
        )
    return records
