"""Looming time-to-contact rule: a non-learned baseline.

For every frame, take the objects in front of the rider that are getting
bigger (positive looming) and keep the smallest time-to-contact
tau = 1 / (box-height growth rate). A window's tau is the largest of these
per-frame minima over its last ``persist`` frames, so a single noisy frame
cannot raise an alarm on its own. The risk class is then read off tau with
three thresholds, e.g. the time-to-event bins themselves:

    tau < 1.5 s -> high, < 3 s -> moderate, < 4.5 s -> slight, else safe

``tau_fixed`` uses the bins as thresholds (no training at all);
``tau_tuned`` picks the thresholds that maximise macro-F1 on the training
folds. Both only use the cached graphs (features/graph_builder.py).

Settings, all optional, in config.yaml:

    baselines:
      looming:
        front_cos: 0.5     # object within +-60 deg of straight ahead of the rider
        min_height: 0.04   # ignore boxes shorter than this fraction of image height
        persist: 2         # frames the low tau must last
        softness: 0.5      # s, only shapes the probabilities used for AUC / AP
"""
from __future__ import annotations

from itertools import combinations
from typing import Dict, Sequence, Tuple

import numpy as np

from features.graph_builder import VideoGraphs
from utils.config import Config

# node feature columns (see features/graph_builder.py)
_H, _BEARING_COS, _LOOM, _TAU, _IS_EGO = 3, 9, 10, 11, 12

DEFAULTS = {"front_cos": 0.5, "min_height": 0.04, "persist": 2, "softness": 0.5}


def rule_settings(cfg: Config) -> Dict[str, float]:
    custom = (cfg.get("baselines") or {}).get("looming") or {}
    return {k: type(v)(custom.get(k, v)) for k, v in DEFAULTS.items()}


def bin_thresholds(cfg: Config) -> Tuple[float, float, float]:
    """(slight, moderate, high) upper TTE bounds from event_labeling.bins."""
    b = cfg.event_labeling.bins
    return float(b.slight[1]), float(b.moderate[1]), float(b.high[1])


def frame_tau(graphs: VideoGraphs, cfg: Config) -> np.ndarray:
    """Smallest looming tau (seconds) of a frontal, approaching object, per frame."""
    s = rule_settings(cfg)
    clamp = float(cfg.features.tau_clamp)
    x = graphs.x.numpy()
    out = np.full(len(graphs), clamp, dtype=np.float32)
    for k in range(len(graphs)):
        f = x[graphs.node_ptr[k]:graphs.node_ptr[k + 1]]
        keep = ((f[:, _IS_EGO] < 0.5) & (f[:, _LOOM] > 0) & (f[:, _BEARING_COS] >= s["front_cos"])
                & (f[:, _H] >= s["min_height"]))
        if keep.any():
            out[k] = float(f[keep, _TAU].min()) * clamp
    return out


def window_tau(per_frame: np.ndarray, end: int, persist: int) -> float:
    """tau of the window ending at frame ``end`` (inclusive)."""
    return float(per_frame[max(end - persist + 1, 0):end + 1].max())


def classify(tau: np.ndarray, thresholds: Sequence[float]) -> np.ndarray:
    """Risk class = how many thresholds tau is below (thresholds descending)."""
    return (np.asarray(tau)[:, None] < np.asarray(thresholds)[None, :]).sum(axis=1)


def probabilities(tau: np.ndarray, thresholds: Sequence[float], softness: float) -> np.ndarray:
    """Soft class probabilities: P(risk > k) = sigmoid((t_k - tau) / softness).

    Monotone in tau, so AUC / AP measure tau's ranking. P(risk >= c) >= 0.5
    exactly when tau < t_c, so the alarm threshold 0.5 matches ``classify``.
    """
    tau = np.asarray(tau, dtype=np.float64)[:, None]
    p_gt = 1.0 / (1.0 + np.exp(-(np.asarray(thresholds)[None, :] - tau) / softness))
    p_gt = np.minimum.accumulate(p_gt, axis=1)
    ones = np.ones((len(tau), 1))
    return (np.hstack([ones, p_gt]) - np.hstack([p_gt, 0 * ones])).astype(np.float32)


def _macro_f1(y: np.ndarray, p: np.ndarray, n_cls: int) -> float:
    cm = np.bincount(y * n_cls + p, minlength=n_cls * n_cls).reshape(n_cls, n_cls)
    tp = np.diag(cm).astype(float)
    support, predicted = cm.sum(1), cm.sum(0)
    f1 = 2 * tp / np.maximum(support + predicted, 1)
    present = support > 0
    return float(f1[present].mean()) if present.any() else 0.0


def fit_thresholds(tau: np.ndarray, y: np.ndarray, n_cls: int = 4,
                   grid: Sequence[float] = tuple(np.arange(0.25, 6.0, 0.25))) -> Tuple[float, ...]:
    """Thresholds (descending, one per class boundary) with the best training macro-F1."""
    y = np.asarray(y, dtype=np.int64)
    below = np.asarray(tau)[:, None] < np.asarray(grid)[None, :]   # [N, G]
    best, best_t = -1.0, None
    for idx in combinations(range(len(grid)), n_cls - 1):          # ascending indices
        pred = below[:, list(idx)].sum(axis=1)
        score = _macro_f1(y, pred, n_cls)
        if score > best:
            best, best_t = score, tuple(float(grid[i]) for i in reversed(idx))
    return best_t
