"""Sequence-level metrics: classification, ordinal error and TTC error."""
from __future__ import annotations

from typing import Dict, Optional, Sequence

import numpy as np
from sklearn.metrics import (
    accuracy_score, confusion_matrix, f1_score, precision_recall_fscore_support, roc_auc_score,
)

RISK_LABELS = ["safe", "slight", "moderate", "high"]


def _safe_auc(y_true: np.ndarray, probs: np.ndarray) -> float:
    """Macro one-vs-rest AUC over the classes that are present."""
    try:
        present = np.unique(y_true)
        if len(present) < 2:
            return float("nan")
        if len(present) == probs.shape[1]:
            return float(roc_auc_score(y_true, probs, multi_class="ovr"))
        aucs = [roc_auc_score((y_true == c).astype(int), probs[:, c]) for c in present]
        return float(np.mean(aucs))
    except ValueError:
        return float("nan")


def compute_metrics(
    y_true: Sequence[int],
    y_pred: Sequence[int],
    probs: Optional[np.ndarray] = None,
    pred_ttc: Optional[Sequence[float]] = None,
    true_ttc: Optional[Sequence[float]] = None,   # NaN where there is no TTC target
    num_classes: int = 4,
) -> Dict[str, object]:
    yt, yp = np.asarray(y_true), np.asarray(y_pred)
    labels = list(range(num_classes))
    prec, rec, f1, support = precision_recall_fscore_support(yt, yp, labels=labels, zero_division=0)
    present = support > 0
    out: Dict[str, object] = {
        "n": int(len(yt)),
        "accuracy": float(accuracy_score(yt, yp)),
        # macro scores average only over classes that occur in this split
        "precision": float(prec[present].mean()) if present.any() else 0.0,
        "recall": float(rec[present].mean()) if present.any() else 0.0,
        "f1": float(f1[present].mean()) if present.any() else 0.0,
        "f1_weighted": float(f1_score(yt, yp, average="weighted", zero_division=0)),
        "ordinal_mae": float(np.mean(np.abs(yt - yp))),          # how many classes off, on average
        "risky_recall": float(np.mean(yp[yt > 0] > 0)) if (yt > 0).any() else float("nan"),
        "roc_auc": _safe_auc(yt, np.asarray(probs)) if probs is not None else float("nan"),
        "ttc_mae_s": float("nan"),
        "per_class": {
            RISK_LABELS[c]: {"precision": float(prec[c]), "recall": float(rec[c]),
                             "f1": float(f1[c]), "support": int(support[c])}
            for c in labels
        },
        "confusion_matrix": confusion_matrix(yt, yp, labels=labels).tolist(),
    }
    if pred_ttc is not None and true_ttc is not None:
        p, t = np.asarray(pred_ttc, dtype=float), np.asarray(true_ttc, dtype=float)
        mask = ~np.isnan(t)
        if mask.any():
            out["ttc_mae_s"] = float(np.mean(np.abs(p[mask] - t[mask])))
    return out
