"""Evaluation metrics for the accident risk prediction system.

Implements every metric required by the MD specification:
- Accuracy / Precision / Recall / F1
- ROC-AUC (one-vs-rest, multiclass)
- Time-to-Accident (TTA)
- Early Risk Detection Score
- Confusion matrix
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Dict, List, Sequence

import numpy as np
from sklearn.metrics import (
    accuracy_score,
    confusion_matrix,
    f1_score,
    precision_score,
    recall_score,
    roc_auc_score,
)

# Risk semantics (MD)
RISK_LABELS = ["safe", "slight", "moderate", "high"]
RISKY_CLASSES = {1, 2, 3}  # anything > 0 is "risky"


@dataclass
class MetricResult:
    accuracy: float = 0.0
    precision: float = 0.0
    recall: float = 0.0
    f1: float = 0.0
    roc_auc: float = float("nan")
    tta: float = float("nan")
    early_risk_detection: float = 0.0
    per_class: Dict[str, Dict[str, float]] = field(default_factory=dict)
    confusion: List[List[int]] = field(default_factory=list)

    def to_dict(self) -> Dict[str, object]:
        return {
            "accuracy": self.accuracy,
            "precision": self.precision,
            "recall": self.recall,
            "f1": self.f1,
            "roc_auc": self.roc_auc,
            "time_to_accident": self.tta,
            "early_risk_detection": self.early_risk_detection,
            "per_class": self.per_class,
            "confusion_matrix": self.confusion,
        }


def _safe_roc_auc(y_true: np.ndarray, y_score: np.ndarray, num_classes: int) -> float:
    """One-vs-rest multiclass AUC. Returns NaN if undefined."""
    try:
        present = np.unique(y_true)
        if len(present) < 2:
            return float("nan")
        if y_score.shape[1] != num_classes:
            return float("nan")
        return float(roc_auc_score(y_true, y_score, multi_class="ovr"))
    except ValueError:
        return float("nan")


def classification_metrics(
    y_true: Sequence[int],
    y_pred: Sequence[int],
    y_score: np.ndarray | None = None,
    num_classes: int = 4,
) -> MetricResult:
    y_true_arr = np.asarray(y_true)
    y_pred_arr = np.asarray(y_pred)

    result = MetricResult()
    result.accuracy = float(accuracy_score(y_true_arr, y_pred_arr))
    result.precision = float(
        precision_score(y_true_arr, y_pred_arr, average="macro", zero_division=0)
    )
    result.recall = float(
        recall_score(y_true_arr, y_pred_arr, average="macro", zero_division=0)
    )
    result.f1 = float(f1_score(y_true_arr, y_pred_arr, average="macro", zero_division=0))

    if y_score is not None:
        result.roc_auc = _safe_roc_auc(y_true_arr, np.asarray(y_score), num_classes)

    labels = list(range(num_classes))
    cm = confusion_matrix(y_true_arr, y_pred_arr, labels=labels)
    result.confusion = cm.tolist()

    per_class: Dict[str, Dict[str, float]] = {}
    for c in labels:
        name = RISK_LABELS[c] if c < len(RISK_LABELS) else str(c)
        per_class[name] = {
            "precision": float(
                precision_score(
                    y_true_arr, y_pred_arr, labels=[c], average="macro", zero_division=0
                )
            ),
            "recall": float(
                recall_score(
                    y_true_arr, y_pred_arr, labels=[c], average="macro", zero_division=0
                )
            ),
            "f1": float(
                f1_score(
                    y_true_arr, y_pred_arr, labels=[c], average="macro", zero_division=0
                )
            ),
        }
    result.per_class = per_class
    return result


def time_to_accident(
    pred_ttc: Sequence[float],
    true_ttc: Sequence[float],
    risky_mask: Sequence[bool] | None = None,
) -> float:
    """Mean absolute error between predicted and ground-truth TTC over risky cases."""
    pred = np.asarray(pred_ttc, dtype=np.float64)
    true = np.asarray(true_ttc, dtype=np.float64)
    if risky_mask is None:
        mask = ~np.isnan(true)
    else:
        mask = np.asarray(risky_mask) & ~np.isnan(true)
    if mask.sum() == 0:
        return float("nan")
    return float(np.mean(np.abs(pred[mask] - true[mask])))


def early_risk_detection_score(
    y_true_seq: Sequence[Sequence[int]],
    y_pred_seq: Sequence[Sequence[int]],
    fps: float = 5.0,
) -> float:
    """Average time (in seconds) by which the model detects risk earlier than
    the ground truth onset, averaged over sequences containing an event.

    A higher value is better; zero means simultaneous detection; negative means
    the model is late.
    """
    if len(y_true_seq) != len(y_pred_seq):
        raise ValueError("y_true_seq and y_pred_seq lengths differ")

    leads: List[float] = []
    for true, pred in zip(y_true_seq, y_pred_seq):
        true_arr = np.asarray(true)
        pred_arr = np.asarray(pred)
        risky_true = np.where(true_arr > 0)[0]
        if len(risky_true) == 0:
            continue
        gt_onset = int(risky_true[0])
        risky_pred = np.where(pred_arr > 0)[0]
        if len(risky_pred) == 0:
            leads.append(-(len(true_arr) - gt_onset) / fps)
            continue
        pred_onset = int(risky_pred[0])
        leads.append((gt_onset - pred_onset) / fps)

    if not leads:
        return 0.0
    return float(np.mean(leads))


def compute_all_metrics(
    y_true: Sequence[int],
    y_pred: Sequence[int],
    y_score: np.ndarray | None,
    pred_ttc: Sequence[float] | None = None,
    true_ttc: Sequence[float] | None = None,
    y_true_seq: Sequence[Sequence[int]] | None = None,
    y_pred_seq: Sequence[Sequence[int]] | None = None,
    fps: float = 5.0,
    num_classes: int = 4,
) -> MetricResult:
    result = classification_metrics(y_true, y_pred, y_score, num_classes=num_classes)
    if pred_ttc is not None and true_ttc is not None:
        risky_mask = [c > 0 for c in y_true]
        result.tta = time_to_accident(pred_ttc, true_ttc, risky_mask)
    if y_true_seq is not None and y_pred_seq is not None:
        result.early_risk_detection = early_risk_detection_score(
            y_true_seq, y_pred_seq, fps=fps
        )
    return result
