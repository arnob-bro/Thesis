"""Visualization helpers - bounding boxes, tracking IDs, graph edges, risk curve.

All functions return a numpy image (BGR) or save to disk.
"""
from __future__ import annotations

from pathlib import Path
from typing import Dict, Iterable, List, Sequence, Tuple

import cv2
import matplotlib.pyplot as plt
import numpy as np

_COLOR_TABLE = [
    (255, 56, 56),
    (255, 159, 56),
    (255, 224, 56),
    (56, 255, 132),
    (56, 226, 255),
    (56, 89, 255),
    (180, 56, 255),
    (255, 56, 224),
]


def _color_for_id(track_id: int) -> Tuple[int, int, int]:
    return _COLOR_TABLE[track_id % len(_COLOR_TABLE)]


def draw_detections(
    image: np.ndarray,
    detections: Iterable[Dict],
    class_names: Dict[int, str] | None = None,
) -> np.ndarray:
    out = image.copy()
    for det in detections:
        x, y, w, h = det["bbox"]
        x1, y1, x2, y2 = int(x), int(y), int(x + w), int(y + h)
        cls = int(det.get("class_id", 0))
        conf = float(det.get("confidence", 0.0))
        color = _color_for_id(cls)
        cv2.rectangle(out, (x1, y1), (x2, y2), color, 2)
        name = class_names.get(cls, str(cls)) if class_names else str(cls)
        cv2.putText(
            out, f"{name} {conf:.2f}", (x1, max(0, y1 - 6)),
            cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 1, cv2.LINE_AA,
        )
    return out


def draw_tracks(image: np.ndarray, tracks: Iterable[Dict]) -> np.ndarray:
    out = image.copy()
    for trk in tracks:
        x, y, w, h = trk["bbox"]
        x1, y1, x2, y2 = int(x), int(y), int(x + w), int(y + h)
        tid = int(trk["track_id"])
        color = _color_for_id(tid)
        cv2.rectangle(out, (x1, y1), (x2, y2), color, 2)
        cv2.putText(
            out, f"ID {tid}", (x1, max(0, y1 - 6)),
            cv2.FONT_HERSHEY_SIMPLEX, 0.6, color, 2, cv2.LINE_AA,
        )
    return out


def draw_graph_edges(
    image: np.ndarray,
    nodes_xy: Sequence[Tuple[float, float]],
    edges: Sequence[Tuple[int, int]],
    edge_attr: Sequence[Sequence[float]] | None = None,
    node_labels: Sequence[str] | None = None,
) -> np.ndarray:
    out = image.copy()
    for idx, (s, t) in enumerate(edges):
        if s >= len(nodes_xy) or t >= len(nodes_xy):
            continue
        p1 = (int(nodes_xy[s][0]), int(nodes_xy[s][1]))
        p2 = (int(nodes_xy[t][0]), int(nodes_xy[t][1]))
        cv2.line(out, p1, p2, (0, 255, 255), 1, cv2.LINE_AA)
        if edge_attr is not None and idx < len(edge_attr):
            values = edge_attr[idx]
            mx = int((p1[0] + p2[0]) / 2)
            my = int((p1[1] + p2[1]) / 2)
            if len(values) >= 4:
                proximity, rv_long, rv_lat, ttc = values[:4]
                label = f"p:{proximity:.2f} t:{ttc:.1f}"
            elif len(values) >= 2:
                d, ttc = values[0], values[1]
                label = f"d:{d:.0f} t:{ttc:.1f}"
            else:
                label = str(values)
            cv2.putText(
                out, label, (mx + 4, my - 4),
                cv2.FONT_HERSHEY_SIMPLEX, 0.35, (0, 255, 255), 1, cv2.LINE_AA,
            )
    for idx, (cx, cy) in enumerate(nodes_xy):
        cv2.circle(out, (int(cx), int(cy)), 4, (0, 255, 0), -1)
        if node_labels is not None and idx < len(node_labels):
            cv2.putText(
                out, node_labels[idx], (int(cx) + 6, int(cy) + 6),
                cv2.FONT_HERSHEY_SIMPLEX, 0.35, (0, 255, 0), 1, cv2.LINE_AA,
            )
        else:
            cv2.putText(
                out, str(idx), (int(cx) + 6, int(cy) + 6),
                cv2.FONT_HERSHEY_SIMPLEX, 0.4, (0, 255, 0), 1, cv2.LINE_AA,
            )
    return out


def plot_risk_curve(
    risk_per_frame: Sequence[int],
    save_path: str | Path,
    fps: float = 5.0,
    title: str = "Predicted Risk Over Time",
) -> None:
    times = np.arange(len(risk_per_frame)) / fps
    fig, ax = plt.subplots(figsize=(8, 3))
    ax.plot(times, list(risk_per_frame), marker="o")
    ax.set_xlabel("Time (s)")
    ax.set_ylabel("Risk Level")
    ax.set_ylim(-0.5, 3.5)
    ax.set_yticks([0, 1, 2, 3])
    ax.set_yticklabels(["safe", "slight", "moderate", "high"])
    ax.set_title(title)
    ax.grid(True, alpha=0.3)
    save_path = Path(save_path)
    save_path.parent.mkdir(parents=True, exist_ok=True)
    fig.tight_layout()
    fig.savefig(save_path, dpi=150)
    plt.close(fig)


def plot_confusion_matrix(
    cm: List[List[int]],
    save_path: str | Path,
    labels: Sequence[str] = ("safe", "slight", "moderate", "high"),
) -> None:
    arr = np.asarray(cm, dtype=np.float32)
    fig, ax = plt.subplots(figsize=(5, 4.2))
    im = ax.imshow(arr, cmap="Blues")
    ax.set_xticks(range(len(labels)))
    ax.set_yticks(range(len(labels)))
    ax.set_xticklabels(labels, rotation=30, ha="right")
    ax.set_yticklabels(labels)
    ax.set_xlabel("Predicted")
    ax.set_ylabel("True")
    thr = arr.max() / 2.0 if arr.size else 0.0
    for i in range(arr.shape[0]):
        for j in range(arr.shape[1]):
            ax.text(
                j, i, int(arr[i, j]),
                ha="center", va="center",
                color="white" if arr[i, j] > thr else "black",
            )
    fig.colorbar(im, ax=ax)
    fig.tight_layout()
    save_path = Path(save_path)
    save_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(save_path, dpi=150)
    plt.close(fig)
