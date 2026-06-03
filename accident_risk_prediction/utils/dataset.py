"""Sequence-level dataset loader for the multimodal accident risk model.

Each sample is a *sequence* (the MD specifies SEQUENCE as the training unit,
never an isolated frame). A sequence directory follows the schema:

    sequences/<split>/sequence_<id>/
        frames/frame_0001.jpg ... frame_000T.jpg
        graph.pt                  # list[T] of torch_geometric.data.Data
        ego_state.json            # aligned ego telemetry per frame
        label.json                # {risk_level, ttc_estimate, ...}
        metadata.json
"""
from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Dict, List, Tuple

import cv2
import numpy as np
import torch
from torch.utils.data import Dataset
from torch_geometric.data import Batch, Data

from utils.config import Config


# ---------------------------------------------------------------------------
# I/O helpers
# ---------------------------------------------------------------------------

def _read_image(path: Path, size: Tuple[int, int]) -> np.ndarray:
    img = cv2.imread(str(path))
    if img is None:
        raise FileNotFoundError(f"Image missing: {path}")
    img = cv2.resize(img, size)
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    return img


def _normalise_image(img: np.ndarray) -> torch.Tensor:
    """HWC uint8 -> CHW float32 in [0,1], ImageNet-normalised."""
    tensor = torch.from_numpy(img).float().div(255.0).permute(2, 0, 1)
    mean = torch.tensor([0.485, 0.456, 0.406]).view(3, 1, 1)
    std = torch.tensor([0.229, 0.224, 0.225]).view(3, 1, 1)
    return (tensor - mean) / std


def _load_ego_state(
    path: Path, num_frames: int, feature_keys: List[str]
) -> torch.Tensor:
    """Load ego telemetry aligned to frames, padded/truncated to num_frames."""
    if not path.exists():
        return torch.zeros(num_frames, len(feature_keys), dtype=torch.float32)
    with path.open("r", encoding="utf-8") as fp:
        data = json.load(fp)

    cols: List[List[float]] = []
    for key in feature_keys:
        series = data.get(key, [])
        if len(series) >= num_frames:
            cols.append(list(series[:num_frames]))
        else:
            pad_val = float(series[-1]) if series else 0.0
            cols.append(list(series) + [pad_val] * (num_frames - len(series)))
    arr = np.asarray(cols, dtype=np.float32).T  # [T, D]
    return torch.from_numpy(arr)


# ---------------------------------------------------------------------------
# Dataset
# ---------------------------------------------------------------------------

class SequenceDataset(Dataset):
    """Loads per-sequence multimodal samples produced by `pipelines/`."""

    def __init__(self, cfg: Config, split: str) -> None:
        if split not in {"train", "val", "test"}:
            raise ValueError(f"Unknown split: {split}")
        self.cfg = cfg
        self.split = split
        self.sequence_length = int(cfg.sequence.length)
        self.image_size = tuple(cfg.frame_extraction.resize)  # (W, H)
        self.vit_image_size = int(cfg.vit.image_size)
        self.ego_feature_keys: List[str] = list(cfg.ego_encoder.input_features)
        self.ego_dim = int(cfg.ego_encoder.input_dim)

        seq_root = Path(cfg.paths.sequences) / split
        if not seq_root.exists():
            raise FileNotFoundError(
                f"Sequence root not found for split '{split}': {seq_root}"
            )
        self.sequence_dirs: List[Path] = sorted(
            p for p in seq_root.iterdir() if p.is_dir()
        )
        if not self.sequence_dirs:
            raise RuntimeError(f"No sequences found under {seq_root}")

    def __len__(self) -> int:
        return len(self.sequence_dirs)

    # ---------------------------------------------------------------- private
    def _load_label(self, seq_dir: Path) -> Tuple[int, float]:
        label_path = seq_dir / "label.json"
        with label_path.open("r", encoding="utf-8") as fp:
            payload = json.load(fp)
        risk = int(payload.get("risk_level", 0))
        ttc = float(payload.get("ttc_estimate", payload.get("ttc", float("nan"))))
        return risk, ttc

    def _load_frames(self, seq_dir: Path) -> torch.Tensor:
        frame_dir = seq_dir / "frames"
        files = sorted(frame_dir.glob("*.jpg"))
        if len(files) == 0:
            raise FileNotFoundError(f"No frames in {frame_dir}")
        if len(files) >= self.sequence_length:
            files = files[: self.sequence_length]
        else:
            files = files + [files[-1]] * (self.sequence_length - len(files))

        # Resize to ViT input size so we don't need a second resize downstream.
        size = (self.vit_image_size, self.vit_image_size)
        frames = [_normalise_image(_read_image(f, size)) for f in files]
        return torch.stack(frames, dim=0)  # [T, 3, H, W]

    def _load_graphs(self, seq_dir: Path) -> List[Data]:
        graph_path = seq_dir / "graph.pt"
        if not graph_path.exists():
            raise FileNotFoundError(f"graph.pt missing: {graph_path}")
        graphs = torch.load(graph_path, weights_only=False)
        if not isinstance(graphs, list):
            raise TypeError(
                f"graph.pt must contain list[Data]; got {type(graphs).__name__}"
            )
        if len(graphs) >= self.sequence_length:
            graphs = graphs[: self.sequence_length]
        else:
            graphs = graphs + [graphs[-1]] * (self.sequence_length - len(graphs))
        return graphs

    # ---------------------------------------------------------------- public
    def __getitem__(self, idx: int) -> Dict[str, Any]:
        seq_dir = self.sequence_dirs[idx]
        frames = self._load_frames(seq_dir)
        graphs = self._load_graphs(seq_dir)
        ego = _load_ego_state(
            seq_dir / "ego_state.json", self.sequence_length, self.ego_feature_keys
        )
        risk, ttc = self._load_label(seq_dir)

        return {
            "sequence_id": seq_dir.name,
            "frames": frames,            # [T, 3, H, W]
            "graphs": graphs,            # list[Data], length T
            "ego_state": ego,            # [T, D_ego]
            "risk_level": torch.tensor(risk, dtype=torch.long),
            "ttc": torch.tensor(
                ttc if not np.isnan(ttc) else 0.0, dtype=torch.float32
            ),
            "ttc_valid": torch.tensor(
                0.0 if np.isnan(ttc) else 1.0, dtype=torch.float32
            ),
        }


# ---------------------------------------------------------------------------
# Collate
# ---------------------------------------------------------------------------

def collate_sequences(batch: List[Dict[str, Any]]) -> Dict[str, Any]:
    """Custom collate.

    For graphs we build one PyG `Batch` per frame index so the GNN can be run
    once per timestep with proper batch indexing.
    """
    batch_size = len(batch)
    seq_len = batch[0]["frames"].shape[0]

    frames = torch.stack([b["frames"] for b in batch], dim=0)        # [B, T, C, H, W]
    ego = torch.stack([b["ego_state"] for b in batch], dim=0)        # [B, T, D_ego]
    risk = torch.stack([b["risk_level"] for b in batch], dim=0)      # [B]
    ttc = torch.stack([b["ttc"] for b in batch], dim=0)              # [B]
    ttc_valid = torch.stack([b["ttc_valid"] for b in batch], dim=0)  # [B]

    graphs_per_t: List[Batch] = []
    for t in range(seq_len):
        graphs_t = [b["graphs"][t] for b in batch]
        graphs_per_t.append(Batch.from_data_list(graphs_t))

    return {
        "sequence_ids": [b["sequence_id"] for b in batch],
        "frames": frames,
        "graphs_per_t": graphs_per_t,
        "ego_state": ego,
        "risk_level": risk,
        "ttc": ttc,
        "ttc_valid": ttc_valid,
        "batch_size": batch_size,
        "seq_len": seq_len,
    }
