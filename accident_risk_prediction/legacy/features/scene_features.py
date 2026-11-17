"""Scene-level feature helpers.

The actual ViT forward pass lives in `models/vit/vit_model.py` (per MD: frozen).
This module just defines a thin caching helper for storing precomputed scene
embeddings to `data/features/scene_embeddings/`.
"""
from __future__ import annotations

from pathlib import Path
from typing import Optional

import numpy as np


def cache_path_for(cache_root: str | Path, sequence_id: str, frame_id: int) -> Path:
    return Path(cache_root) / sequence_id / f"scene_{frame_id:06d}.npy"


def save_scene_embedding(
    cache_root: str | Path, sequence_id: str, frame_id: int, embedding: np.ndarray,
) -> None:
    path = cache_path_for(cache_root, sequence_id, frame_id)
    path.parent.mkdir(parents=True, exist_ok=True)
    np.save(path, embedding.astype(np.float32))


def load_scene_embedding(
    cache_root: str | Path, sequence_id: str, frame_id: int,
) -> Optional[np.ndarray]:
    path = cache_path_for(cache_root, sequence_id, frame_id)
    if not path.exists():
        return None
    return np.load(path)
