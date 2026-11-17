"""Dataset that reads sequences from the manifest and the per-video cache."""
from __future__ import annotations

import csv
from pathlib import Path
from typing import Any, Dict, List, Optional

import numpy as np
import torch
from torch.utils.data import Dataset
from torch_geometric.data import Batch

from features.ego_motion import FLIP_NEGATE
from features.graph_builder import VideoGraphs
from utils.config import Config


def read_manifest(path: str | Path) -> List[Dict[str, Any]]:
    path = Path(path)
    if not path.exists():
        raise FileNotFoundError(
            f"Manifest not found: {path}. Run tools/mark_collisions.py, "
            "pipelines/build_video_cache.py and tools/make_event_sequences.py first."
        )
    rows = []
    with path.open("r", encoding="utf-8-sig", newline="") as fp:
        for r in csv.DictReader(fp):
            r["fold"], r["start"], r["end"] = int(r["fold"]), int(r["start"]), int(r["end"])
            r["risk_level"] = int(r["risk_level"])
            r["tte"] = float(r["tte"]) if r["tte"] != "" else None
            rows.append(r)
    return rows


def split_rows(rows: List[Dict], cfg: Config, fold: Optional[int]) -> Dict[str, List[Dict]]:
    """fold=None -> use the `split` column. Otherwise test = fold, val = fold+1, train = rest."""
    if fold is None:
        return {s: [r for r in rows if r["split"] == s] for s in ("train", "val", "test")}
    k = int(cfg.cv.folds)
    val_fold = (fold + 1) % k
    return {
        "train": [r for r in rows if r["fold"] not in (fold, val_fold)],
        "val": [r for r in rows if r["fold"] == val_fold],
        "test": [r for r in rows if r["fold"] == fold],
    }


class VideoStore:
    """Loads each video's cached arrays once and keeps them in memory."""

    def __init__(self, cfg: Config) -> None:
        self.root = Path(cfg.paths.video_cache)
        self._cache: Dict[str, Dict[str, Any]] = {}

    def get(self, video_id: str) -> Dict[str, Any]:
        if video_id not in self._cache:
            d = self.root / video_id
            for f in ("vit.npy", "graphs.npz", "ego_motion.npz"):
                if not (d / f).exists():
                    raise FileNotFoundError(
                        f"{d / f} missing. Run: python -m pipelines.build_video_cache "
                        "--config configs/config.yaml --videos <folder>"
                    )
            flip = d / "vit_flip.npy"
            self._cache[video_id] = {
                "vit": np.load(d / "vit.npy", mmap_mode="r"),
                "vit_flip": np.load(flip, mmap_mode="r") if flip.exists() else None,
                "graphs": VideoGraphs(d / "graphs.npz"),
                "ego": np.load(d / "ego_motion.npz")["feats"],
            }
        return self._cache[video_id]


class ManifestDataset(Dataset):
    def __init__(self, cfg: Config, rows: List[Dict], store: Optional[VideoStore] = None,
                 augment: bool = False) -> None:
        self.rows = rows
        self.store = store or VideoStore(cfg)
        self.flip_prob = float(cfg.training.flip_prob) if augment else 0.0
        self.tte_scale = float(cfg.event_labeling.safe_min_tte)

    def __len__(self) -> int:
        return len(self.rows)

    def __getitem__(self, idx: int) -> Dict[str, Any]:
        r = self.rows[idx]
        v = self.store.get(r["video_id"])
        s, e = r["start"], r["end"] + 1
        flip = self.flip_prob > 0 and v["vit_flip"] is not None and np.random.rand() < self.flip_prob

        vit = torch.from_numpy(np.asarray((v["vit_flip"] if flip else v["vit"])[s:e], dtype=np.float32))
        ego = torch.from_numpy(np.array(v["ego"][s:e], dtype=np.float32))
        if flip:
            ego[:, list(FLIP_NEGATE)] *= -1.0
        graphs = [v["graphs"].get(k, flip) for k in range(s, e)]

        risk = int(r.get("risk_level") or 0)
        has_ttc = r.get("tte") is not None and risk > 0
        return {
            "row": r, "vit": vit, "ego": ego, "graphs": graphs,
            "risk": torch.tensor(risk, dtype=torch.long),
            "ttc": torch.tensor(r["tte"] / self.tte_scale if has_ttc else 0.0, dtype=torch.float32),
            "ttc_valid": torch.tensor(1.0 if has_ttc else 0.0, dtype=torch.float32),
        }


def collate(batch: List[Dict[str, Any]]) -> Dict[str, Any]:
    """Graphs are flattened sample-major: graph index = b * T + t."""
    return {
        "rows": [b["row"] for b in batch],
        "vit": torch.stack([b["vit"] for b in batch]),            # [B, T, 768]
        "ego": torch.stack([b["ego"] for b in batch]),            # [B, T, 6]
        "graphs": Batch.from_data_list([g for b in batch for g in b["graphs"]]),
        "risk": torch.stack([b["risk"] for b in batch]),
        "ttc": torch.stack([b["ttc"] for b in batch]),
        "ttc_valid": torch.stack([b["ttc_valid"] for b in batch]),
    }
