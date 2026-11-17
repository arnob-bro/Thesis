"""Per-sequence graph construction.

For each sequence directory produced by `sequence_generator`, this script:
 - loads `tracks.json` (output of per-sequence tracking)
 - computes per-frame object features (position, velocity, acceleration, class)
 - builds interaction features (proximity, rel_velocity_long, rel_velocity_lat, ttc)
 - constructs a `torch_geometric.data.Data` object per frame and saves the
   list to `graph.pt` inside the sequence folder

This implements the requested ordering: tracking -> object features ->
interaction features -> graph construction.
"""
from __future__ import annotations

import argparse
import json
import math
from collections import defaultdict
from pathlib import Path
from typing import Dict, List

import numpy as np
import torch
from torch_geometric.data import Data

from features.object_features import compute_object_features, normalize_node_features
from features.interaction_features import build_edges_knn, normalize_edge_features
from utils.config import Config, load_config
from utils.logger import get_logger
from utils.seed import set_seed

LOGGER = get_logger("graph_per_sequence")


def load_tracks(path: Path) -> Dict[int, List[dict]]:
    if not path.exists():
        return {}
    with path.open("r", encoding="utf-8") as fp:
        raw = json.load(fp)
    grouped: Dict[int, List[dict]] = defaultdict(list)
    for d in raw:
        grouped[int(d["frame_id"])].append(d)
    return dict(grouped)


def build_frame_graph_from_features(
    features, image_size, num_classes, k_neighbors, edge_dim
) -> Data:
    n = len(features)
    if n == 0:
        x = torch.zeros(1, 8, dtype=torch.float32)
        edge_index = torch.zeros(2, 0, dtype=torch.long)
        edge_attr = torch.zeros(0, edge_dim, dtype=torch.float32)
        track_id = torch.full((1,), -1, dtype=torch.long)
        return Data(
            x=x,
            edge_index=edge_index,
            edge_attr=edge_attr,
            num_nodes=1,
            track_id=track_id,
        )

    x_np = normalize_node_features(features, image_size, num_classes)
    image_diag = math.hypot(image_size[0], image_size[1])
    edges, attrs = build_edges_knn(features, k_neighbors, image_diag)
    edge_attr_np = normalize_edge_features(attrs, image_diag)

    if not edges:
        edge_index = torch.zeros(2, 0, dtype=torch.long)
        edge_attr = torch.zeros(0, edge_dim, dtype=torch.float32)
    else:
        edge_index = torch.tensor(edges, dtype=torch.long).t().contiguous()
        edge_attr = torch.from_numpy(edge_attr_np)

    x = torch.from_numpy(x_np)
    # Preserve original track IDs for visualization downstream
    track_ids = [int(f.track_id) for f in features]
    track_id_tensor = torch.tensor(track_ids, dtype=torch.long)
    return Data(
        x=x,
        edge_index=edge_index,
        edge_attr=edge_attr,
        num_nodes=x.shape[0],
        track_id=track_id_tensor,
    )


def run(cfg: Config, split: str) -> None:
    set_seed(cfg.seed)
    seq_root = Path(cfg.paths.sequences) / split
    if not seq_root.exists():
        raise FileNotFoundError(f"Sequences root missing: {seq_root}. Run sequence_generator first.")

    sequences = sorted(p for p in seq_root.iterdir() if p.is_dir())
    LOGGER.info("Building graphs for %d sequences (split=%s)", len(sequences), split)

    for seq_dir in sequences:
        metadata_path = seq_dir / "metadata.json"
        if not metadata_path.exists():
            LOGGER.warning("No metadata for %s, skipping", seq_dir)
            continue
        with metadata_path.open("r", encoding="utf-8") as fp:
            meta = json.load(fp)

        image_size = tuple(meta.get("image_size", cfg.frame_extraction.resize))
        frame_ids: List[int] = list(meta.get("frame_ids", []))

        tracks_path = seq_dir / "tracks.json"
        tracks_by_frame_raw = load_tracks(tracks_path)

        # Convert raw tracks into TrackPoint dataclass objects expected by
        # compute_object_features. The TrackPoint class is defined in
        # features.object_features and expected fields: frame_id, track_id,
        # class_id, bbox.
        from features.object_features import TrackPoint

        tracks_by_frame: Dict[int, List[TrackPoint]] = {}
        for fid, raw_list in tracks_by_frame_raw.items():
            pts: List[TrackPoint] = []
            for r in raw_list:
                pts.append(
                    TrackPoint(
                        frame_id=int(r["frame_id"]),
                        track_id=int(r["track_id"]),
                        class_id=int(r.get("class_id", 0)),
                        bbox=[float(v) for v in r["bbox"]],
                    )
                )
            tracks_by_frame[int(fid)] = pts

        obj_features = compute_object_features(
            tracks_by_frame,
            fps=float(cfg.frame_extraction.frame_rate),
            clamp_velocity=cfg.features.clamp_velocity,
            clamp_acceleration=cfg.features.clamp_acceleration,
        )

        graphs: List[Data] = []
        for fid in frame_ids:
            feats = obj_features.get(fid, [])
            g = build_frame_graph_from_features(
                features=feats,
                image_size=image_size,
                num_classes=cfg.features.num_object_classes,
                k_neighbors=cfg.graph.k_neighbors,
                edge_dim=cfg.features.edge_dim,
            )
            graphs.append(g)

        torch.save(graphs, seq_dir / "graph.pt")
        LOGGER.info("Saved graph.pt for %s (%d frames)", seq_dir.name, len(graphs))


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Per-sequence graph builder")
    parser.add_argument("--config", type=str, required=True)
    parser.add_argument("--split", choices=["train", "val", "test"], required=True)
    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()
    run(load_config(args.config), args.split)
