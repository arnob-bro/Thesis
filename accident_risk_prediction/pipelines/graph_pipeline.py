"""Stage 4: convert tracked detections into per-frame graphs and assemble
sequence directories ready for `utils.dataset.SequenceDataset`.

Per the MD spec:
    - 1 frame  ==  1 graph (no cross-frame graph edges; temporal modelling is
      handled by the Temporal Transformer)
    - Graph nodes:  external traffic participants (ego excluded)
    - Graph edges:  K-nearest neighbours (K=3..5)
    - Edge features:  [proximity, rel_velocity_long, rel_velocity_lat, ttc]
    - Node features:  [x, y, w, h, vel, acc, dir, object_type]

Output (per video, per sequence):
    data/sequences/<split>/<video>_seq<N>/
        frames/frame_0001.jpg ... frame_000T.jpg
        graph.pt           # list[T] of torch_geometric.data.Data
        label.json
        ego_state.json     # placeholder if no external ego file present
        metadata.json
"""
from __future__ import annotations

import argparse
import json
import math
import shutil
from collections import defaultdict
from pathlib import Path
from typing import Dict, List

import numpy as np
import torch
from torch_geometric.data import Data

from features.interaction_features import (
    build_edges_knn,
    normalize_edge_features,
)
from features.object_features import (
    TrackPoint,
    compute_object_features,
    normalize_node_features,
)
from utils.config import Config, load_config
from utils.logger import get_logger
from utils.seed import set_seed

LOGGER = get_logger("graph_pipeline")


def load_tracks(path: Path) -> Dict[int, List[TrackPoint]]:
    with path.open("r", encoding="utf-8") as fp:
        raw = json.load(fp)
    grouped: Dict[int, List[TrackPoint]] = defaultdict(list)
    for d in raw:
        grouped[int(d["frame_id"])].append(
            TrackPoint(
                frame_id=int(d["frame_id"]),
                track_id=int(d["track_id"]),
                class_id=int(d["class_id"]),
                bbox=[float(v) for v in d["bbox"]],
            )
        )
    return dict(grouped)


def build_frame_graph(
    features: List,
    image_size,
    num_classes: int,
    k_neighbors: int,
    edge_dim: int,
) -> Data:
    n = len(features)
    if n == 0:
        # Empty frame: emit a single dummy isolated node so PyG batching works.
        x = torch.zeros(1, 8, dtype=torch.float32)
        edge_index = torch.zeros(2, 0, dtype=torch.long)
        edge_attr = torch.zeros(0, edge_dim, dtype=torch.float32)
        return Data(x=x, edge_index=edge_index, edge_attr=edge_attr, num_nodes=1)

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
    return Data(x=x, edge_index=edge_index, edge_attr=edge_attr, num_nodes=x.shape[0])


def find_label_file(
    risk_root: Path, video_name: str, sequence_index: int,
) -> Path | None:
    """Risk labels are sequence-level; we look in two conventional locations."""
    candidates = [
        risk_root / video_name / f"sequence_{sequence_index:04d}.json",
        risk_root / f"{video_name}_seq{sequence_index:04d}.json",
    ]
    for c in candidates:
        if c.exists():
            return c
    return None


def find_ego_state_file(
    ego_root: Path, video_name: str, sequence_index: int,
) -> Path | None:
    candidates = [
        ego_root / video_name / f"sequence_{sequence_index:04d}.json",
        ego_root / video_name / f"sequence_{sequence_index:04d}.csv",
        ego_root / f"{video_name}_seq{sequence_index:04d}.json",
        ego_root / f"{video_name}_seq{sequence_index:04d}.csv",
    ]
    for c in candidates:
        if c.exists():
            return c
    return None


def materialise_ego_state(
    ego_src: Path | None,
    feature_keys: List[str],
    seq_len: int,
    out_path: Path,
) -> None:
    """Normalise ego state into the JSON-of-lists schema expected by the
    dataset loader. Falls back to zeros if no source is available."""
    if ego_src is None:
        payload = {k: [0.0] * seq_len for k in feature_keys}
    elif ego_src.suffix == ".json":
        with ego_src.open("r", encoding="utf-8") as fp:
            raw = json.load(fp)
        payload = {}
        for k in feature_keys:
            series = list(raw.get(k, []))
            if len(series) < seq_len:
                series = series + [series[-1] if series else 0.0] * (seq_len - len(series))
            payload[k] = series[:seq_len]
    elif ego_src.suffix == ".csv":
        import csv
        cols: Dict[str, List[float]] = {k: [] for k in feature_keys}
        with ego_src.open("r", encoding="utf-8") as fp:
            reader = csv.DictReader(fp)
            for row in reader:
                for k in feature_keys:
                    try:
                        cols[k].append(float(row.get(k, 0.0)))
                    except ValueError:
                        cols[k].append(0.0)
        payload = {}
        for k in feature_keys:
            series = cols[k]
            if len(series) < seq_len:
                series = series + [series[-1] if series else 0.0] * (seq_len - len(series))
            payload[k] = series[:seq_len]
    else:
        raise ValueError(f"Unsupported ego state extension: {ego_src.suffix}")

    out_path.parent.mkdir(parents=True, exist_ok=True)
    with out_path.open("w", encoding="utf-8") as fp:
        json.dump(payload, fp)


def materialise_label(
    label_src: Path | None,
    seq_id: str,
    out_path: Path,
) -> None:
    if label_src is None:
        payload = {"sequence_id": seq_id, "risk_level": 0, "ttc_estimate": float("nan")}
    else:
        with label_src.open("r", encoding="utf-8") as fp:
            payload = json.load(fp)
        payload.setdefault("sequence_id", seq_id)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with out_path.open("w", encoding="utf-8") as fp:
        json.dump(payload, fp, indent=2)


def process_video(
    video_name: str,
    frames_dir: Path,
    tracks_path: Path,
    cfg: Config,
    split: str,
) -> int:
    tracks_by_frame = load_tracks(tracks_path)
    if not tracks_by_frame:
        LOGGER.warning("No tracks for video %s, skipping", video_name)
        return 0

    obj_features = compute_object_features(
        tracks_by_frame,
        fps=float(cfg.frame_extraction.frame_rate),
        clamp_velocity=cfg.features.clamp_velocity,
        clamp_acceleration=cfg.features.clamp_acceleration,
    )

    def frame_id_from_path(frame_path: Path) -> int:
        stem = frame_path.stem
        if stem.isdigit():
            return int(stem)
        parts = stem.split("_")
        if parts and parts[-1].isdigit():
            return int(parts[-1])
        raise ValueError(
            f"Unable to parse frame_id from file name: {frame_path.name}"
        )

    frame_entries = sorted(
        (frame_id_from_path(fp), fp) for fp in frames_dir.glob("*.jpg")
    )
    frame_ids = [fid for fid, _ in frame_entries]
    frame_files = [fp for _, fp in frame_entries]
    if not frame_files:
        LOGGER.warning("No extracted frames for %s, skipping", video_name)
        return 0

    seq_len = int(cfg.sequence.length)
    stride = int(cfg.sequence.stride)
    image_size = tuple(cfg.frame_extraction.resize)
    risk_root = Path(cfg.paths.annotations) / "risk_labels" / split
    ego_root = Path(cfg.paths.ego_state) / split
    out_root = Path(cfg.paths.sequences) / split

    sequences_made = 0
    for seq_idx, start in enumerate(range(0, len(frame_files) - seq_len + 1, stride)):
        window_files = frame_files[start : start + seq_len]
        window_fids = frame_ids[start : start + seq_len]

        # Build per-frame graphs
        graphs: List[Data] = []
        for fid in window_fids:
            feats = obj_features.get(fid, [])
            graphs.append(
                build_frame_graph(
                    features=feats,
                    image_size=image_size,
                    num_classes=cfg.features.num_object_classes,
                    k_neighbors=cfg.graph.k_neighbors,
                    edge_dim=cfg.features.edge_dim,
                )
            )

        seq_id = f"{video_name}_seq{seq_idx:04d}"
        out_dir = out_root / seq_id
        (out_dir / "frames").mkdir(parents=True, exist_ok=True)

        # Copy frames into sequence folder
        for i, src in enumerate(window_files, start=1):
            dst = out_dir / "frames" / f"frame_{i:04d}.jpg"
            if not dst.exists():
                shutil.copy2(src, dst)

        # Serialise graphs
        torch.save(graphs, out_dir / "graph.pt")

        # Label + ego state
        materialise_label(
            find_label_file(risk_root, video_name, seq_idx),
            seq_id,
            out_dir / "label.json",
        )
        materialise_ego_state(
            find_ego_state_file(ego_root, video_name, seq_idx),
            list(cfg.ego_encoder.input_features),
            seq_len,
            out_dir / "ego_state.json",
        )

        meta = {
            "sequence_id": seq_id,
            "video": video_name,
            "start_frame": window_fids[0],
            "end_frame": window_fids[-1],
            "fps": int(cfg.frame_extraction.frame_rate),
            "image_size": list(image_size),
            "num_nodes_per_frame": [int(g.num_nodes) for g in graphs],
        }
        with (out_dir / "metadata.json").open("w", encoding="utf-8") as fp:
            json.dump(meta, fp, indent=2)

        sequences_made += 1
    return sequences_made


def run(cfg: Config, split: str) -> None:
    set_seed(cfg.seed)
    frames_root = Path(cfg.paths.extracted_frames) / split
    trk_root = Path(cfg.paths.annotations) / "tracking" / split

    if not trk_root.exists():
        raise FileNotFoundError(f"Tracking root missing: {trk_root}")

    videos = sorted(p for p in trk_root.iterdir() if p.is_dir())
    LOGGER.info("Building graphs/sequences for %d videos (split=%s)", len(videos), split)

    total = 0
    for video_dir in videos:
        video_name = video_dir.name
        tracks_path = video_dir / "tracks.json"
        frames_dir = frames_root / video_name
        if not tracks_path.exists() or not frames_dir.exists():
            LOGGER.warning("Skipping %s (missing tracks or frames)", video_name)
            continue
        n = process_video(video_name, frames_dir, tracks_path, cfg, split)
        total += n
        LOGGER.info("Video %s -> %d sequences", video_name, n)
    LOGGER.info("Built %d total sequences for split=%s", total, split)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Stage 4 — graph + sequence build")
    parser.add_argument("--config", type=str, required=True)
    parser.add_argument("--split", choices=["train", "val", "test"], required=True)
    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()
    run(load_config(args.config), args.split)
