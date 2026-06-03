from __future__ import annotations

import argparse
import json
from collections import defaultdict
from pathlib import Path
from typing import List

import cv2
import torch
from torch_geometric.data import Data

from utils.visualization import draw_graph_edges
from utils.config import load_config


def load_graphs(path: Path) -> List[Data]:
    with path.open("rb") as fp:
        with torch.serialization.safe_globals([Data]):
            return torch.load(fp, weights_only=False)


def load_metadata(path: Path) -> dict:
    with path.open("r", encoding="utf-8") as fp:
        return json.load(fp)


def frame_id_from_path(frame_path: Path) -> int:
    stem = frame_path.stem
    if stem.isdigit():
        return int(stem)
    parts = stem.split("_")
    if parts and parts[-1].isdigit():
        return int(parts[-1])
    raise ValueError(f"Unable to parse frame_id from file name: {frame_path.name}")


def choose_frame_path(candidates: List[Path], video_name: str | None = None) -> Path:
    if video_name:
        for path in candidates:
            if f"_{video_name}_" in path.stem or path.stem.startswith(f"{video_name}_"):
                return path
    return sorted(candidates, key=lambda p: p.name)[0]


def resolve_frame_paths(frames_dir: Path, metadata: dict) -> List[Path]:
    if "frame_names" in metadata:
        frame_paths = [frames_dir / name for name in metadata["frame_names"]]
        for path in frame_paths:
            if not path.exists():
                raise FileNotFoundError(f"Frame not found: {path}")
        return frame_paths

    frame_files = list(frames_dir.glob("*.jpg"))
    frame_files_by_id: dict[int, List[Path]] = defaultdict(list)
    for fp in frame_files:
        try:
            fid = frame_id_from_path(fp)
        except ValueError:
            continue
        frame_files_by_id[fid].append(fp)

    frame_paths: List[Path] = []
    video_name = metadata.get("video")
    for fid in metadata.get("frame_ids", []):
        candidates = frame_files_by_id.get(fid)
        if not candidates:
            raise FileNotFoundError(
                f"Unable to resolve frame for frame_id {fid} in {frames_dir}"
            )
        chosen = choose_frame_path(candidates, video_name)
        frame_paths.append(chosen)
        candidates.remove(chosen)

    return frame_paths


def visualize_sequence(split: str, sequence: str | None = None, config: str | None = None) -> None:
    if sequence is None:
        sequences_root = Path(f"data/sequences/{split}")
        if not sequences_root.exists():
            raise FileNotFoundError(f"Split directory missing: {sequences_root}")

        sequence_dirs = sorted([p for p in sequences_root.iterdir() if p.is_dir()])
        if not sequence_dirs:
            raise FileNotFoundError(f"No sequence directories found under: {sequences_root}")

        for seq_dir in sequence_dirs:
            print(f"Visualizing sequence: {seq_dir.name}")
            visualize_sequence(split=split, sequence=seq_dir.name, config=config)
        return

    seq_dir = Path(f"data/sequences/{split}/{sequence}")
    frames_dir = seq_dir / "frames"
    graph_path = seq_dir / "graph.pt"
    meta_path = seq_dir / "metadata.json"

    if not seq_dir.exists():
        raise FileNotFoundError(f"Sequence directory missing: {seq_dir}")
    if not graph_path.exists():
        raise FileNotFoundError(f"Graph file missing: {graph_path}")
    if not meta_path.exists():
        raise FileNotFoundError(f"Metadata file missing: {meta_path}")

    graphs = load_graphs(graph_path)
    metadata = load_metadata(meta_path)
    cfg = load_config(config) if config else None
    frame_files = resolve_frame_paths(frames_dir, metadata)

    output_dir = seq_dir / "qa_graphs"
    output_dir.mkdir(parents=True, exist_ok=True)

    frame_ids = metadata.get("frame_ids", [])
    if len(frame_files) != len(graphs):
        print(
            f"Warning: sequence has {len(frame_files)} resolved frames but {len(graphs)} graphs."
            " Using min length for visualization."
        )

    output_len = min(len(graphs), len(frame_files), len(frame_ids))
    for idx in range(output_len):
        graph = graphs[idx]
        frame_fp = frame_files[idx]
        fid = frame_ids[idx] if idx < len(frame_ids) else idx
        image = cv2.imread(str(frame_fp))
        if image is None:
            print(f"Unable to read frame image: {frame_fp}")
            continue

        if graph.edge_index.numel() == 0:
            edges = []
        else:
            edges = list(zip(graph.edge_index[0].tolist(), graph.edge_index[1].tolist()))

        nodes_xy = []
        for node in graph.x.tolist():
            cx = float(node[0]) * metadata["image_size"][0]
            cy = float(node[1]) * metadata["image_size"][1]
            nodes_xy.append((cx, cy))

        num_classes = int(cfg.features.num_object_classes) if cfg else None
        clamp_v = float(cfg.features.clamp_velocity) if cfg else 50.0
        clamp_a = float(cfg.features.clamp_acceleration) if cfg else 20.0

        xs = graph.x.cpu().numpy() if hasattr(graph, "x") else None
        track_ids = None
        if hasattr(graph, "track_id"):
            try:
                track_ids = graph.track_id.cpu().tolist()
            except Exception:
                track_ids = None

        node_labels: List[str] = []
        for i, _ in enumerate(nodes_xy):
            tid = track_ids[i] if track_ids is not None and i < len(track_ids) else i
            if xs is not None and i < xs.shape[0]:
                nx = xs[i]
                den_vx = nx[4] * clamp_v
                den_vy = nx[5] * clamp_v
                den_speed = (den_vx ** 2 + den_vy ** 2) ** 0.5
                den_a = nx[6] * clamp_a
                den_cls = int(round(nx[7] * max((num_classes or 1) - 1, 1))) if num_classes else int(round(nx[7]))
                node_labels.append(
                    f"ID:{tid} c:{den_cls} v:{den_speed:.1f} a:{den_a:.1f}"
                )
            else:
                node_labels.append(f"ID:{tid}")

        edge_attrs = None
        if hasattr(graph, "edge_attr") and graph.edge_attr is not None:
            try:
                edge_attrs = [list(e) for e in graph.edge_attr.cpu().tolist()]
            except Exception:
                edge_attrs = None

        annotated = draw_graph_edges(
            image=image,
            nodes_xy=nodes_xy,
            edges=edges,
            edge_attr=edge_attrs,
            node_labels=node_labels,
        )

        save_path = output_dir / f"frame_{fid:06d}.jpg"
        cv2.imwrite(str(save_path), annotated)

    print(f"Saved graph QA frames to: {output_dir}")
    print(f"Visualized {len(graphs)} frames.")


def main() -> None:
    parser = argparse.ArgumentParser(description="Visualize graphs over sequence frames")
    parser.add_argument("--split", required=True, choices=["train", "val", "test"])
    parser.add_argument("--sequence", required=False, help="Optional sequence folder name. If omitted, all sequences in the split are visualized.")
    parser.add_argument("--config", required=False, help="Optional config YAML for denorm values")
    args = parser.parse_args()
    visualize_sequence(split=args.split, sequence=args.sequence, config=args.config)


if __name__ == "__main__":
    main()
