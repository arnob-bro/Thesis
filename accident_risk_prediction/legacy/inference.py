"""Single-video inference entrypoint.

Pipeline: frame extraction → YOLOv8 → ByteTrack → per-frame KNN graphs →
ViT + GAT + Temporal + Ego encoders → Cross-Attention Fusion → Heads.

Usage:
    python inference.py \
        --config configs/config.yaml \
        --checkpoint checkpoints/best.pt \
        --video path/to/clip.mp4 \
        --ego_state path/to/ego.csv \
        --output prediction.json
"""
from __future__ import annotations

import argparse
import csv
import json
import math
import shutil
import tempfile
from collections import defaultdict
from pathlib import Path
from typing import Dict, List, Optional

import cv2
import numpy as np
import torch
from torch_geometric.data import Batch, Data

from features.interaction_features import (
    build_edges_knn,
    normalize_edge_features,
)
from features.object_features import (
    TrackPoint,
    compute_object_features,
    normalize_node_features,
)
from models.full_model import AccidentRiskModel
from models.tracking.byte_tracker import ByteTrackWrapper
from models.yolo.yolo_detector import YoloDetector
from utils.config import Config, load_config
from utils.logger import get_logger
from utils.seed import set_seed

LOGGER = get_logger("inference")


# ---------------------------------------------------------------------------
# Tiny helpers (mirror the offline pipelines, single-video version)
# ---------------------------------------------------------------------------

def _extract_frames(video: Path, out_dir: Path, target_fps: int, resize_wh) -> List[Path]:
    out_dir.mkdir(parents=True, exist_ok=True)
    cap = cv2.VideoCapture(str(video))
    if not cap.isOpened():
        raise RuntimeError(f"Cannot open video {video}")
    src_fps = cap.get(cv2.CAP_PROP_FPS) or float(target_fps)
    step = max(int(round(src_fps / target_fps)), 1)
    saved: List[Path] = []
    fid, idx = 0, 0
    while True:
        ok, frame = cap.read()
        if not ok:
            break
        if idx % step == 0:
            fid += 1
            if resize_wh is not None:
                frame = cv2.resize(frame, resize_wh)
            p = out_dir / f"frame_{fid:06d}.jpg"
            cv2.imwrite(str(p), frame)
            saved.append(p)
        idx += 1
    cap.release()
    return saved


def _detect(detector: YoloDetector, frame_files: List[Path]) -> Dict[int, List[Dict]]:
    grouped: Dict[int, List[Dict]] = defaultdict(list)
    for i, fp in enumerate(frame_files, start=1):
        for det in detector.predict_image(fp, frame_id=i):
            grouped[i].append(det)
    return dict(grouped)


def _track(
    tracker: ByteTrackWrapper, dets_by_frame: Dict[int, List[Dict]],
) -> Dict[int, List[Dict]]:
    tracker.reset()
    out: Dict[int, List[Dict]] = {}
    for fid in sorted(dets_by_frame):
        out[fid] = tracker.update(dets_by_frame[fid], fid)
    return tracker.filter_short_tracks(out)


def _tracks_to_points(
    tracks_by_frame: Dict[int, List[Dict]],
) -> Dict[int, List[TrackPoint]]:
    return {
        fid: [
            TrackPoint(
                frame_id=int(d["frame_id"]),
                track_id=int(d["track_id"]),
                class_id=int(d["class_id"]),
                bbox=[float(v) for v in d["bbox"]],
            )
            for d in dets
        ]
        for fid, dets in tracks_by_frame.items()
    }


def _build_graph(
    features: List, image_size, num_classes: int, k: int, edge_dim: int,
) -> Data:
    if not features:
        x = torch.zeros(1, 8, dtype=torch.float32)
        ei = torch.zeros(2, 0, dtype=torch.long)
        ea = torch.zeros(0, edge_dim, dtype=torch.float32)
        return Data(x=x, edge_index=ei, edge_attr=ea, num_nodes=1)
    x_np = normalize_node_features(features, image_size, num_classes)
    diag = math.hypot(image_size[0], image_size[1])
    edges, attrs = build_edges_knn(features, k, diag)
    ea_np = normalize_edge_features(attrs, diag)
    if edges:
        ei = torch.tensor(edges, dtype=torch.long).t().contiguous()
        ea = torch.from_numpy(ea_np)
    else:
        ei = torch.zeros(2, 0, dtype=torch.long)
        ea = torch.zeros(0, edge_dim, dtype=torch.float32)
    x = torch.from_numpy(x_np)
    return Data(x=x, edge_index=ei, edge_attr=ea, num_nodes=x.shape[0])


def _read_image_as_tensor(path: Path, size: int) -> torch.Tensor:
    img = cv2.imread(str(path))
    if img is None:
        raise FileNotFoundError(path)
    img = cv2.resize(img, (size, size))
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    t = torch.from_numpy(img).float().div(255.0).permute(2, 0, 1)
    mean = torch.tensor([0.485, 0.456, 0.406]).view(3, 1, 1)
    std = torch.tensor([0.229, 0.224, 0.225]).view(3, 1, 1)
    return (t - mean) / std


def _load_ego_csv(
    path: Optional[Path], feature_keys: List[str], n_frames: int,
) -> np.ndarray:
    cols: Dict[str, List[float]] = {k: [] for k in feature_keys}
    if path is None or not path.exists():
        return np.zeros((n_frames, len(feature_keys)), dtype=np.float32)

    if path.suffix == ".csv":
        with path.open("r", encoding="utf-8") as fp:
            reader = csv.DictReader(fp)
            for row in reader:
                for k in feature_keys:
                    try:
                        cols[k].append(float(row.get(k, 0.0)))
                    except ValueError:
                        cols[k].append(0.0)
    elif path.suffix == ".json":
        payload = json.loads(path.read_text(encoding="utf-8"))
        for k in feature_keys:
            cols[k] = [float(v) for v in payload.get(k, [])]
    else:
        raise ValueError(f"Unsupported ego_state extension: {path.suffix}")

    arr_cols = []
    for k in feature_keys:
        series = cols[k]
        if len(series) < n_frames:
            pad = series[-1] if series else 0.0
            series = series + [pad] * (n_frames - len(series))
        arr_cols.append(series[:n_frames])
    return np.asarray(arr_cols, dtype=np.float32).T


# ---------------------------------------------------------------------------
# Main inference
# ---------------------------------------------------------------------------

def run_inference(
    cfg: Config,
    checkpoint: Path,
    video: Path,
    ego_state: Optional[Path],
    output: Path,
) -> None:
    set_seed(int(cfg.seed))
    device = torch.device(cfg.training.device if torch.cuda.is_available() else "cpu")

    # Load model
    model = AccidentRiskModel(cfg).to(device)
    ckpt = torch.load(checkpoint, map_location=device, weights_only=False)
    model.load_state_dict(ckpt["model_state"])
    model.eval()
    LOGGER.info("Loaded checkpoint %s", checkpoint)

    # Stage 1: frame extraction (temp dir)
    workdir = Path(tempfile.mkdtemp(prefix="ar_infer_"))
    frame_dir = workdir / "frames"
    try:
        frame_files = _extract_frames(
            video, frame_dir,
            target_fps=int(cfg.frame_extraction.frame_rate),
            resize_wh=tuple(cfg.frame_extraction.resize),
        )
        if not frame_files:
            raise RuntimeError("No frames extracted from video.")
        LOGGER.info("Extracted %d frames", len(frame_files))

        # Stage 2 + 3: detection + tracking
        detector = YoloDetector(
            weights=cfg.yolo.weights,
            conf_threshold=cfg.yolo.conf_threshold,
            iou_threshold=cfg.yolo.iou_threshold,
            device=cfg.yolo.device if torch.cuda.is_available() else "cpu",
            coco_to_ours=dict(cfg.yolo.coco_to_ours),
        )
        tracker = ByteTrackWrapper(
            track_thresh=cfg.tracking.track_thresh,
            match_thresh=cfg.tracking.match_thresh,
            track_buffer=cfg.tracking.track_buffer,
            frame_rate=cfg.tracking.frame_rate,
            min_track_length=cfg.tracking.min_track_length,
        )
        dets = _detect(detector, frame_files)
        tracks = _track(tracker, dets)
        track_points = _tracks_to_points(tracks)
        LOGGER.info("Tracked %d frames", len(track_points))

        # Stage 4: object features + graphs
        obj_feats = compute_object_features(
            track_points,
            fps=float(cfg.frame_extraction.frame_rate),
            clamp_velocity=cfg.features.clamp_velocity,
            clamp_acceleration=cfg.features.clamp_acceleration,
        )
        image_size = tuple(cfg.frame_extraction.resize)
        graphs_all: List[Data] = []
        for i in range(1, len(frame_files) + 1):
            graphs_all.append(
                _build_graph(
                    obj_feats.get(i, []),
                    image_size,
                    cfg.features.num_object_classes,
                    cfg.graph.k_neighbors,
                    cfg.features.edge_dim,
                )
            )

        # Stage 5: ego state + ViT-ready frames
        feature_keys = list(cfg.ego_encoder.input_features)
        ego_arr = _load_ego_csv(ego_state, feature_keys, len(frame_files))

        vit_size = int(cfg.vit.image_size)
        frames_tensor = torch.stack(
            [_read_image_as_tensor(p, vit_size) for p in frame_files], dim=0,
        )                                                       # [N, 3, H, W]

        # Sequence inference (non-overlapping windows)
        seq_len = int(cfg.sequence.length)
        stride = int(cfg.sequence.stride)
        predictions: List[Dict] = []

        with torch.no_grad():
            for s, start in enumerate(
                range(0, len(frame_files) - seq_len + 1, stride)
            ):
                f_window = frames_tensor[start : start + seq_len].unsqueeze(0).to(device)
                g_window = graphs_all[start : start + seq_len]
                graphs_per_t = [Batch.from_data_list([g]).to(device) for g in g_window]
                ego_window = (
                    torch.from_numpy(ego_arr[start : start + seq_len])
                    .unsqueeze(0).to(device)
                )

                sample = {
                    "frames": f_window,
                    "graphs_per_t": graphs_per_t,
                    "ego_state": ego_window,
                }
                out = model(sample)
                probs = torch.softmax(out["logits"], dim=-1).cpu().numpy()[0]
                cls = int(probs.argmax())
                ttc = float(out["ttc"].cpu().item())

                predictions.append(
                    {
                        "sequence_index": s,
                        "start_frame": start + 1,
                        "end_frame": start + seq_len,
                        "risk_level": cls,
                        "risk_probabilities": [float(p) for p in probs],
                        "ttc_estimate": ttc,
                    }
                )

        output.parent.mkdir(parents=True, exist_ok=True)
        with output.open("w", encoding="utf-8") as fp:
            json.dump(
                {
                    "video": str(video),
                    "num_frames": len(frame_files),
                    "num_sequences": len(predictions),
                    "predictions": predictions,
                },
                fp, indent=2,
            )
        LOGGER.info("Wrote %d predictions -> %s", len(predictions), output)

    finally:
        shutil.rmtree(workdir, ignore_errors=True)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Single-video accident risk inference")
    parser.add_argument("--config", type=str, required=True)
    parser.add_argument("--checkpoint", type=str, required=True)
    parser.add_argument("--video", type=str, required=True)
    parser.add_argument(
        "--ego_state", type=str, default=None,
        help="Optional CSV/JSON ego telemetry; zeros are used if omitted",
    )
    parser.add_argument("--output", type=str, default="prediction.json")
    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()
    cfg = load_config(args.config)
    run_inference(
        cfg=cfg,
        checkpoint=Path(args.checkpoint),
        video=Path(args.video),
        ego_state=Path(args.ego_state) if args.ego_state else None,
        output=Path(args.output),
    )
