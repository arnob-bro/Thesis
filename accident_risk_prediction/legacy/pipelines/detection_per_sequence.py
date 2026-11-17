"""Per-sequence YOLO detection.

Processes `data/sequences/<split>/*/frames/*.jpg` and writes
`data/sequences/<split>/<sequence>/detections.json` with preserved frame_ids.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Dict, List

from utils.config import Config, load_config
from utils.logger import get_logger
from utils.seed import set_seed
from models.yolo.yolo_detector import YoloDetector

LOGGER = get_logger("detection_per_sequence")


def frame_id_from_path(frame_path: Path) -> int:
    stem = frame_path.stem
    if stem.isdigit():
        return int(stem)
    parts = stem.split("_")
    if parts and parts[-1].isdigit():
        return int(parts[-1])
    raise ValueError(f"Unable to parse frame_id from file name: {frame_path.name}")


def run(cfg: Config, split: str) -> None:
    set_seed(cfg.seed)
    seq_root = Path(cfg.paths.sequences) / split
    if not seq_root.exists():
        raise FileNotFoundError(f"Sequences root missing: {seq_root}. Run sequence_generator first.")

    detector = YoloDetector(
        weights=cfg.yolo.weights,
        conf_threshold=cfg.yolo.conf_threshold,
        iou_threshold=cfg.yolo.iou_threshold,
        device=cfg.yolo.device,
        coco_to_ours=dict(cfg.yolo.coco_to_ours),
    )

    sequences = sorted(p for p in seq_root.iterdir() if p.is_dir())
    LOGGER.info("Running detection on %d sequences (split=%s)", len(sequences), split)

    for seq_dir in sequences:
        frames_dir = seq_dir / "frames"
        if not frames_dir.exists():
            LOGGER.warning("No frames in %s, skipping", seq_dir)
            continue

        files = sorted(frames_dir.glob("*.jpg"), key=lambda p: frame_id_from_path(p))
        dets: List[Dict] = []
        for fp in files:
            fid = frame_id_from_path(fp)
            for d in detector.predict_image(fp, frame_id=fid):
                dets.append(d)

        out_path = seq_dir / "detections.json"
        with out_path.open("w", encoding="utf-8") as fp:
            json.dump(dets, fp)

        LOGGER.info("Wrote %d detections for %s", len(dets), seq_dir.name)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Per-sequence YOLO detection")
    parser.add_argument("--config", type=str, required=True)
    parser.add_argument("--split", choices=["train", "val", "test"], required=True)
    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()
    run(load_config(args.config), args.split)
