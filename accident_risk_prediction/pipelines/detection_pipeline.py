"""Stage 2: YOLOv8 detection on extracted frames; cached to disk.

Output (per video):
    data/annotations/detections/<split>/<video>/detections.json
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

LOGGER = get_logger("detection_pipeline")


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


def detect_video_frames(
    detector: YoloDetector, frames_dir: Path,
) -> List[Dict]:
    files = sorted(frames_dir.glob("*.jpg"))
    all_dets: List[Dict] = []
    for fp in files:
        frame_id = frame_id_from_path(fp)
        dets = detector.predict_image(fp, frame_id=frame_id)
        all_dets.extend(dets)
    return all_dets


def run(cfg: Config, split: str) -> None:
    set_seed(cfg.seed)
    frames_root = Path(cfg.paths.extracted_frames) / split
    out_root = Path(cfg.paths.annotations) / "detections" / split
    out_root.mkdir(parents=True, exist_ok=True)

    if not frames_root.exists():
        raise FileNotFoundError(f"Frames root missing: {frames_root}")

    detector = YoloDetector(
        weights=cfg.yolo.weights,
        conf_threshold=cfg.yolo.conf_threshold,
        iou_threshold=cfg.yolo.iou_threshold,
        device=cfg.yolo.device,
        coco_to_ours=dict(cfg.yolo.coco_to_ours),
    )

    videos = sorted(p for p in frames_root.iterdir() if p.is_dir())
    LOGGER.info("Detecting on %d videos (split=%s)", len(videos), split)

    for video_dir in videos:
        dets = detect_video_frames(detector, video_dir)
        out_dir = out_root / video_dir.name
        out_dir.mkdir(parents=True, exist_ok=True)
        with (out_dir / "detections.json").open("w", encoding="utf-8") as fp:
            json.dump(dets, fp)
        LOGGER.info("Video %s -> %d detections", video_dir.name, len(dets))


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Stage 2 — YOLOv8 detection")
    parser.add_argument("--config", type=str, required=True)
    parser.add_argument("--split", choices=["train", "val", "test"], required=True)
    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()
    run(load_config(args.config), args.split)
