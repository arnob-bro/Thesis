"""Stage 3: ByteTrack tracking on cached detections.

Output (per video):
    data/annotations/tracking/<split>/<video>/tracks.json

Tracks shorter than `min_track_length` are removed (MD constraint).
"""
from __future__ import annotations

import argparse
import json
from collections import defaultdict
from pathlib import Path
from typing import Dict, List

from utils.config import Config, load_config
from utils.logger import get_logger
from utils.seed import set_seed
from models.tracking.byte_tracker import ByteTrackWrapper

LOGGER = get_logger("tracking_pipeline")


def load_detections(path: Path) -> Dict[int, List[Dict]]:
    with path.open("r", encoding="utf-8") as fp:
        raw = json.load(fp)
    grouped: Dict[int, List[Dict]] = defaultdict(list)
    for det in raw:
        grouped[int(det["frame_id"])].append(det)
    return dict(grouped)


def track_video(
    tracker: ByteTrackWrapper, detections_by_frame: Dict[int, List[Dict]],
) -> Dict[int, List[Dict]]:
    tracker.reset()
    out: Dict[int, List[Dict]] = {}
    for fid in sorted(detections_by_frame.keys()):
        tracked = tracker.update(detections_by_frame[fid], fid)
        out[fid] = tracked
    return tracker.filter_short_tracks(out)


def run(cfg: Config, split: str) -> None:
    set_seed(cfg.seed)
    det_root = Path(cfg.paths.annotations) / "detections" / split
    trk_root = Path(cfg.paths.annotations) / "tracking" / split
    trk_root.mkdir(parents=True, exist_ok=True)

    if not det_root.exists():
        raise FileNotFoundError(
            f"Detections root missing: {det_root}. Run detection_pipeline first."
        )

    tracker = ByteTrackWrapper(
        track_thresh=cfg.tracking.track_thresh,
        match_thresh=cfg.tracking.match_thresh,
        track_buffer=cfg.tracking.track_buffer,
        frame_rate=cfg.tracking.frame_rate,
        min_track_length=cfg.tracking.min_track_length,
    )

    videos = sorted(p for p in det_root.iterdir() if p.is_dir())
    LOGGER.info("Tracking on %d videos (split=%s)", len(videos), split)

    for video_dir in videos:
        det_file = video_dir / "detections.json"
        if not det_file.exists():
            LOGGER.warning("Missing %s, skipping", det_file)
            continue
        grouped = load_detections(det_file)
        tracked = track_video(tracker, grouped)

        flat: List[Dict] = []
        for fid in sorted(tracked):
            flat.extend(tracked[fid])

        out_dir = trk_root / video_dir.name
        out_dir.mkdir(parents=True, exist_ok=True)
        with (out_dir / "tracks.json").open("w", encoding="utf-8") as fp:
            json.dump(flat, fp)
        LOGGER.info("Video %s -> %d tracked detections", video_dir.name, len(flat))


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Stage 3 — ByteTrack tracking")
    parser.add_argument("--config", type=str, required=True)
    parser.add_argument("--split", choices=["train", "val", "test"], required=True)
    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()
    run(load_config(args.config), args.split)
