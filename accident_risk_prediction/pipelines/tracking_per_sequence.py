"""Per-sequence ByteTrack tracking.

Processes `data/sequences/<split>/*/detections.json` and writes
`data/sequences/<split>/<sequence>/tracks.json` with preserved frame/track ids.
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

LOGGER = get_logger("tracking_per_sequence")


def load_detections(path: Path) -> Dict[int, List[Dict]]:
    with path.open("r", encoding="utf-8") as fp:
        raw = json.load(fp)
    grouped: Dict[int, List[Dict]] = defaultdict(list)
    for det in raw:
        grouped[int(det["frame_id"])].append(det)
    return dict(grouped)


def run(cfg: Config, split: str) -> None:
    set_seed(cfg.seed)
    seq_root = Path(cfg.paths.sequences) / split
    if not seq_root.exists():
        raise FileNotFoundError(f"Sequences root missing: {seq_root}. Run sequence_generator first.")

    tracker = ByteTrackWrapper(
        track_thresh=cfg.tracking.track_thresh,
        match_thresh=cfg.tracking.match_thresh,
        track_buffer=cfg.tracking.track_buffer,
        frame_rate=cfg.tracking.frame_rate,
        min_track_length=cfg.tracking.min_track_length,
    )

    sequences = sorted(p for p in seq_root.iterdir() if p.is_dir())
    LOGGER.info("Running tracking on %d sequences (split=%s)", len(sequences), split)

    for seq_dir in sequences:
        det_file = seq_dir / "detections.json"
        if not det_file.exists():
            LOGGER.warning("Missing %s, skipping", det_file)
            continue

        grouped = load_detections(det_file)

        # Run tracker over sorted frames
        tracker.reset()
        out: Dict[int, List[Dict]] = {}
        for fid in sorted(grouped.keys()):
            tracked = tracker.update(grouped[fid], fid)
            out[fid] = tracked

        filtered = tracker.filter_short_tracks(out)

        flat: List[Dict] = []
        for fid in sorted(filtered):
            flat.extend(filtered[fid])

        out_path = seq_dir / "tracks.json"
        with out_path.open("w", encoding="utf-8") as fp:
            json.dump(flat, fp)

        LOGGER.info("Wrote %d tracked detections for %s", len(flat), seq_dir.name)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Per-sequence ByteTrack tracking")
    parser.add_argument("--config", type=str, required=True)
    parser.add_argument("--split", choices=["train", "val", "test"], required=True)
    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()
    run(load_config(args.config), args.split)
