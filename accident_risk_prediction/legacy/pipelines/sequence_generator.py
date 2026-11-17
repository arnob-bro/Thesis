"""Stage 1.5: Split extracted frames into fixed-length sequence folders.

This stage produces the per-sequence directory layout used by the restructured
pipeline:

    data/sequences/<split>/<video>_seq<NNNN>/
        frames/<original frame files...>
        metadata.json

The original frame filenames are preserved so downstream stages can recover
the true frame IDs. A `metadata.json` file stores the sequence -> frame_id map
and auxiliary info.
"""
from __future__ import annotations

import argparse
import json
import shutil
from pathlib import Path
from typing import List

from utils.config import Config, load_config
from utils.logger import get_logger
from utils.seed import set_seed

LOGGER = get_logger("sequence_generator")


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

    frames_root = Path(cfg.paths.extracted_frames) / split
    out_root = Path(cfg.paths.sequences) / split
    out_root.mkdir(parents=True, exist_ok=True)

    if not frames_root.exists():
        raise FileNotFoundError(f"Frames root missing: {frames_root}")

    seq_len = int(cfg.sequence.length)
    stride = int(cfg.sequence.stride)

    videos = sorted(p for p in frames_root.iterdir() if p.is_dir())
    LOGGER.info("Generating sequences for %d videos (split=%s)", len(videos), split)

    total = 0
    for video_dir in videos:
        video_name = video_dir.name
        frame_files = sorted(list(video_dir.glob("*.jpg")), key=frame_id_from_path)
        if not frame_files:
            LOGGER.warning("No frames for %s, skipping", video_name)
            continue

        for seq_idx, start in enumerate(range(0, len(frame_files) - seq_len + 1, stride)):
            window = frame_files[start : start + seq_len]
            seq_id = f"{video_name}_seq{seq_idx:04d}"
            out_dir = out_root / seq_id
            frames_out = out_dir / "frames"
            frames_out.mkdir(parents=True, exist_ok=True)

            # Copy (preserve original names) so downstream stages can recover IDs
            frame_ids: List[int] = []
            frame_names: List[str] = []
            for fp in window:
                shutil.copy2(fp, frames_out / fp.name)
                frame_ids.append(frame_id_from_path(fp))
                frame_names.append(fp.name)

            meta = {
                "sequence_id": seq_id,
                "video": video_name,
                "start_frame": frame_ids[0],
                "end_frame": frame_ids[-1],
                "frame_ids": frame_ids,
                "frame_names": frame_names,
                "fps": int(cfg.frame_extraction.frame_rate),
                "image_size": list(cfg.frame_extraction.resize),
            }
            with (out_dir / "metadata.json").open("w", encoding="utf-8") as fp:
                json.dump(meta, fp, indent=2)

            total += 1

    LOGGER.info("Created %d sequences for split=%s", total, split)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Split extracted frames into sequences")
    parser.add_argument("--config", type=str, required=True)
    parser.add_argument("--split", choices=["train", "val", "test"], required=True)
    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()
    run(load_config(args.config), args.split)
