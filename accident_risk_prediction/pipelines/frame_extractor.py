"""Stage 1: extract frames from raw videos at the configured FPS.

Naming convention (MD dataset_documentation.md):
    frame_<videoid>_<frame_number>.jpg

Output directory:
    data/extracted_frames/<split>/<video_stem>/frame_000001.jpg
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import List

import cv2

from utils.config import Config, load_config
from utils.logger import get_logger
from utils.seed import set_seed

LOGGER = get_logger("frame_extractor")


def list_videos(raw_root: Path, split: str) -> List[Path]:
    folder = raw_root / split
    if not folder.exists():
        raise FileNotFoundError(f"Raw video folder missing: {folder}")
    exts = {".mp4", ".avi", ".mov", ".mkv"}
    return sorted(p for p in folder.iterdir() if p.suffix.lower() in exts)


def extract_from_video(
    video_path: Path,
    out_dir: Path,
    target_fps: int,
    resize_wh: tuple[int, int] | None,
) -> int:
    out_dir.mkdir(parents=True, exist_ok=True)

    cap = cv2.VideoCapture(str(video_path))
    if not cap.isOpened():
        raise RuntimeError(f"Cannot open video {video_path}")

    src_fps = cap.get(cv2.CAP_PROP_FPS) or float(target_fps)
    step = max(int(round(src_fps / target_fps)), 1)

    saved = 0
    frame_idx = 0
    while True:
        ok, frame = cap.read()
        if not ok:
            break
        if frame_idx % step == 0:
            if resize_wh is not None:
                frame = cv2.resize(frame, resize_wh)
            saved += 1
            out_path = out_dir / f"frame_{video_path.stem}_{saved:06d}.jpg"
            cv2.imwrite(str(out_path), frame)
        frame_idx += 1
    cap.release()

    meta = {
        "video": video_path.name,
        "source_fps": src_fps,
        "target_fps": target_fps,
        "extracted": saved,
        "resize": list(resize_wh) if resize_wh else None,
    }
    with (out_dir / "extraction_meta.json").open("w", encoding="utf-8") as fp:
        json.dump(meta, fp, indent=2)
    return saved


def run(cfg: Config, split: str) -> None:
    set_seed(cfg.seed)
    raw_root = Path(cfg.paths.raw_videos)
    frames_root = Path(cfg.paths.extracted_frames) / split
    fps = int(cfg.frame_extraction.frame_rate)
    resize = tuple(cfg.frame_extraction.resize)         # (W, H)

    videos = list_videos(raw_root, split)
    LOGGER.info("Found %d videos in split=%s", len(videos), split)

    for video in videos:
        out_dir = frames_root / video.stem
        n = extract_from_video(video, out_dir, fps, resize)
        LOGGER.info("Extracted %d frames from %s -> %s", n, video.name, out_dir)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Stage 1 — frame extraction")
    parser.add_argument("--config", type=str, required=True)
    parser.add_argument("--split", choices=["train", "val", "test"], required=True)
    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()
    run(load_config(args.config), args.split)
