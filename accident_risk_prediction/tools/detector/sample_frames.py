"""Step 1 - pick frames from your videos to label for the detector.

    python -m tools.detector.sample_frames --videos data/raw_videos

Per video it saves `sampling.per_video` evenly spread frames, plus
`sampling.per_event` frames from the seconds before each marked collision
(those are the moments the detector most needs to get right). Near-identical
frames are skipped. Output:

    data/detector/own/images/<video_id>/<video_id>_<frame>.jpg

Re-running never overwrites or re-samples a video that already has frames.
"""
from __future__ import annotations

import argparse
from pathlib import Path
from typing import List

import cv2
import numpy as np

from utils.config import load_config
from utils.events import find_videos, load_events, make_video_id


def pick_indices(n_frames: int, fps: float, per_video: int, events: List[float],
                 per_event: int, window_s: float) -> List[int]:
    idx = set(np.linspace(0, max(n_frames - 1, 0), per_video + 2, dtype=int)[1:-1].tolist())
    for t in events:
        for s in np.linspace(max(t - window_s, 0.0), t, per_event):
            idx.add(min(int(round(s * fps)), n_frames - 1))
    return sorted(i for i in idx if i >= 0)


def main() -> None:
    p = argparse.ArgumentParser(description="Sample frames to label for the detector")
    p.add_argument("--config", default="configs/detector.yaml")
    p.add_argument("--main-config", default="configs/config.yaml", help="Used to find marked collision times")
    p.add_argument("--videos", required=True)
    args = p.parse_args()

    cfg = load_config(args.config)
    s = cfg.sampling
    out_root = Path(cfg.paths.own_images)
    events = load_events(load_config(args.main_config).event_labeling.events_file)["videos"]

    total = 0
    for video in find_videos(args.videos):
        vid = make_video_id(video.name)
        out = out_root / vid
        if out.exists() and any(out.glob("*.jpg")):
            continue
        cap = cv2.VideoCapture(str(video))
        if not cap.isOpened():
            print(f"skip {vid}: cannot open")
            continue
        fps = float(cap.get(cv2.CAP_PROP_FPS)) or 30.0
        n = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        times = [e["time_s"] for e in events.get(video.name, {}).get("events", [])]
        wanted = pick_indices(n, fps, int(s.per_video), times, int(s.per_event), float(s.event_window_s))

        out.mkdir(parents=True, exist_ok=True)
        prev_small, saved = None, 0
        for i in wanted:
            cap.set(cv2.CAP_PROP_POS_FRAMES, i)
            ok, frame = cap.read()
            if not ok:
                continue
            small = cv2.cvtColor(cv2.resize(frame, (64, 36)), cv2.COLOR_BGR2GRAY).astype(np.float32)
            if prev_small is not None and float(np.abs(small - prev_small).mean()) < float(s.min_difference):
                continue
            prev_small = small
            h, w = frame.shape[:2]
            if w > int(s.max_width):
                frame = cv2.resize(frame, (int(s.max_width), int(round(h * int(s.max_width) / w))))
            ok, buf = cv2.imencode(".jpg", frame, [cv2.IMWRITE_JPEG_QUALITY, 92])
            (out / f"{vid}_{i:07d}.jpg").write_bytes(buf.tobytes())
            saved += 1
        cap.release()
        total += saved
        print(f"{vid}: {saved} frames")
    print(f"\n{total} new frames -> {out_root}")


if __name__ == "__main__":
    main()
