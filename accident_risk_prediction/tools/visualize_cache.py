"""Visual check of a processed video: what is tracked and what is removed.

    python -m tools.visualize_cache --config configs/config.yaml --video-id 12

Writes data/qa/<video_id>.mp4 with, on every sampled frame:
    green   tracked objects (class + track id) that go into the graph
    red     detections removed as the rider's own bike / body
    yellow  the rider region (`ego_box` in the config)

Use it to tune `ego_filter` and `ego_box`: red boxes should only ever cover
your own handlebars, tank or arms, never another road user.
"""
from __future__ import annotations

import argparse
import json
from collections import defaultdict
from pathlib import Path

import cv2
import numpy as np

from features.graph_builder import is_ego_detection
from utils.config import load_config


def main() -> None:
    p = argparse.ArgumentParser(description="Render tracked / removed boxes for one cached video")
    p.add_argument("--config", required=True)
    p.add_argument("--video-id", required=True)
    p.add_argument("--out", help="Output .mp4 (default data/qa/<video_id>.mp4)")
    args = p.parse_args()

    cfg = load_config(args.config)
    cache = Path(cfg.paths.video_cache) / args.video_id
    meta = json.loads((cache / "meta.json").read_text(encoding="utf-8"))
    W, H = meta["image_size"]
    classes = list(cfg.yolo.classes)

    payload = json.loads((cache / "detections.json").read_text(encoding="utf-8"))
    mult = int(payload["fps_multiplier"])
    removed = defaultdict(list)
    for d in payload["detections"]:
        if d["frame_id"] % mult == 0 and is_ego_detection(d, cfg, (W, H)):
            removed[d["frame_id"] // mult].append(d)
    tracks = defaultdict(list)
    for d in json.loads((cache / "tracks.json").read_text(encoding="utf-8")):
        tracks[d["frame_id"]].append(d)

    out = Path(args.out or f"data/qa/{args.video_id}.mp4")
    out.parent.mkdir(parents=True, exist_ok=True)
    writer = cv2.VideoWriter(str(out), cv2.VideoWriter_fourcc(*"mp4v"), float(meta["fps"]), (W, H))
    cx, cy, bw, bh = cfg.ego_box
    ego = (int((cx - bw / 2) * W), int((cy - bh / 2) * H), int((cx + bw / 2) * W), int((cy + bh / 2) * H))

    n_removed = n_tracked = 0
    for k, path in enumerate(sorted((cache / "frames").glob("*.jpg"))):
        img = cv2.imdecode(np.frombuffer(path.read_bytes(), np.uint8), cv2.IMREAD_COLOR)
        cv2.rectangle(img, ego[:2], ego[2:], (0, 220, 220), 1)
        for d in removed[k]:
            x, y, w, h = (int(v) for v in d["bbox"])
            cv2.rectangle(img, (x, y), (x + w, y + h), (0, 0, 255), 2)
            cv2.putText(img, "own bike", (x + 3, y + 14), cv2.FONT_HERSHEY_SIMPLEX, 0.45, (0, 0, 255), 1, cv2.LINE_AA)
        for d in tracks[k]:
            x, y, w, h = (int(v) for v in d["bbox"])
            cv2.rectangle(img, (x, y), (x + w, y + h), (0, 200, 0), 2)
            cv2.putText(img, f"{classes[d['class_id']]} #{d['track_id']}", (x + 3, max(y - 4, 12)),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.45, (0, 200, 0), 1, cv2.LINE_AA)
        cv2.putText(img, f"{k / meta['fps']:.1f}s", (6, 16), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1, cv2.LINE_AA)
        writer.write(img)
        n_removed += len(removed[k])
        n_tracked += len(tracks[k])
    writer.release()
    print(f"{out}: {n_tracked} tracked boxes (green), {n_removed} removed as own bike (red)")


if __name__ == "__main__":
    main()
