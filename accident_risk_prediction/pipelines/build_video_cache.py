"""Process every video ONCE and cache everything the model needs.

    data/video_cache/<video_id>/
        frames/000000.jpg ...    frames sampled at frame_extraction.frame_rate
        meta.json
        detections.json          raw detector output
        tracks.json              ByteTrack over the whole video (rider's own bike removed)
        ego_motion.npz           camera-motion features + affine transforms
        graphs.npz               rider-centric interaction graph per frame
        vit.npy / vit_flip.npy   frozen ViT embedding per frame (and mirrored)

Sequences are then just (video, start, end) rows in the manifest, so labels
and bins can change without touching this cache.

Usage
    python -m pipelines.build_video_cache --config configs/config.yaml --videos data/raw_videos
    # only rebuild graphs after changing feature settings:
    python -m pipelines.build_video_cache --config configs/config.yaml --steps graphs --force

By default it processes the videos marked "done" in the collision events
file. Use --all to process every video found under --videos.
"""
from __future__ import annotations

import argparse
import json
from collections import defaultdict
from pathlib import Path
from typing import Dict, List, Optional, Sequence

import cv2
import numpy as np
import torch

from features.ego_motion import estimate_video
from features.graph_builder import build_video_graphs, filter_ego_detections
from utils.config import Config, load_config
from utils.events import find_videos, load_events, make_video_id
from utils.logger import get_logger

LOGGER = get_logger("build_video_cache")
STEPS = ("frames", "detect", "track", "ego", "graphs", "vit")


class Models:
    """Lazy holders so a step that is skipped never loads its model."""

    def __init__(self, cfg: Config) -> None:
        self.cfg = cfg
        self._yolo = None
        self._vit = None
        self.device = torch.device(cfg.yolo.device if torch.cuda.is_available() else "cpu")

    @property
    def yolo(self):
        if self._yolo is None:
            from models.yolo.yolo_detector import YoloDetector
            y = self.cfg.yolo
            self._yolo = YoloDetector(y.weights, y.conf_threshold, y.iou_threshold,
                                      str(self.device), dict(y.class_map),
                                      imgsz=y.get("imgsz"), agnostic_nms=bool(y.get("agnostic_nms", False)))
        return self._yolo

    @property
    def vit(self):
        if self._vit is None:
            from models.vit.vit_model import FrozenViT
            self._vit = FrozenViT(self.cfg.vit.model_name, self.cfg.vit.output_dim).to(self.device).eval()
        return self._vit


def frame_paths(cache: Path) -> List[Path]:
    return sorted((cache / "frames").glob("*.jpg"))


def _read(path: Path) -> np.ndarray:
    return cv2.imdecode(np.frombuffer(path.read_bytes(), np.uint8), cv2.IMREAD_COLOR)


# ---------------------------------------------------------------------------
# Steps
# ---------------------------------------------------------------------------

def step_frames(video: Path, cache: Path, cfg: Config, video_id: str) -> None:
    fps = float(cfg.frame_extraction.frame_rate)
    size = tuple(int(v) for v in cfg.frame_extraction.resize)
    cap = cv2.VideoCapture(str(video))
    if not cap.isOpened():
        raise IOError(f"Cannot open video: {video}")
    src_fps = float(cap.get(cv2.CAP_PROP_FPS)) or 30.0
    (cache / "frames").mkdir(parents=True, exist_ok=True)
    k, idx = 0, 0
    while True:
        if not cap.grab():
            break
        if idx == int(round(k / fps * src_fps)):
            ok, frame = cap.retrieve()
            if ok:
                ok, buf = cv2.imencode(".jpg", cv2.resize(frame, size), [cv2.IMWRITE_JPEG_QUALITY, 92])
                (cache / "frames" / f"{k:06d}.jpg").write_bytes(buf.tobytes())
            k += 1
        idx += 1
    cap.release()
    meta = {"video_id": video_id, "source_file": video.name, "fps": fps, "src_fps": src_fps,
            "n_frames": k, "image_size": list(size)}
    (cache / "meta.json").write_text(json.dumps(meta, indent=2, ensure_ascii=False), encoding="utf-8")


def step_detect(video: Path, cache: Path, cfg: Config, models: Models) -> None:
    """Detect at `tracking.fps_multiplier` x the sampling rate, straight from the video.

    At 5 fps objects move too far between frames for the tracker to match
    them, so detection and tracking run denser (default 15 fps) and only the
    sampled frames are kept afterwards.
    """
    mult = int(cfg.tracking.fps_multiplier)
    fps = float(cfg.frame_extraction.frame_rate) * mult
    size = tuple(int(v) for v in cfg.frame_extraction.resize)
    n_dense = len(frame_paths(cache)) * mult
    cap = cv2.VideoCapture(str(video))
    if not cap.isOpened():
        raise IOError(f"Cannot open video: {video}")
    src_fps = float(cap.get(cv2.CAP_PROP_FPS)) or 30.0
    dets: List[Dict] = []
    j, idx = 0, 0
    while j < n_dense:
        if not cap.grab():
            break
        if idx == int(round(j / fps * src_fps)):
            ok, frame = cap.retrieve()
            if ok:
                dets.extend(models.yolo.predict_array(cv2.resize(frame, size), j))
            j += 1
        idx += 1
    cap.release()
    (cache / "detections.json").write_text(json.dumps({"fps_multiplier": mult, "detections": dets}), encoding="utf-8")


def step_track(cache: Path, cfg: Config) -> None:
    from models.tracking.byte_tracker import ByteTrackWrapper
    size = json.loads((cache / "meta.json").read_text(encoding="utf-8"))["image_size"]
    payload = json.loads((cache / "detections.json").read_text(encoding="utf-8"))
    mult, raw = int(payload["fps_multiplier"]), payload["detections"]
    n_dense = len(frame_paths(cache)) * mult
    by_frame: Dict[int, List[Dict]] = defaultdict(list)
    for d in raw:
        by_frame[int(d["frame_id"])].append(d)
    t = cfg.tracking
    tracker = ByteTrackWrapper(t.track_thresh, t.match_thresh, int(t.track_buffer) * mult,
                               int(t.frame_rate) * mult, t.min_track_length)
    tracked: Dict[int, List[Dict]] = {}
    n_ego = 0
    for j in range(n_dense):
        kept = filter_ego_detections(by_frame.get(j, []), cfg, size)
        n_ego += len(by_frame.get(j, [])) - len(kept)
        tracked[j] = tracker.update(kept, j)
    tracked = tracker.filter_short_tracks(tracked)
    flat = []   # keep only the sampled frames, re-indexed to sampled-frame numbers
    for j in sorted(tracked):
        if j % mult == 0:
            flat.extend(dict(d, frame_id=j // mult) for d in tracked[j])
    (cache / "tracks.json").write_text(json.dumps(flat), encoding="utf-8")
    on_sampled = sum(len(v) for j, v in by_frame.items() if j % mult == 0)
    LOGGER.info("  on sampled frames: detections %d | tracked objects %d | rider's-own-bike boxes removed (all frames) %d",
                on_sampled, len(flat), n_ego)


def step_ego(cache: Path, cfg: Config) -> None:
    feats, affine = estimate_video(frame_paths(cache), cfg.ego_box)
    np.savez_compressed(cache / "ego_motion.npz", feats=feats, affine=affine)


def step_graphs(cache: Path, cfg: Config) -> None:
    meta = json.loads((cache / "meta.json").read_text(encoding="utf-8"))
    by_frame: Dict[int, List[Dict]] = defaultdict(list)
    for d in json.loads((cache / "tracks.json").read_text(encoding="utf-8")):
        by_frame[int(d["frame_id"])].append(d)
    affine = np.load(cache / "ego_motion.npz")["affine"]
    arrays = build_video_graphs(by_frame, meta["n_frames"], affine, cfg, meta["image_size"])
    np.savez_compressed(cache / "graphs.npz", **arrays)


@torch.no_grad()
def step_vit(cache: Path, cfg: Config, models: Models) -> None:
    size = int(cfg.vit.image_size)
    paths = frame_paths(cache)
    out, out_flip = [], []
    for i in range(0, len(paths), int(cfg.cache.vit_batch)):
        imgs = []
        for p in paths[i:i + int(cfg.cache.vit_batch)]:
            img = cv2.cvtColor(cv2.resize(_read(p), (size, size)), cv2.COLOR_BGR2RGB)
            imgs.append(torch.from_numpy(img).permute(2, 0, 1))
        x = torch.stack(imgs).to(models.device).float().div(255.0)
        x = (x - 0.5) / 0.5                      # normalisation this ViT checkpoint was trained with
        out.append(models.vit(x).float().cpu().numpy())
        if cfg.cache.flip_aug:
            out_flip.append(models.vit(torch.flip(x, dims=[3])).float().cpu().numpy())
    np.save(cache / "vit.npy", np.concatenate(out).astype(np.float16))
    if cfg.cache.flip_aug:
        np.save(cache / "vit_flip.npy", np.concatenate(out_flip).astype(np.float16))


_OUTPUTS = {"frames": "meta.json", "detect": "detections.json", "track": "tracks.json",
            "ego": "ego_motion.npz", "graphs": "graphs.npz", "vit": "vit.npy"}
# a step must be redone when an earlier step it depends on was redone
_DEPENDS = {"detect": ["frames"], "track": ["detect"], "ego": ["frames"],
            "graphs": ["track", "ego"], "vit": ["frames"]}


def process_video(video: Optional[Path], video_id: str, cache: Path, cfg: Config, models: Models,
                  steps: Sequence[str] = STEPS, force: bool = False) -> None:
    redone = set()
    for step in STEPS:
        if step not in steps:
            continue
        done = (cache / _OUTPUTS[step]).exists()
        stale = any(d in redone for d in _DEPENDS.get(step, []))
        if done and not force and not stale:
            continue
        LOGGER.info("  %s", step)
        if step == "frames":
            if video is None:
                raise FileNotFoundError(f"video file for {video_id} not found; pass --videos")
            step_frames(video, cache, cfg, video_id)
        elif step == "detect":
            if video is None:
                raise FileNotFoundError(f"video file for {video_id} not found; pass --videos")
            step_detect(video, cache, cfg, models)
        elif step == "track":
            step_track(cache, cfg)
        elif step == "ego":
            step_ego(cache, cfg)
        elif step == "graphs":
            step_graphs(cache, cfg)
        elif step == "vit":
            step_vit(cache, cfg, models)
        redone.add(step)


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Build the per-video cache")
    p.add_argument("--config", required=True)
    p.add_argument("--videos", help="Video file or folder (searched recursively)")
    p.add_argument("--all", action="store_true", help="Process every video under --videos, not only marked ones")
    p.add_argument("--only", nargs="*", help="Only these file names or video ids")
    p.add_argument("--steps", nargs="*", choices=STEPS, default=list(STEPS))
    p.add_argument("--force", action="store_true", help="Redo the selected steps even if cached")
    return p.parse_args()


def main() -> None:
    args = parse_args()
    cfg = load_config(args.config)
    lookup = {p.name: p for p in find_videos(args.videos)} if args.videos else {}
    events = load_events(cfg.event_labeling.events_file)["videos"]

    jobs = []   # (video_id, path or None)
    if args.all:
        jobs = [(events.get(n, {}).get("video_id") or make_video_id(n), p) for n, p in lookup.items()]
    else:
        for name, e in sorted(events.items()):
            if e.get("status") != "done":
                continue
            path = Path(e["path"]) if e.get("path") and Path(e["path"]).exists() else lookup.get(name)
            jobs.append((e["video_id"], path))
    if args.only:
        jobs = [j for j in jobs if j[0] in args.only or (j[1] and j[1].name in args.only)]

    models = Models(cfg)
    root = Path(cfg.paths.video_cache)
    LOGGER.info("%d videos -> %s", len(jobs), root)
    for i, (vid, path) in enumerate(jobs, start=1):
        LOGGER.info("[%d/%d] %s", i, len(jobs), vid)
        try:
            process_video(path, vid, root / vid, cfg, models, args.steps, args.force)
        except (IOError, FileNotFoundError) as exc:
            LOGGER.error("  skipped: %s", exc)


if __name__ == "__main__":
    main()
