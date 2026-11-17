"""Turn marked collision times into the sequence manifest.

Reads ``collision_events.json`` (tools/mark_collisions.py) and writes
``data/manifests/sequences.csv`` with one row per sequence:

    sequence_id, video_id, split, fold, start, end, risk_level, tte, ...

``start`` / ``end`` are frame indices into ``data/video_cache/<video_id>/``,
so no frames are copied and the manifest is rebuilt in seconds whenever the
bins in config.yaml change.

Labels (motorcycle profile, see ``event_labeling`` in config.yaml):

    TTE in [0, 1.5)   -> 3 high        TTE in [3, 4.5) -> 1 slight
    TTE in [1.5, 3)   -> 2 moderate    TTE in [4.5, 6) -> skipped (ambiguous)
    TTE >= 6 or no upcoming event -> 0 safe

Split and fold are assigned per VIDEO, so a ride never leaks across sets.

Usage
    python -m tools.make_event_sequences --config configs/config.yaml
    python -m tools.make_event_sequences --config configs/config.yaml --dry-run
"""
from __future__ import annotations

import argparse
import csv
import json
import math
from collections import Counter, defaultdict
from pathlib import Path
from typing import Dict, List, Optional

import cv2

from utils.config import load_config
from utils.events import assign_split, find_videos, fmt_time, load_events
from utils.windows import RISK_NAMES, assign_folds, plan_windows

FIELDS = ["sequence_id", "video_id", "source_file", "split", "fold", "start", "end",
          "risk_level", "tte", "event_type", "event_time", "t_end"]


def _n_samples(cache: Path, video: Optional[Path], target_fps: float) -> Optional[int]:
    meta = cache / "meta.json"
    if meta.exists():
        return int(json.loads(meta.read_text(encoding="utf-8"))["n_frames"])
    if video is not None and video.exists():
        cap = cv2.VideoCapture(str(video))
        fps = float(cap.get(cv2.CAP_PROP_FPS)) or 30.0
        n = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        cap.release()
        return int(math.floor(n / fps * target_fps))
    return None


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Build the sequence manifest from collision times")
    p.add_argument("--config", required=True)
    p.add_argument("--videos", help="Folder to look up videos that are not cached yet")
    p.add_argument("--events", help="Override events JSON path")
    p.add_argument("--manifest", help="Override manifest path")
    p.add_argument("--dry-run", action="store_true", help="Print counts, write nothing")
    return p.parse_args()


def main() -> None:
    args = parse_args()
    cfg = load_config(args.config)
    lab = cfg.event_labeling
    data = load_events(args.events or lab.events_file)
    manifest = Path(args.manifest or cfg.paths.manifest)
    fps = float(cfg.frame_extraction.frame_rate)
    seq_len = int(cfg.sequence.length)
    cache_root = Path(cfg.paths.video_cache)
    lookup = {p.name: p for p in find_videos(args.videos)} if args.videos else {}

    done = {n: e for n, e in data["videos"].items() if e.get("status") == "done"}
    for n, e in data["videos"].items():
        if n not in done:
            print(f"skip {e['video_id']}: status={e.get('status')} (finish marking first)")
    folds = assign_folds([(e["video_id"], bool(e.get("events"))) for e in done.values()],
                         int(cfg.cv.folds), int(cfg.seed))

    rows: List[Dict] = []
    counts: Dict[str, Counter] = defaultdict(Counter)
    fold_counts: Dict[int, Counter] = defaultdict(Counter)
    n_videos: Counter = Counter()
    for name, e in sorted(done.items(), key=lambda kv: kv[1]["video_id"]):
        vid = e["video_id"]
        path = Path(e["path"]) if e.get("path") and Path(e["path"]).exists() else lookup.get(name)
        n = _n_samples(cache_root / vid, path, fps)
        if n is None:
            print(f"skip {vid}: not cached and video file not found (pass --videos)")
            continue
        split = assign_split(path or Path(name), vid, e, dict(cfg.split))
        windows = plan_windows(n, fps, seq_len, e.get("events", []), e.get("ignore", []), lab)
        c = Counter(w.risk for w in windows)
        evs = ", ".join(fmt_time(ev["time_s"]) for ev in e.get("events", [])) or "none"
        print(f"{vid:<38} fold {folds[vid]} {split:<5} events[{evs}]  "
              + " ".join(f"{RISK_NAMES[k]}={c.get(k, 0)}" for k in range(4)))
        counts[split].update(c)
        fold_counts[folds[vid]].update(c)
        n_videos[split] += 1
        for i, w in enumerate(windows):
            rows.append({
                "sequence_id": f"{vid}_seq{i:04d}", "video_id": vid, "source_file": name,
                "split": split, "fold": folds[vid], "start": w.start, "end": w.end,
                "risk_level": w.risk,
                "tte": "" if w.tte is None else round(w.tte, 3),
                "event_type": w.event_type or "",
                "event_time": "" if w.event_time is None else w.event_time,
                "t_end": round(w.t_end, 3),
            })

    head = " ".join(f"{RISK_NAMES[k]:>9}" for k in range(4)) + f" {'total':>7}"
    print(f"\n{'fold':<6} {head}")
    for f in sorted(fold_counts):
        c = fold_counts[f]
        print(f"{f:<6} " + " ".join(f"{c.get(k, 0):>9}" for k in range(4)) + f" {sum(c.values()):>7}")
    print(f"\n{'split':<6} {'videos':>6} {head}")
    for split in ("train", "val", "test"):
        c = counts[split]
        print(f"{split:<6} {n_videos[split]:>6} "
              + " ".join(f"{c.get(k, 0):>9}" for k in range(4)) + f" {sum(c.values()):>7}")

    if args.dry_run:
        print("\n(dry run: nothing written)")
        return
    manifest.parent.mkdir(parents=True, exist_ok=True)
    with manifest.open("w", encoding="utf-8-sig", newline="") as fp:
        w = csv.DictWriter(fp, fieldnames=FIELDS)
        w.writeheader()
        w.writerows(rows)
    print(f"\nWrote {len(rows)} sequences -> {manifest}")


if __name__ == "__main__":
    main()
