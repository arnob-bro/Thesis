"""Shared helpers for the collision-event annotation file.

Schema of ``collision_events.json`` (one entry per video, keyed by file name)::

    {
      "version": 1,
      "videos": {
        "12.mp4": {
          "path": "data/raw_videos/train/12.mp4",
          "video_id": "12",                 # ASCII id used in sequence names
          "split": null,                    # optional override: train | val | test
          "fps": 30.0,
          "duration_s": 132.5,
          "status": "done",                 # done | in_progress
          "events": [{"time_s": 63.4, "type": "collision", "note": ""}],
          "ignore": [[0.0, 5.2], [70.0, 95.0]]   # intros, replays, slow-mo, edits
        }
      }
    }

A video with ``status == "done"`` and no events is valid: every usable
window in it becomes a SAFE sample.
"""
from __future__ import annotations

import csv
import hashlib
import json
import re
from pathlib import Path
from typing import Dict, List, Optional

VIDEO_EXTS = {".mp4", ".mov", ".mkv", ".avi", ".m4v", ".webm"}
SPLITS = ("train", "val", "test")


def load_events(path: str | Path) -> Dict:
    path = Path(path)
    if not path.exists():
        return {"version": 1, "videos": {}}
    with path.open("r", encoding="utf-8") as fp:
        data = json.load(fp)
    data.setdefault("version", 1)
    data.setdefault("videos", {})
    return data


def save_events(data: Dict, path: str | Path) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    for entry in data["videos"].values():
        entry["events"] = sorted(entry.get("events", []), key=lambda e: e["time_s"])
        entry["ignore"] = sorted(entry.get("ignore", []))
    tmp = path.with_suffix(".tmp")
    with tmp.open("w", encoding="utf-8") as fp:
        json.dump(data, fp, indent=2, ensure_ascii=False)
    tmp.replace(path)  # atomic: a crash never leaves a half-written file


def make_video_id(name: str) -> str:
    """ASCII-safe id. Plain names (``12.mp4``) keep their stem; names with
    Bengali/emoji/spaces become ``<ascii_slug>_<hash6>`` so folder names and
    OpenCV paths stay safe on Windows."""
    stem = Path(name).stem
    if re.fullmatch(r"[A-Za-z0-9_-]+", stem):
        return stem
    slug = re.sub(r"[^A-Za-z0-9]+", "_", stem.encode("ascii", "ignore").decode()).strip("_")
    digest = hashlib.sha1(stem.encode("utf-8")).hexdigest()[:6]
    return f"{slug[:30]}_{digest}" if slug else f"v_{digest}"


def find_videos(root: str | Path) -> List[Path]:
    root = Path(root)
    if root.is_file():
        return [root]
    return sorted(p for p in root.rglob("*") if p.suffix.lower() in VIDEO_EXTS)


def fmt_time(seconds: Optional[float]) -> str:
    if seconds is None:
        return "-"
    m, s = divmod(float(seconds), 60.0)
    return f"{int(m):02d}:{s:05.2f}"


def parse_time(text: str) -> float:
    """Accept ``63.4``, ``1:03.4`` or ``0:01:03.4``."""
    parts = [float(p) for p in str(text).strip().split(":")]
    seconds = 0.0
    for p in parts:
        seconds = seconds * 60.0 + p
    return seconds


def assign_split(video_path: Path, video_id: str, entry: Dict, ratios: Dict[str, float]) -> str:
    """Split is chosen per VIDEO (never per sequence) to avoid leakage.

    Priority: explicit ``split`` in the entry > parent folder named
    train/val/test > deterministic hash of the video id (stable as you add
    more videos).
    """
    if entry.get("split") in SPLITS:
        return entry["split"]
    for part in reversed(video_path.parts[:-1]):
        if part.lower() in SPLITS:
            return part.lower()
    h = int(hashlib.md5(video_id.encode("utf-8")).hexdigest()[:8], 16) / 0xFFFFFFFF
    edge = 0.0
    for split in SPLITS:
        edge += float(ratios.get(split, 0.0))
        if h < edge:
            return split
    return "train"


def export_csv(data: Dict, path: str | Path) -> None:
    with Path(path).open("w", encoding="utf-8-sig", newline="") as fp:
        w = csv.writer(fp)
        w.writerow(["video", "video_id", "status", "event_time_s", "event_time", "type", "note"])
        for name, e in sorted(data["videos"].items()):
            events = e.get("events") or [None]
            for ev in events:
                w.writerow([
                    name, e.get("video_id"), e.get("status"),
                    "" if ev is None else ev["time_s"],
                    "" if ev is None else fmt_time(ev["time_s"]),
                    "none" if ev is None else ev["type"],
                    "" if ev is None else ev.get("note", ""),
                ])
