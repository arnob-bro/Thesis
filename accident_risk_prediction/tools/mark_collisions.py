"""Mark collision / near-miss times in videos and list them.

Three ways to use it:

1. Interactive player (OpenCV window) - watch each video and press a key at
   the moment of impact:

       python -m tools.mark_collisions --config configs/config.yaml --videos data/raw_videos

2. Type times by hand (e.g. while watching in VLC) into a CSV and import it:

       python -m tools.mark_collisions --config configs/config.yaml --import-csv my_times.csv

   CSV columns: video,time,type[,note]
       12.mp4,1:03.4,collision
       12.mp4,0:00-0:06,ignore          <- ignore range (intro, replay, slow-mo)
       15.mp4,,none                     <- annotated, no event (all safe)

3. List / export what has been marked:

       python -m tools.mark_collisions --config configs/config.yaml --list
       python -m tools.mark_collisions --config configs/config.yaml --export-csv collision_times.csv

Player keys
    space        play / pause
    d / a        +1 / -1 frame          (also right / left arrow)
    l / j        +1 s / -1 s
    L / J        +10 s / -10 s
    + / -        playback speed up / down
    c            mark COLLISION at the current frame (first contact / fall start)
    n            mark NEAR MISS (moment of closest approach)
    x            mark LOSS OF CONTROL (skid, tank-slap, solo fall onset)
    [ / ]        start / end an IGNORE range (intro, replay, slow motion, cut)
    u            undo last mark
    s            save as DONE and go to next video
    w            save progress (in_progress) and go to next video
    q / Esc      save progress and quit
"""
from __future__ import annotations

import argparse
import csv
import sys
from pathlib import Path
from typing import Dict, List, Optional

import cv2
import numpy as np

from utils.config import load_config
from utils.events import (
    export_csv, find_videos, fmt_time, load_events, make_video_id,
    parse_time, save_events,
)

WIN = "mark_collisions"
KEY_LEFT = {2424832, 65361}
KEY_RIGHT = {2555904, 65363}
TYPE_KEYS = {ord("c"): "collision", ord("n"): "near_miss", ord("x"): "loss_of_control"}
TYPE_COLOURS = {"collision": (0, 0, 255), "near_miss": (0, 165, 255), "loss_of_control": (255, 0, 255)}


# ---------------------------------------------------------------------------
# Player
# ---------------------------------------------------------------------------

class Player:
    def __init__(self, path: Path) -> None:
        self.cap = cv2.VideoCapture(str(path))
        if not self.cap.isOpened():
            raise IOError(f"Cannot open video: {path}")
        self.fps = float(self.cap.get(cv2.CAP_PROP_FPS)) or 30.0
        self.n_frames = int(self.cap.get(cv2.CAP_PROP_FRAME_COUNT))
        self.pos = -1
        self.frame: Optional[np.ndarray] = None
        self.seek(0)

    @property
    def duration(self) -> float:
        return self.n_frames / self.fps

    @property
    def time_s(self) -> float:
        return max(self.pos, 0) / self.fps

    def seek(self, idx: int) -> None:
        idx = int(np.clip(idx, 0, max(self.n_frames - 1, 0)))
        if idx != self.pos + 1:
            self.cap.set(cv2.CAP_PROP_POS_FRAMES, idx)
        ok, frame = self.cap.read()
        if ok:
            self.pos, self.frame = idx, frame

    def next(self) -> bool:
        ok, frame = self.cap.read()
        if ok:
            self.pos += 1
            self.frame = frame
        return ok

    def release(self) -> None:
        self.cap.release()


def _render(player: Player, entry: Dict, playing: bool, speed: float,
            pending_ignore: Optional[float], idx: int, total: int) -> np.ndarray:
    frame = player.frame
    h, w = frame.shape[:2]
    scale = min(1.0, 1280.0 / w, 720.0 / h)
    img = cv2.resize(frame, (int(w * scale), int(h * scale))) if scale < 1 else frame.copy()
    H, W = img.shape[:2]

    # Header
    cv2.rectangle(img, (0, 0), (W, 58), (0, 0, 0), -1)
    head = (f"[{idx}/{total}] {entry['video_id']}   {fmt_time(player.time_s)} / "
            f"{fmt_time(player.duration)}   frame {player.pos}   "
            f"{'PLAY' if playing else 'PAUSE'} x{speed:g}")
    cv2.putText(img, head, (8, 22), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 1, cv2.LINE_AA)
    help_txt = "c=collision n=near-miss x=loss-ctrl [ ]=ignore u=undo s=done w=later q=quit"
    cv2.putText(img, help_txt, (8, 48), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (180, 180, 180), 1, cv2.LINE_AA)

    # Timeline with marks
    bar_y = H - 18
    cv2.rectangle(img, (0, bar_y - 6), (W, H), (30, 30, 30), -1)
    dur = max(player.duration, 1e-6)
    for a, b in entry.get("ignore", []):
        cv2.rectangle(img, (int(a / dur * W), bar_y - 4), (int(b / dur * W), H - 2), (90, 90, 90), -1)
    if pending_ignore is not None:
        cv2.rectangle(img, (int(pending_ignore / dur * W), bar_y - 4),
                      (int(player.time_s / dur * W), H - 2), (0, 200, 200), -1)
    for ev in entry.get("events", []):
        x = int(ev["time_s"] / dur * W)
        cv2.line(img, (x, bar_y - 6), (x, H), TYPE_COLOURS.get(ev["type"], (0, 0, 255)), 3)
    cv2.line(img, (int(player.time_s / dur * W), bar_y - 8), (int(player.time_s / dur * W), H), (255, 255, 255), 1)

    # Event list
    y = 80
    for ev in entry.get("events", []):
        txt = f"{ev['type']:<16} {fmt_time(ev['time_s'])}"
        cv2.putText(img, txt, (8, y), cv2.FONT_HERSHEY_SIMPLEX, 0.55, TYPE_COLOURS.get(ev["type"], (0, 0, 255)), 2, cv2.LINE_AA)
        y += 22
    if pending_ignore is not None:
        cv2.putText(img, f"ignore from {fmt_time(pending_ignore)} ... press ]", (8, y),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.55, (0, 200, 200), 2, cv2.LINE_AA)
    return img


def annotate_video(path: Path, entry: Dict, idx: int, total: int) -> str:
    """Returns 'done', 'later' or 'quit'. Mutates `entry` in place."""
    player = Player(path)
    entry["fps"] = round(player.fps, 3)
    entry["duration_s"] = round(player.duration, 3)
    history: List[str] = []           # for undo: "event" | "ignore"
    playing, speed = False, 1.0
    pending_ignore: Optional[float] = None

    cv2.namedWindow(WIN, cv2.WINDOW_NORMAL)
    try:
        while True:
            cv2.imshow(WIN, _render(player, entry, playing, speed, pending_ignore, idx, total))
            delay = max(1, int(1000.0 / (player.fps * speed))) if playing else 30
            key = cv2.waitKeyEx(delay)
            if cv2.getWindowProperty(WIN, cv2.WND_PROP_VISIBLE) < 1:
                return "quit"

            if playing and key == -1:
                if not player.next():
                    playing = False
                continue
            if key == -1:
                continue

            ch = key & 0xFF if key < 256 else None
            step_1s = int(round(player.fps))
            if ch == ord(" "):
                playing = not playing
            elif ch == ord("d") or key in KEY_RIGHT:
                playing = False; player.seek(player.pos + 1)
            elif ch == ord("a") or key in KEY_LEFT:
                playing = False; player.seek(player.pos - 1)
            elif ch == ord("l"):
                player.seek(player.pos + step_1s)
            elif ch == ord("j"):
                player.seek(player.pos - step_1s)
            elif ch == ord("L"):
                player.seek(player.pos + 10 * step_1s)
            elif ch == ord("J"):
                player.seek(player.pos - 10 * step_1s)
            elif ch in (ord("+"), ord("=")):
                speed = min(speed * 2, 8.0)
            elif ch in (ord("-"), ord("_")):
                speed = max(speed / 2, 0.125)
            elif ch in TYPE_KEYS:
                playing = False
                entry["events"].append({"time_s": round(player.time_s, 3), "type": TYPE_KEYS[ch], "note": ""})
                history.append("event")
                print(f"  + {TYPE_KEYS[ch]} at {fmt_time(player.time_s)}")
            elif ch == ord("["):
                pending_ignore = player.time_s
            elif ch == ord("]") and pending_ignore is not None:
                a, b = sorted((pending_ignore, player.time_s))
                entry["ignore"].append([round(a, 3), round(b, 3)])
                history.append("ignore")
                pending_ignore = None
                print(f"  + ignore {fmt_time(a)} - {fmt_time(b)}")
            elif ch == ord("u"):
                if pending_ignore is not None:
                    pending_ignore = None
                elif history:
                    kind = history.pop()
                    removed = entry["events"].pop() if kind == "event" else entry["ignore"].pop()
                    print(f"  - undo {removed}")
            elif ch == ord("s"):
                return "done"
            elif ch == ord("w"):
                return "later"
            elif ch in (ord("q"), 27):
                return "quit"
    finally:
        player.release()


# ---------------------------------------------------------------------------
# CSV import / listing
# ---------------------------------------------------------------------------

def import_csv(data: Dict, csv_path: Path, video_paths: Dict[str, Path]) -> None:
    with csv_path.open("r", encoding="utf-8-sig", newline="") as fp:
        rows = list(csv.reader(fp))
    if rows and rows[0] and rows[0][0].strip().lower() == "video":
        rows = rows[1:]
    touched = set()
    for row in rows:
        if not row or not row[0].strip():
            continue
        name = Path(row[0].strip()).name
        time_txt = row[1].strip() if len(row) > 1 else ""
        etype = (row[2].strip().lower() if len(row) > 2 else "collision") or "collision"
        note = row[3].strip() if len(row) > 3 else ""
        entry = _get_entry(data, name, video_paths.get(name))
        if name not in touched:          # CSV replaces previous marks for that video
            entry["events"], entry["ignore"] = [], []
            touched.add(name)
        if etype == "none":
            pass
        elif etype == "ignore":
            a, b = (parse_time(t) for t in time_txt.split("-"))
            entry["ignore"].append([a, b])
        else:
            entry["events"].append({"time_s": parse_time(time_txt), "type": etype, "note": note})
        entry["status"] = "done"
    print(f"Imported {len(touched)} videos from {csv_path}")


def print_listing(data: Dict) -> None:
    videos = data["videos"]
    n_events = sum(len(v.get("events", [])) for v in videos.values())
    print(f"{len(videos)} videos, {n_events} events\n")
    print(f"{'video_id':<38} {'status':<12} {'events'}")
    print("-" * 90)
    for name, e in sorted(videos.items(), key=lambda kv: kv[1].get("video_id", "")):
        evs = ", ".join(f"{fmt_time(ev['time_s'])} {ev['type']}" for ev in e.get("events", [])) or "(none)"
        print(f"{e.get('video_id', ''):<38} {e.get('status', ''):<12} {evs}")
        if e.get("ignore"):
            ign = ", ".join(f"{fmt_time(a)}-{fmt_time(b)}" for a, b in e["ignore"])
            print(f"{'':<38} {'':<12} ignore: {ign}")
        if e.get("video_id") != Path(name).stem:
            print(f"{'':<38} {'':<12} file: {name}")


def _get_entry(data: Dict, name: str, path: Optional[Path]) -> Dict:
    entry = data["videos"].setdefault(name, {
        "path": None, "video_id": make_video_id(name), "split": None,
        "fps": None, "duration_s": None, "status": "in_progress",
        "events": [], "ignore": [],
    })
    if path is not None:
        entry["path"] = path.as_posix()
    return entry


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Mark collision times in videos")
    p.add_argument("--config", required=True)
    p.add_argument("--videos", help="Video file or folder (searched recursively)")
    p.add_argument("--events", help="Override events JSON path")
    p.add_argument("--only", nargs="*", help="Only these file names or video ids")
    p.add_argument("--redo", action="store_true", help="Also open videos already marked done")
    p.add_argument("--import-csv", help="Import hand-typed times from CSV")
    p.add_argument("--export-csv", help="Export all marks to CSV")
    p.add_argument("--list", action="store_true", help="Print collision times per video")
    return p.parse_args()


def main() -> None:
    args = parse_args()
    cfg = load_config(args.config)
    events_path = Path(args.events or cfg.event_labeling.events_file)
    data = load_events(events_path)

    video_paths = {p.name: p for p in find_videos(args.videos)} if args.videos else {}

    if args.import_csv:
        import_csv(data, Path(args.import_csv), video_paths)
        save_events(data, events_path)
    if args.list:
        print_listing(data)
    if args.export_csv:
        export_csv(data, args.export_csv)
        print(f"Exported -> {args.export_csv}")
    if not args.videos or args.import_csv:
        return

    todo = []
    for name, path in sorted(video_paths.items()):
        entry = data["videos"].get(name)
        if args.only and name not in args.only and make_video_id(name) not in args.only:
            continue
        if entry and entry.get("status") == "done" and not args.redo:
            continue
        todo.append((name, path))
    print(f"{len(todo)} videos to annotate (events file: {events_path})")

    for i, (name, path) in enumerate(todo, start=1):
        entry = _get_entry(data, name, path)
        print(f"[{i}/{len(todo)}] {entry['video_id']}  ({name})")
        try:
            result = annotate_video(path, entry, i, len(todo))
        except IOError as exc:
            print(f"  ! {exc}", file=sys.stderr)
            continue
        if result == "done":
            entry["status"] = "done"
        save_events(data, events_path)
        if result == "quit":
            break
    cv2.destroyAllWindows()
    print(f"Saved {events_path}")


if __name__ == "__main__":
    main()
