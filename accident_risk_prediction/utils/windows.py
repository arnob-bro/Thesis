"""Window planning: turn collision times into labelled sequences.

A window is ``sequence.length`` sampled frames. Its time-to-event is

    TTE = (next event time) - (time of the window's LAST frame)

so the model only ever sees frames before the moment it must decide.
"""
from __future__ import annotations

import random
from dataclasses import dataclass
from typing import Dict, List, Optional, Tuple

from utils.config import Config

RISK_NAMES = {0: "safe", 1: "slight", 2: "moderate", 3: "high"}
BIN_TO_CLASS = {"slight": 1, "moderate": 2, "high": 3}


@dataclass
class Window:
    end: int                    # index of last sampled frame (0-based)
    start: int
    t_end: float
    risk: Optional[int]         # None = ambiguous buffer (only in dense mode)
    tte: Optional[float]        # None = no upcoming event
    event_type: Optional[str]
    event_time: Optional[float]


def label_from_tte(tte: Optional[float], lab: Config) -> Optional[int]:
    """Risk class for a time-to-event, or None for the ambiguous buffer."""
    if tte is None or tte >= float(lab.safe_min_tte):
        return 0
    for name, (lo, hi) in lab.bins.items():
        if float(lo) <= tte < float(hi):
            return BIN_TO_CLASS[name]
    return None


def post_event_skip(event_type: Optional[str], lab: Config) -> float:
    """Seconds of aftermath to drop after an event of this type.

    ``post_event_skip`` is either one number for every type, or a mapping
    ``{type: seconds, default: seconds}``. A near miss needs a much shorter
    skip than a crash: the rider keeps riding, so the seconds after it can be
    the real lead-up to a second, separate event.
    """
    skip = lab.post_event_skip
    if isinstance(skip, (int, float)):
        return float(skip)
    return float(skip.get(event_type, skip.get("default", 5.0)))


def classify(end: int, fps: float, seq_len: int, events: List[Dict],
             ignore: List[List[float]], lab: Config) -> Optional[Window]:
    """Label one window, or return None if it must not be used at all."""
    start = end - seq_len + 1
    t_s, t_e = start / fps, end / fps
    safe_min = float(lab.safe_min_tte)

    if any(t_s <= b and t_e >= a for a, b in ignore):
        return None
    if any(t_e >= ev["time_s"] and t_s <= ev["time_s"] + post_event_skip(ev.get("type"), lab)
           for ev in events):
        return None   # touches an event or its aftermath
    nxt = next((ev for ev in events if ev["time_s"] >= t_e), None)
    tte = None if nxt is None else nxt["time_s"] - t_e
    if tte is not None and tte < safe_min and any(t_e < a and b < nxt["time_s"] for a, b in ignore):
        return None   # an edit cut sits between window and event: real gap unknown
    return Window(end, start, t_e, label_from_tte(tte, lab), tte,
                  nxt["type"] if nxt else None, nxt["time_s"] if nxt else None)


def plan_windows(n_samples: int, fps: float, seq_len: int, events: List[Dict],
                 ignore: List[List[float]], lab: Config) -> List[Window]:
    """Training windows: dense before events, sparse and capped elsewhere."""
    events = sorted(events, key=lambda e: e["time_s"])
    risk_stride, safe_stride = max(1, int(lab.risk_stride)), max(1, int(lab.safe_stride))
    risky: List[Window] = []
    safe: List[Window] = []
    for end in range(seq_len - 1, n_samples, risk_stride):
        w = classify(end, fps, seq_len, events, ignore, lab)
        if w is None or w.risk is None:
            continue
        if w.risk > 0:
            risky.append(w)
        elif (end - (seq_len - 1)) % safe_stride == 0:
            safe.append(w)
    cap = lab.get("max_safe_per_video")
    if cap is not None and len(safe) > int(cap):
        step = len(safe) / int(cap)
        safe = [safe[int(i * step)] for i in range(int(cap))]   # evenly spread
    return sorted(risky + safe, key=lambda w: w.end)


def dense_windows(n_samples: int, fps: float, seq_len: int, events: List[Dict],
                  ignore: List[List[float]], lab: Config) -> List[Window]:
    """Every usable window (stride 1), for sliding-window evaluation."""
    events = sorted(events, key=lambda e: e["time_s"])
    out = []
    for end in range(seq_len - 1, n_samples):
        w = classify(end, fps, seq_len, events, ignore, lab)
        if w is not None:
            out.append(w)
    return out


def assign_folds(videos: List[Tuple[str, bool]], k: int, seed: int) -> Dict[str, int]:
    """Grouped folds over videos, balanced between videos with / without events."""
    folds: Dict[str, int] = {}
    for flag in (True, False):
        ids = sorted(v for v, has_event in videos if has_event == flag)
        random.Random(seed).shuffle(ids)
        offset = 0 if flag else len([1 for _, h in videos if h]) % max(k, 1)
        for i, vid in enumerate(ids):
            folds[vid] = (i + offset) % k
    return folds
