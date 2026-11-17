"""Accident-anticipation metrics from sliding-window scores over whole videos.

The model is run on every window of a test video (stride 1 frame). Each
window has an alarm score = P(risk >= evaluation.alarm_class). Then:

* event recall   share of events with an alarm raised before impact
                 (inside the horizon = event_labeling.safe_min_tte seconds)
* mean TTA       mean time-to-accident: how many seconds before impact the
                 score first crosses the threshold (detected events only)
* false alarms   alarm onsets per minute of safe riding
* AP             average precision of the score, risky windows vs safe windows
"""
from __future__ import annotations

from collections import defaultdict
from typing import Dict, List

import numpy as np
from sklearn.metrics import average_precision_score

from utils.config import Config


def anticipation_metrics(records: List[Dict], cfg: Config) -> Dict[str, object]:
    """records: one dict per window with video_id, end, risk (None=buffer),
    tte (None=no upcoming event), event_time, score."""
    fps = float(cfg.frame_extraction.frame_rate)
    horizon = float(cfg.event_labeling.safe_min_tte)

    events: Dict[tuple, List[Dict]] = defaultdict(list)   # windows inside the horizon of an event
    safe: Dict[str, List[Dict]] = defaultdict(list)
    for r in records:
        if r["tte"] is not None and r["tte"] < horizon:
            events[(r["video_id"], r["event_time"])].append(r)
        elif r["risk"] == 0:
            safe[r["video_id"]].append(r)
    for lst in list(events.values()) + list(safe.values()):
        lst.sort(key=lambda r: r["end"])
    safe_minutes = sum(len(v) for v in safe.values()) / fps / 60.0

    def at(thr: float) -> Dict[str, float]:
        ttas = []
        for wins in events.values():
            hit = next((w for w in wins if w["score"] >= thr), None)
            if hit is not None:
                ttas.append(hit["tte"])
        onsets = 0
        for wins in safe.values():
            prev_end, prev_on = None, False
            for w in wins:
                on = w["score"] >= thr
                if on and not (prev_on and prev_end == w["end"] - 1):
                    onsets += 1
                prev_end, prev_on = w["end"], on
        return {
            "threshold": round(thr, 3),
            "event_recall": len(ttas) / len(events) if events else float("nan"),
            "mean_tta_s": float(np.mean(ttas)) if ttas else 0.0,
            "false_alarms_per_min": onsets / safe_minutes if safe_minutes > 0 else float("nan"),
        }

    y = [1 if r["risk"] else 0 for r in records if r["risk"] is not None]
    s = [r["score"] for r in records if r["risk"] is not None]
    ap = float(average_precision_score(y, s)) if 0 < sum(y) < len(y) else float("nan")

    sweep = [at(t) for t in np.arange(0.05, 1.0, 0.05)]
    # operating point: highest threshold that still catches >= 80% of events
    ok = [p for p in sweep if p["event_recall"] >= 0.8]
    return {
        "n_events": len(events),
        "safe_minutes": round(safe_minutes, 2),
        "average_precision": ap,
        "at_threshold": at(float(cfg.evaluation.alarm_threshold)),
        "at_recall_80": max(ok, key=lambda p: p["threshold"]) if ok else None,
        "sweep": sweep,
    }
