"""ByteTrack wrapper over the `supervision` library.

Tracking constraints (MD):
    - preserve stable object IDs
    - reject tracks shorter than 3 frames
    - reject low-confidence tracks
"""
from __future__ import annotations

from collections import defaultdict
from typing import Dict, List

import numpy as np

try:
    import supervision as sv
except ImportError as exc:                              # pragma: no cover
    sv = None
    _IMPORT_ERROR: Exception | None = exc
else:
    _IMPORT_ERROR = None


class ByteTrackWrapper:
    def __init__(
        self,
        track_thresh: float = 0.5,
        match_thresh: float = 0.8,
        track_buffer: int = 30,
        frame_rate: int = 5,
        min_track_length: int = 3,
    ) -> None:
        if sv is None:
            raise ImportError(
                "`supervision` package is required for ByteTrackWrapper"
            ) from _IMPORT_ERROR
        self.tracker = sv.ByteTrack(
            track_activation_threshold=track_thresh,
            lost_track_buffer=track_buffer,
            minimum_matching_threshold=match_thresh,
            frame_rate=frame_rate,
        )
        self.min_track_length = int(min_track_length)
        self._frame_counts: Dict[int, int] = defaultdict(int)

    def reset(self) -> None:
        self.tracker.reset()
        self._frame_counts.clear()

    def update(self, detections: List[Dict], frame_id: int) -> List[Dict]:
        """Update tracker with one frame's detections, return tracked objects."""
        if sv is None:
            raise ImportError("supervision missing")

        if not detections:
            empty = sv.Detections.empty()
            tracked = self.tracker.update_with_detections(empty)
            return []

        xyxy = np.array(
            [
                [d["bbox"][0], d["bbox"][1],
                 d["bbox"][0] + d["bbox"][2], d["bbox"][1] + d["bbox"][3]]
                for d in detections
            ],
            dtype=np.float32,
        )
        conf = np.array([d["confidence"] for d in detections], dtype=np.float32)
        cls = np.array([d["class_id"] for d in detections], dtype=int)

        sv_dets = sv.Detections(xyxy=xyxy, confidence=conf, class_id=cls)
        tracked = self.tracker.update_with_detections(sv_dets)

        out: List[Dict] = []
        if tracked.tracker_id is None:
            return out
        for i, tid in enumerate(tracked.tracker_id):
            if tid is None or int(tid) < 0:
                continue
            x1, y1, x2, y2 = tracked.xyxy[i]
            cls_i = (
                int(tracked.class_id[i]) if tracked.class_id is not None else 0
            )
            conf_i = (
                float(tracked.confidence[i]) if tracked.confidence is not None else 0.0
            )
            out.append(
                {
                    "frame_id": int(frame_id),
                    "track_id": int(tid),
                    "class_id": cls_i,
                    "confidence": conf_i,
                    "bbox": [
                        float(x1), float(y1),
                        float(x2 - x1), float(y2 - y1),
                    ],
                }
            )
            self._frame_counts[int(tid)] += 1
        return out

    def filter_short_tracks(self, frames: Dict[int, List[Dict]]) -> Dict[int, List[Dict]]:
        """Remove tracks whose total observed length is below `min_track_length`."""
        counts: Dict[int, int] = defaultdict(int)
        for frame_dets in frames.values():
            for det in frame_dets:
                counts[det["track_id"]] += 1
        keep_ids = {tid for tid, c in counts.items() if c >= self.min_track_length}
        return {
            fid: [d for d in dets if d["track_id"] in keep_ids]
            for fid, dets in frames.items()
        }
