"""Thin wrapper around an Ultralytics YOLO detector (COCO or fine-tuned).

YOLOv8 is FROZEN per MD (`implementation_contracts_specification.md`):
used only as pretrained detector + feature source. We never re-train it.
"""
from __future__ import annotations

from pathlib import Path
from typing import Dict, List

import numpy as np

try:  # Ultralytics may not be installed at import-doc time
    from ultralytics import YOLO
except ImportError as exc:                              # pragma: no cover
    YOLO = None                                          # type: ignore[assignment]
    _IMPORT_ERROR: Exception | None = exc
else:
    _IMPORT_ERROR = None


class YoloDetector:
    """Pretrained YOLOv8 detector producing per-frame bbox detections."""

    def __init__(
        self,
        weights: str,
        conf_threshold: float = 0.25,
        iou_threshold: float = 0.45,
        device: str = "cuda",
        coco_to_ours: Dict[int, int] | None = None,
        imgsz: int | None = None,
        agnostic_nms: bool = False,
    ) -> None:
        if YOLO is None:
            raise ImportError(
                "ultralytics package is required for YoloDetector"
            ) from _IMPORT_ERROR
        self.model = YOLO(weights)
        self.conf = float(conf_threshold)
        self.iou = float(iou_threshold)
        self.device = device
        # imgsz None = the size the model was trained at. agnostic_nms keeps one box
        # per object when a fine-grained detector hesitates between two classes.
        self.extra = {"agnostic_nms": bool(agnostic_nms)}
        if imgsz:
            self.extra["imgsz"] = int(imgsz)
        # Default coco_to_ours mapping; -1 means "drop this class"
        self.coco_to_ours = {int(k): int(v) for k, v in (coco_to_ours or {}).items()}

    def predict_image(self, image_path: str | Path, frame_id: int) -> List[Dict]:
        results = self.model.predict(
            source=str(image_path),
            conf=self.conf,
            iou=self.iou,
            device=self.device,
            verbose=False,
            **self.extra,
        )
        if not results:
            return []
        return self._results_to_detections(results[0], frame_id)

    def predict_array(self, image: np.ndarray, frame_id: int) -> List[Dict]:
        results = self.model.predict(
            source=image,
            conf=self.conf,
            iou=self.iou,
            device=self.device,
            verbose=False,
            **self.extra,
        )
        if not results:
            return []
        return self._results_to_detections(results[0], frame_id)

    # --------------------------------------------------------------- private
    def _results_to_detections(self, result, frame_id: int) -> List[Dict]:
        if result.boxes is None or len(result.boxes) == 0:
            return []
        boxes_xywh = result.boxes.xywh.cpu().numpy()    # (cx, cy, w, h)
        confs = result.boxes.conf.cpu().numpy()
        coco_cls = result.boxes.cls.cpu().numpy().astype(int)

        detections: List[Dict] = []
        for (cx, cy, w, h), conf, coco_id in zip(boxes_xywh, confs, coco_cls):
            ours = self.coco_to_ours.get(int(coco_id), -1)
            if ours < 0:
                continue
            x = float(cx - w / 2.0)
            y = float(cy - h / 2.0)
            detections.append(
                {
                    "frame_id": int(frame_id),
                    "class_id": int(ours),
                    "confidence": float(conf),
                    "bbox": [x, y, float(w), float(h)],
                }
            )
        return detections
