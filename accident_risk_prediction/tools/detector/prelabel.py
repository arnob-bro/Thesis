"""Step 2 - pre-fill labels so you only correct them instead of drawing every box.

    python -m tools.detector.prelabel

Runs the current COCO detector on every sampled frame and writes YOLO-format
labels next to them:

    data/detector/own/labels/<video_id>/<image>.txt     class cx cy w h (0-1)
    data/detector/own/labels/classes.txt                class names, in id order

Car / motorcycle / bicycle / pedestrian / bus / truck boxes are filled in.
Boxes that look like your own handlebars or arms are filled in as
`ego_vehicle`. What is left for you: fix wrong boxes, change CNGs and
easy-bikes (usually pre-labelled car or bus) to their real class, and draw
the rickshaws.

Images that already have a label file are skipped, so your corrections are
never overwritten.
"""
from __future__ import annotations

import argparse
from pathlib import Path

import cv2
import numpy as np
from ultralytics import YOLO

from features.graph_builder import is_ego_detection
from utils.config import load_config


def main() -> None:
    p = argparse.ArgumentParser(description="Pre-label sampled frames with the COCO detector")
    p.add_argument("--config", default="configs/detector.yaml")
    p.add_argument("--main-config", default="configs/config.yaml", help="Provides the ego filter settings")
    args = p.parse_args()

    cfg = load_config(args.config)
    main_cfg = load_config(args.main_config)
    classes = list(cfg.classes)
    coco_map = {int(k): classes.index(v) for k, v in cfg.prelabel.coco_to_class.items()}
    # the ego filter works on the main pipeline's class ids
    pipeline_id = {name: i for i, name in enumerate(main_cfg.yolo.classes)}
    ego_id = classes.index("ego_vehicle") if cfg.prelabel.add_ego_vehicle and "ego_vehicle" in classes else None

    img_root, lbl_root = Path(cfg.paths.own_images), Path(cfg.paths.own_labels)
    lbl_root.mkdir(parents=True, exist_ok=True)
    (lbl_root / "classes.txt").write_text("\n".join(classes) + "\n", encoding="utf-8")

    model = YOLO(cfg.prelabel.weights)
    done = n_boxes = n_ego = 0
    for img_path in sorted(img_root.rglob("*.jpg")):
        out = lbl_root / img_path.relative_to(img_root).with_suffix(".txt")
        if out.exists():
            continue
        img = cv2.imdecode(np.frombuffer(img_path.read_bytes(), np.uint8), cv2.IMREAD_COLOR)
        H, W = img.shape[:2]
        res = model.predict(img, conf=float(cfg.prelabel.conf), verbose=False)[0]
        lines, ego_boxes = [], []
        for (cx, cy, w, h), c in zip(res.boxes.xywh.cpu().numpy(), res.boxes.cls.cpu().numpy().astype(int)):
            if c not in coco_map:
                continue
            cls = coco_map[c]
            det = {"class_id": pipeline_id.get(classes[cls], -1), "bbox": [cx - w / 2, cy - h / 2, w, h]}
            if ego_id is not None and is_ego_detection(det, main_cfg, (W, H)):
                ego_boxes.append((cx - w / 2, cy - h / 2, cx + w / 2, cy + h / 2))
                continue
            lines.append(f"{cls} {cx / W:.6f} {cy / H:.6f} {w / W:.6f} {h / H:.6f}")
        if ego_boxes:   # bike + rider's arms often come as 2-3 overlapping boxes: keep one union box
            x1, y1 = min(b[0] for b in ego_boxes), min(b[1] for b in ego_boxes)
            x2, y2 = max(b[2] for b in ego_boxes), max(b[3] for b in ego_boxes)
            lines.insert(0, f"{ego_id} {(x1 + x2) / 2 / W:.6f} {(y1 + y2) / 2 / H:.6f} "
                            f"{(x2 - x1) / W:.6f} {(y2 - y1) / H:.6f}")
            n_ego += 1
        out.parent.mkdir(parents=True, exist_ok=True)
        out.write_text("\n".join(lines) + ("\n" if lines else ""), encoding="utf-8")
        done += 1
        n_boxes += len(lines)
    print(f"pre-labelled {done} images, {n_boxes} boxes ({n_ego} as ego_vehicle) -> {lbl_root}")


if __name__ == "__main__":
    main()
