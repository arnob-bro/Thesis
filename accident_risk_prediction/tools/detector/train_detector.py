"""Step 4 - fine-tune the detector and report per-class accuracy.

    python -m tools.detector.train_detector

Fine-tunes `train.base_weights` (downloaded by Ultralytics on first use) on
data/detector/merged and prints the per-class mAP on your held-out helmet
videos, plus the lines to paste into configs/config.yaml to use the new model.
"""
from __future__ import annotations

import argparse
from pathlib import Path

from ultralytics import YOLO

from utils.config import load_config


def main() -> None:
    p = argparse.ArgumentParser(description="Fine-tune the Bangladesh traffic detector")
    p.add_argument("--config", default="configs/detector.yaml")
    p.add_argument("--epochs", type=int, help="Override epochs (e.g. 1 for a quick test)")
    args = p.parse_args()
    cfg = load_config(args.config)
    t = cfg.train
    data = Path(cfg.paths.merged) / "data.yaml"
    if not data.exists():
        raise SystemExit(f"{data} not found. Run: python -m tools.detector.merge_datasets")

    model = YOLO(t.base_weights)
    model.train(data=str(data), imgsz=int(t.imgsz), epochs=int(args.epochs or t.epochs),
                patience=int(t.patience), batch=int(t.batch), workers=int(t.get("workers", 2)),
                degrees=float(t.degrees),
                project=str(Path(t.project).resolve()), name=str(t.name), exist_ok=True)
    best = Path(t.project).resolve() / t.name / "weights" / "best.pt"
    metrics = YOLO(str(best)).val(data=str(data), imgsz=int(t.imgsz), workers=int(t.get("workers", 2)), verbose=False,
                                  project=str(Path(t.project).resolve()), name=f"{t.name}_val", exist_ok=True)

    classes = list(cfg.classes)
    print(f"\n{'class':<14}{'mAP50':>8}{'mAP50-95':>10}")
    for i, cls_idx in enumerate(metrics.box.ap_class_index):
        print(f"{classes[int(cls_idx)]:<14}{metrics.box.ap50[i]:>8.3f}{metrics.box.ap[i]:>10.3f}")
    print(f"{'all':<14}{metrics.box.map50:>8.3f}{metrics.box.map:>10.3f}")

    keep = [c for c in classes if c != "ego_vehicle"]
    print("\nTo use it, set this in configs/config.yaml, then rebuild with")
    print("  python -m pipelines.build_video_cache --config configs/config.yaml --videos data/raw_videos "
          "--steps detect track graphs --force\n")
    print("yolo:")
    print(f"  weights: {best.as_posix()}")
    print(f"  classes: [{', '.join(keep)}]")
    print("  class_map: {" + ", ".join(f"{classes.index(c)}: {i}" for i, c in enumerate(keep)) + "}   # ego_vehicle is dropped")
    print("ego_filter:\n  enabled: false   # the detector now recognises your own bike itself")


if __name__ == "__main__":
    main()
