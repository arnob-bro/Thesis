"""Step 3 - build the training-ready dataset.

    python -m tools.detector.merge_datasets

Combines your corrected helmet frames with any public datasets listed in
configs/detector.yaml, remaps every class to the final class list, and writes

    data/detector/merged/images/{train,val}/...
    data/detector/merged/labels/{train,val}/...
    data/detector/merged/data.yaml

Whole videos of yours go to validation (never single frames), and by default
validation contains only helmet-POV frames, because that is where the
detector has to work.
"""
from __future__ import annotations

import argparse
import random
import shutil
import xml.etree.ElementTree as ET
from collections import Counter
from pathlib import Path
from typing import Dict, List, Optional

import yaml

from utils.config import load_config

IMG_EXTS = {".jpg", ".jpeg", ".png", ".bmp"}


def _yolo_lines(path: Path, id_map: Optional[Dict[int, int]]) -> List[str]:
    out = []
    if not path.exists():
        return out
    for line in path.read_text(encoding="utf-8").splitlines():
        parts = line.split()
        if len(parts) < 5:
            continue
        cls = int(float(parts[0]))
        if id_map is not None:
            if cls not in id_map:
                continue
            cls = id_map[cls]
        out.append(" ".join([str(cls)] + parts[1:5]))
    return out


def _voc_lines(path: Path, name_map: Dict[str, int]) -> List[str]:
    out = []
    if not path.exists():
        return out
    root = ET.parse(path).getroot()
    W, H = float(root.findtext("size/width")), float(root.findtext("size/height"))
    for obj in root.iter("object"):
        name = (obj.findtext("name") or "").strip().lower()
        if name not in name_map:
            continue
        b = obj.find("bndbox")
        x1, y1, x2, y2 = (float(b.findtext(k)) for k in ("xmin", "ymin", "xmax", "ymax"))
        out.append(f"{name_map[name]} {(x1 + x2) / 2 / W:.6f} {(y1 + y2) / 2 / H:.6f} "
                   f"{(x2 - x1) / W:.6f} {(y2 - y1) / H:.6f}")
    return out


def main() -> None:
    p = argparse.ArgumentParser(description="Merge own + public data into one YOLO dataset")
    p.add_argument("--config", default="configs/detector.yaml")
    args = p.parse_args()
    cfg = load_config(args.config)
    classes = list(cfg.classes)
    rng = random.Random(int(cfg.merge.seed))
    out = Path(cfg.paths.merged)
    if out.exists():
        shutil.rmtree(out)   # generated folder: always rebuilt from the sources
    counts = {"train": Counter(), "val": Counter()}
    n_img = Counter()

    def add(img: Path, lines: List[str], split: str, prefix: str) -> None:
        if not lines:
            return
        name = f"{prefix}_{img.stem}"
        (out / "images" / split).mkdir(parents=True, exist_ok=True)
        (out / "labels" / split).mkdir(parents=True, exist_ok=True)
        shutil.copy2(img, out / "images" / split / f"{name}{img.suffix.lower()}")
        (out / "labels" / split / f"{name}.txt").write_text("\n".join(lines) + "\n", encoding="utf-8")
        n_img[split] += 1
        for l in lines:
            counts[split][classes[int(l.split()[0])]] += 1

    # --- your own helmet frames: split by video ---
    own_img, own_lbl = Path(cfg.paths.own_images), Path(cfg.paths.own_labels)
    videos = sorted(d.name for d in own_img.iterdir() if d.is_dir()) if own_img.exists() else []
    rng.shuffle(videos)
    n_val = max(1, int(round(len(videos) * float(cfg.merge.val_fraction)))) if len(videos) > 1 else 0
    val_videos = set(videos[:n_val])
    for vid in videos:
        for img in sorted((own_img / vid).glob("*.jpg")):
            add(img, _yolo_lines(own_lbl / vid / f"{img.stem}.txt", None), "val" if vid in val_videos else "train", "own")

    # --- public datasets ---
    for ds in cfg.get("public_datasets") or []:
        tgt = {str(k).strip().lower(): classes.index(v) for k, v in ds["map"].items()}
        id_map = None
        if ds.get("format", "yolo") == "yolo":
            names = ds.get("names")
            if names is None:
                names = yaml.safe_load(Path(ds["data_yaml"]).read_text(encoding="utf-8"))["names"]
            if isinstance(names, dict):
                names = [names[k] for k in sorted(names)]
            id_map = {i: tgt[str(n).strip().lower()] for i, n in enumerate(names) if str(n).strip().lower() in tgt}
        img_root, lbl_root = Path(ds["images"]), Path(ds["labels"])
        for img in sorted(q for q in img_root.rglob("*") if q.suffix.lower() in IMG_EXTS):
            rel = img.relative_to(img_root)
            if id_map is not None:
                lines = _yolo_lines(lbl_root / rel.with_suffix(".txt"), id_map)
            else:
                lines = _voc_lines(lbl_root / rel.with_suffix(".xml"), tgt)
            to_val = bool(cfg.merge.public_in_val) and rng.random() < float(cfg.merge.val_fraction)
            add(img, lines, "val" if to_val else "train", ds["name"])

    if not n_img["train"]:
        raise SystemExit("No labelled training images found. Run sample_frames and prelabel first.")
    data = {"path": str(out.resolve()), "train": "images/train", "val": "images/val" if n_img["val"] else "images/train",
            "names": {i: n for i, n in enumerate(classes)}}
    (out / "data.yaml").write_text(yaml.safe_dump(data, sort_keys=False), encoding="utf-8")

    print(f"{'class':<14}{'train':>8}{'val':>8}")
    for c in classes:
        flag = "   <- few examples" if counts["train"][c] < 200 else ""
        print(f"{c:<14}{counts['train'][c]:>8}{counts['val'][c]:>8}{flag}")
    print(f"\nimages: train {n_img['train']}, val {n_img['val']} ({len(val_videos)} of your videos held out)")
    print(f"dataset -> {out / 'data.yaml'}")


if __name__ == "__main__":
    main()
