from __future__ import annotations

import argparse
import json
from collections import defaultdict
from pathlib import Path
from typing import Dict, List

import cv2

from utils.visualization import draw_detections

CLASS_NAMES = {
    0: "car",
    1: "motorcycle",
    2: "bicycle",
    3: "pedestrian",
    4: "bus",
    5: "truck",
    6: "rickshaw",
}


def load_json(path: Path):
    with path.open("r", encoding="utf-8") as fp:
        return json.load(fp)


def frame_id_from_path(frame_path: Path) -> int:
    stem = frame_path.stem
    if stem.isdigit():
        return int(stem)

    parts = stem.split("_")
    if parts and parts[-1].isdigit():
        return int(parts[-1])

    raise ValueError(f"Unable to parse frame_id from file name: {frame_path.name}")


def build_frame_map(frames_dir: Path, metadata: dict) -> Dict[int, Path]:
    if "frame_names" in metadata and "frame_ids" in metadata:
        frame_map: Dict[int, Path] = {}
        for fid, name in zip(metadata["frame_ids"], metadata["frame_names"]):
            frame_map[int(fid)] = frames_dir / name
        return frame_map

    frame_map: Dict[int, List[Path]] = defaultdict(list)
    for fp in frames_dir.glob("*.jpg"):
        try:
            fid = frame_id_from_path(fp)
            frame_map[fid].append(fp)
        except ValueError:
            continue

    resolved: Dict[int, Path] = {}
    for fid, candidates in frame_map.items():
        resolved[fid] = sorted(candidates, key=lambda p: p.name)[0]
    return resolved


def group_by_frame(detections):
    grouped = defaultdict(list)
    for det in detections:
        grouped[int(det["frame_id"])].append(det)
    return grouped


def visualize_sequence(split: str, sequence: str | None = None, output: str | None = None) -> None:
    seq_root = Path(f"data/sequences/{split}")
    if not seq_root.exists():
        raise FileNotFoundError(f"Split directory missing: {seq_root}")

    sequences = [Path(f"data/sequences/{split}/{sequence}")] if sequence else sorted(
        p for p in seq_root.iterdir() if p.is_dir()
    )

    if sequence and not sequences[0].exists():
        raise FileNotFoundError(f"Sequence directory missing: {sequences[0]}")

    for seq_dir in sequences:
        det_path = seq_dir / "detections.json"
        frames_dir = seq_dir / "frames"
        meta_path = seq_dir / "metadata.json"

        if not det_path.exists():
            print(f"Detections file missing: {det_path}, skipping")
            continue
        if not frames_dir.exists():
            print(f"Frames directory missing: {frames_dir}, skipping")
            continue

        metadata = load_json(meta_path) if meta_path.exists() else {}
        frame_map = build_frame_map(frames_dir, metadata)
        detections = load_json(det_path)
        grouped = group_by_frame(detections)

        output_dir = Path(output) if output else seq_dir / "qa_detections"
        output_dir.mkdir(parents=True, exist_ok=True)

        total = 0
        for frame_id, frame_dets in sorted(grouped.items()):
            frame_path = frame_map.get(frame_id)
            if frame_path is None or not frame_path.exists():
                print(f"Missing sequence frame for frame_id {frame_id}: {seq_dir}")
                continue

            image = cv2.imread(str(frame_path))
            if image is None:
                print(f"Unable to read frame image: {frame_path}")
                continue

            annotated = draw_detections(
                image=image,
                detections=frame_dets,
                class_names=CLASS_NAMES,
            )

            save_path = output_dir / frame_path.name
            cv2.imwrite(str(save_path), annotated)
            total += 1

        print(f"Saved QA frames to: {output_dir}")
        print(f"Frames visualized for {seq_dir.name}: {total}")


def main() -> None:
    parser = argparse.ArgumentParser(description="Visualize per-sequence detections")
    parser.add_argument("--split", required=True, choices=["train", "val", "test"])
    parser.add_argument("--sequence", required=False, help="Optional sequence name; if omitted, all sequences are processed")
    parser.add_argument("--output", required=False, help="Optional output directory")
    args = parser.parse_args()
    visualize_sequence(split=args.split, sequence=args.sequence, output=args.output)


if __name__ == "__main__":
    main()
