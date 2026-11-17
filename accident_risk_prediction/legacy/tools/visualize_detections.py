from __future__ import annotations

import argparse
import json
from collections import defaultdict
from pathlib import Path

import cv2

# CHANGE THIS IMPORT TO MATCH YOUR PROJECT
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
    with open(path, "r") as f:
        return json.load(f)


def group_by_frame(detections):
    grouped = defaultdict(list)

    for det in detections:
        grouped[det["frame_id"]].append(det)

    return grouped


def visualize_video(split: str, video: str):

    detections_path = Path(
        f"data/annotations/detections/{split}/{video}/detections.json"
    )

    frame_dir = Path(
        f"data/extracted_frames/{split}/{video}"
    )

    output_dir = Path(
        f"data/qa/detections/{split}/{video}"
    )

    output_dir.mkdir(parents=True, exist_ok=True)

    detections = load_json(detections_path)

    grouped = group_by_frame(detections)

    total = len(grouped)

    for frame_id, frame_dets in grouped.items():

        frame_path = frame_dir / f"frame_{frame_id:06d}.jpg"
        if not frame_path.exists():
            frame_path = frame_dir / f"frame_{video}_{frame_id:06d}.jpg"

        if not frame_path.exists():
            print(f"Missing frame: {frame_path}")
            continue

        image = cv2.imread(str(frame_path))

        annotated = draw_detections(
            image=image,
            detections=frame_dets,
            class_names=CLASS_NAMES,
        )

        save_path = output_dir / frame_path.name

        cv2.imwrite(str(save_path), annotated)

    print(f"\nDone.")
    print(f"Saved QA frames to:")
    print(output_dir)
    print(f"Frames visualized: {total}")


def main():

    parser = argparse.ArgumentParser()

    parser.add_argument(
        "--split",
        required=True,
    )

    parser.add_argument(
        "--video",
        required=True,
    )

    args = parser.parse_args()

    visualize_video(
        split=args.split,
        video=args.video,
    )


if __name__ == "__main__":
    main()