"""Count risk labels in sequence folders.

Usage:
  python -m tools.count_sequence_labels --config configs/config.yaml --split all
  python -m tools.count_sequence_labels --config configs/config.yaml --split train
  python -m tools.count_sequence_labels --config configs/config.yaml --split all --include-ranged
"""
from __future__ import annotations

import argparse
import json
import math
from collections import Counter
from pathlib import Path
from typing import Iterable, List

from utils.config import load_config


LABEL_NAMES = {
    0: "safe",
    1: "slight",
    2: "moderate",
    3: "high",
}


def split_dirs(seq_root: Path, split: str, include_ranged: bool) -> List[Path]:
    if not include_ranged:
        return [seq_root / split]

    # Includes folders like "train 0-500" or "test(601-700)" along with "train".
    return sorted(
        p for p in seq_root.iterdir()
        if p.is_dir() and p.name.startswith(split)
    )


def sequence_dirs(split_roots: Iterable[Path]) -> Iterable[Path]:
    for root in split_roots:
        if not root.exists():
            continue
        for path in sorted(root.iterdir()):
            if path.is_dir():
                yield path


def has_valid_ttc(value: object) -> bool:
    if value is None:
        return False
    try:
        return not math.isnan(float(value))
    except (TypeError, ValueError):
        return False


def count_split(seq_root: Path, split: str, include_ranged: bool) -> Counter:
    counts: Counter = Counter()
    roots = split_dirs(seq_root, split, include_ranged)
    counts["split_folders"] = sum(1 for p in roots if p.exists())

    for seq_dir in sequence_dirs(roots):
        counts["total_sequences"] += 1
        label_path = seq_dir / "label.json"
        if not label_path.exists():
            counts["missing_label"] += 1
            continue

        try:
            with label_path.open("r", encoding="utf-8") as fp:
                payload = json.load(fp)
            risk_level = int(payload.get("risk_level"))
        except (json.JSONDecodeError, TypeError, ValueError):
            counts["invalid_label"] += 1
            continue

        if risk_level not in LABEL_NAMES:
            counts["invalid_label"] += 1
            continue

        counts[risk_level] += 1
        if risk_level > 0:
            counts["risky"] += 1
        if has_valid_ttc(payload.get("ttc_estimate", payload.get("ttc"))):
            counts["ttc_present"] += 1
        else:
            counts["ttc_missing_or_nan"] += 1

    return counts


def print_counts(split: str, counts: Counter) -> None:
    print(f"\n{split}")
    print("-" * len(split))
    print(f"split folders scanned : {counts['split_folders']}")
    print(f"total sequences       : {counts['total_sequences']}")
    for label_id, label_name in LABEL_NAMES.items():
        print(f"{label_id} {label_name:<8}        : {counts[label_id]}")
    print(f"risky total           : {counts['risky']}")
    print(f"ttc present           : {counts['ttc_present']}")
    print(f"ttc missing/NaN       : {counts['ttc_missing_or_nan']}")
    print(f"missing label.json    : {counts['missing_label']}")
    print(f"invalid label.json    : {counts['invalid_label']}")


def main() -> None:
    parser = argparse.ArgumentParser(description="Count risk labels in sequence folders")
    parser.add_argument("--config", default="configs/config.yaml")
    parser.add_argument("--split", choices=["train", "val", "test", "all"], default="all")
    parser.add_argument(
        "--include-ranged",
        action="store_true",
        help="Also scan split folders whose names start with train/val/test.",
    )
    args = parser.parse_args()

    cfg = load_config(args.config)
    seq_root = Path(cfg.paths.sequences)
    splits = ["train", "val", "test"] if args.split == "all" else [args.split]

    grand_total: Counter = Counter()
    for split in splits:
        counts = count_split(seq_root, split, args.include_ranged)
        print_counts(split, counts)
        grand_total.update(counts)

    if len(splits) > 1:
        print_counts("all", grand_total)


if __name__ == "__main__":
    main()
