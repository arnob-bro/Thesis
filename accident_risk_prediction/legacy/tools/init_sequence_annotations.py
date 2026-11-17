"""Initialise editable `label.json` and `ego_state.json` for sequence folders.

Creates placeholder files for sequences that are missing them so users can
open and edit them later. Behaviour mirrors `write_sequence_annotations.py`
but operates on all sequence folders under `cfg.paths.sequences`.

Usage:
  python -m tools.init_sequence_annotations --config configs/config.yaml --split train

Options:
  --overwrite    Replace existing files
  --dry-run      Print actions without writing
  --open         Try to open created files with the OS default app (Windows: Start)
"""
from __future__ import annotations

import argparse
import json
import os
from pathlib import Path
from typing import List

from utils.config import load_config


def init_label(seq_dir: Path, seq_id: str, overwrite: bool = False, dry_run: bool = False) -> bool:
    out = seq_dir / "label.json"
    if out.exists() and not overwrite:
        return False
    payload = {"sequence_id": seq_id, "risk_level": 0, "ttc_estimate": float("nan")}
    if dry_run:
        print(f"DRY: would write label -> {out}")
        return True
    out.parent.mkdir(parents=True, exist_ok=True)
    with out.open("w", encoding="utf-8") as fp:
        json.dump(payload, fp, indent=2)
    print(f"Wrote label: {out}")
    return True


def init_ego_state(seq_dir: Path, feature_keys: List[str], seq_len: int, overwrite: bool = False, dry_run: bool = False) -> bool:
    out = seq_dir / "ego_state.json"
    if out.exists() and not overwrite:
        return False
    payload = {k: [0.0] * seq_len for k in feature_keys}
    if dry_run:
        print(f"DRY: would write ego_state -> {out} (keys: {feature_keys})")
        return True
    out.parent.mkdir(parents=True, exist_ok=True)
    with out.open("w", encoding="utf-8") as fp:
        json.dump(payload, fp, indent=2)
    print(f"Wrote ego_state: {out}")
    return True


def try_open(path: Path) -> None:
    try:
        if os.name == "nt":
            os.startfile(str(path))
        else:
            # cross-platform best-effort
            from subprocess import Popen

            Popen(["xdg-open" if os.name == "posix" else "open", str(path)])
    except Exception:
        pass


def main() -> None:
    parser = argparse.ArgumentParser(description="Initialise label and ego_state for sequences")
    parser.add_argument("--config", required=True)
    parser.add_argument("--split", required=True, choices=["train", "val", "test", "all"])
    parser.add_argument("--overwrite", action="store_true")
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument("--open", action="store_true", help="Open created files after writing (best-effort)")
    args = parser.parse_args()

    cfg = load_config(args.config)
    seq_root = Path(cfg.paths.sequences)
    splits = [args.split] if args.split != "all" else ["train", "val", "test"]

    ego_keys = list(cfg.ego_encoder.input_features)
    seq_len = int(cfg.sequence.length)

    total_created = 0
    for sp in splits:
        root = seq_root / sp
        if not root.exists():
            print(f"Sequence root for split missing: {root}")
            continue
        for p in sorted(root.iterdir()):
            if not p.is_dir():
                continue
            seq_id = p.name
            created_label = init_label(p, seq_id, overwrite=args.overwrite, dry_run=args.dry_run)
            created_ego = init_ego_state(p, ego_keys, seq_len, overwrite=args.overwrite, dry_run=args.dry_run)
            if created_label or created_ego:
                total_created += int(created_label) + int(created_ego)
                if args.open and not args.dry_run:
                    # open the sequence directory for convenience
                    try_open(p)

    print(f"Done. Files created or would be created: {total_created}")


if __name__ == "__main__":
    main()
