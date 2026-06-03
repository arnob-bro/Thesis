"""Distribute label data from a central master JSON to per-sequence folders.

Master JSON format:
    {
      "sequence_id_1": {
        "risk_level": 0,
        "ttc_estimate": NaN
      },
      "sequence_id_2": {
        "risk_level": 3,
        "ttc_estimate": 0.8
      },
      ...
    }

Usage:
    python -m tools.export_labels_to_sequences --config configs/config.yaml --master data/labels_master.json --split train
    python -m tools.export_labels_to_sequences --config configs/config.yaml --master data/labels_master.json --split all --dry-run
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any, Dict

from utils.config import load_config
from utils.logger import get_logger

LOGGER = get_logger("export_labels")


def load_master(path: Path) -> Dict[str, Dict[str, Any]]:
    if not path.exists():
        LOGGER.error("Master file not found: %s", path)
        return {}
    with path.open("r", encoding="utf-8") as fp:
        return json.load(fp)


def export_label(master: Dict[str, Dict[str, Any]], seq_dir: Path, seq_id: str, overwrite: bool = False, dry_run: bool = False) -> bool:
    out = seq_dir / "label.json"
    if out.exists() and not overwrite:
        return False

    if seq_id not in master:
        LOGGER.warning("Sequence %s not in master, skipping", seq_id)
        return False

    label_data = master[seq_id]
    if dry_run:
        LOGGER.info("DRY: would write label for %s -> %s", seq_id, out)
        return True

    out.parent.mkdir(parents=True, exist_ok=True)
    with out.open("w", encoding="utf-8") as fp:
        json.dump(label_data, fp, indent=2)
    LOGGER.info("Exported label for %s -> %s", seq_id, out)
    return True


def main() -> None:
    parser = argparse.ArgumentParser(description="Export labels from master JSON to sequences")
    parser.add_argument("--config", type=str, required=True)
    parser.add_argument("--master", type=str, required=True, help="Path to master labels JSON")
    parser.add_argument("--split", choices=["train", "val", "test", "all"], required=True)
    parser.add_argument("--overwrite", action="store_true")
    parser.add_argument("--dry-run", action="store_true")
    args = parser.parse_args()

    cfg = load_config(args.config)
    master = load_master(Path(args.master))
    if not master:
        return

    splits = [args.split] if args.split != "all" else ["train", "val", "test"]
    seq_root = Path(cfg.paths.sequences)

    total = 0
    for sp in splits:
        root = seq_root / sp
        if not root.exists():
            LOGGER.warning("Sequence root for split missing: %s", root)
            continue
        for p in sorted(root.iterdir()):
            if not p.is_dir():
                continue
            seq_id = p.name
            if export_label(master, p, seq_id, overwrite=args.overwrite, dry_run=args.dry_run):
                total += 1

    LOGGER.info("Done. Exported labels for %d sequences", total)


if __name__ == "__main__":
    main()
