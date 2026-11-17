"""Distribute ego_state data from a central master JSON to per-sequence folders.

Master JSON format:
    {
      "sequence_id_1": {
        "speed": [v1, v2, ...],
        "rpm_estimate": [r1, r2, ...],
        "brake": [b1, b2, ...],
        "acceleration": [a1, a2, ...]
      },
      "sequence_id_2": { ... },
      ...
    }

Usage:
    python -m tools.export_ego_states_to_sequences --config configs/config.yaml --master data/ego_states_master.json --split train
    python -m tools.export_ego_states_to_sequences --config configs/config.yaml --master data/ego_states_master.json --split all --dry-run
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any, Dict

from utils.config import load_config
from utils.logger import get_logger

LOGGER = get_logger("export_ego_states")


def load_master(path: Path) -> Dict[str, Dict[str, Any]]:
    if not path.exists():
        LOGGER.error("Master file not found: %s", path)
        return {}
    with path.open("r", encoding="utf-8") as fp:
        return json.load(fp)


def export_ego_states(master: Dict[str, Dict[str, Any]], seq_dir: Path, seq_id: str, overwrite: bool = False, dry_run: bool = False) -> bool:
    out = seq_dir / "ego_state.json"
    if out.exists() and not overwrite:
        return False

    if seq_id not in master:
        LOGGER.warning("Sequence %s not in master, skipping", seq_id)
        return False

    ego_data = master[seq_id]
    if dry_run:
        LOGGER.info("DRY: would write ego_state for %s -> %s", seq_id, out)
        return True

    out.parent.mkdir(parents=True, exist_ok=True)
    with out.open("w", encoding="utf-8") as fp:
        json.dump(ego_data, fp, indent=2)
    LOGGER.info("Exported ego_state for %s -> %s", seq_id, out)
    return True


def main() -> None:
    parser = argparse.ArgumentParser(description="Export ego_state from master JSON to sequences")
    parser.add_argument("--config", type=str, required=True)
    parser.add_argument("--master", type=str, required=True, help="Path to master ego_states JSON")
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
            if export_ego_states(master, p, seq_id, overwrite=args.overwrite, dry_run=args.dry_run):
                total += 1

    LOGGER.info("Done. Exported ego_state for %d sequences", total)


if __name__ == "__main__":
    main()
