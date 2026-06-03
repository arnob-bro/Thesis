"""Stage 4b: cross-check ego-state files against built sequences.

The graph pipeline already copies ego state into each sequence folder. This
script audits that every sequence has an ego_state.json with the expected
length and emits a CSV summary at `data/metadata/ego_state_audit_<split>.csv`.
"""
from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path
from typing import List

from utils.config import Config, load_config
from utils.logger import get_logger

LOGGER = get_logger("ego_state_pipeline")


def audit_sequences(cfg: Config, split: str) -> List[dict]:
    seq_root = Path(cfg.paths.sequences) / split
    if not seq_root.exists():
        raise FileNotFoundError(f"Sequence root missing: {seq_root}")

    expected = int(cfg.sequence.length)
    feature_keys = list(cfg.ego_encoder.input_features)

    rows: List[dict] = []
    for seq_dir in sorted(seq_root.iterdir()):
        if not seq_dir.is_dir():
            continue
        ego_file = seq_dir / "ego_state.json"
        if not ego_file.exists():
            rows.append({"sequence_id": seq_dir.name, "status": "missing",
                         "length": 0, "missing_keys": ",".join(feature_keys)})
            continue
        with ego_file.open("r", encoding="utf-8") as fp:
            payload = json.load(fp)
        missing = [k for k in feature_keys if k not in payload]
        ref_len = len(payload.get(feature_keys[0], [])) if feature_keys else 0
        status = "ok"
        if missing or ref_len != expected:
            status = "padded"
        rows.append({
            "sequence_id": seq_dir.name,
            "status": status,
            "length": ref_len,
            "missing_keys": ",".join(missing),
        })
    return rows


def run(cfg: Config, split: str) -> None:
    rows = audit_sequences(cfg, split)
    out_dir = Path(cfg.paths.metadata)
    out_dir.mkdir(parents=True, exist_ok=True)
    out_path = out_dir / f"ego_state_audit_{split}.csv"
    with out_path.open("w", encoding="utf-8", newline="") as fp:
        writer = csv.DictWriter(
            fp, fieldnames=["sequence_id", "status", "length", "missing_keys"]
        )
        writer.writeheader()
        writer.writerows(rows)
    LOGGER.info("Audited %d sequences -> %s", len(rows), out_path)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Stage 4b — ego-state audit")
    parser.add_argument("--config", type=str, required=True)
    parser.add_argument("--split", choices=["train", "val", "test"], required=True)
    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()
    run(load_config(args.config), args.split)
