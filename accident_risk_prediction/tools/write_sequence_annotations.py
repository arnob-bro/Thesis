"""Write `label.json` and `ego_state.json` into sequence folders.

Usage examples:

# Write labels from CSV (columns: sequence_id,risk_level,ttc_estimate)
python -m tools.write_sequence_annotations --config configs/config.yaml --split train --labels-csv labels.csv

# Import ego CSV files from a directory (filenames: <sequence_id>.csv or .json)
python -m tools.write_sequence_annotations --config configs/config.yaml --split train --ego-dir /path/to/ego_dir

Both operations can be combined. Use `--overwrite` to replace existing files.
"""
from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path
from typing import Dict, List, Optional

from utils.config import load_config


def write_label(seq_dir: Path, seq_id: str, risk: int, ttc: Optional[float], overwrite: bool = False) -> None:
    out = seq_dir / "label.json"
    if out.exists() and not overwrite:
        print(f"Skipping label for {seq_id}: exists (use --overwrite to replace)")
        return
    payload = {"sequence_id": seq_id, "risk_level": int(risk)}
    if ttc is None or (isinstance(ttc, float) and (ttc != ttc)):
        payload["ttc_estimate"] = float("nan")
    else:
        payload["ttc_estimate"] = float(ttc)
    out.parent.mkdir(parents=True, exist_ok=True)
    with out.open("w", encoding="utf-8") as fp:
        json.dump(payload, fp, indent=2)
    print(f"Wrote label: {out}")


def read_labels_csv(path: Path) -> List[Dict[str, str]]:
    with path.open("r", encoding="utf-8") as fp:
        reader = csv.DictReader(fp)
        rows = [r for r in reader]
    return rows


def import_ego_file(src: Path, dst_seq_dir: Path, overwrite: bool = False) -> None:
    dst = dst_seq_dir / "ego_state.json"
    if dst.exists() and not overwrite:
        print(f"Skipping ego for {dst_seq_dir.name}: exists (use --overwrite)")
        return
    if src.suffix.lower() == ".json":
        dst.parent.mkdir(parents=True, exist_ok=True)
        with src.open("r", encoding="utf-8") as fp, dst.open("w", encoding="utf-8") as out:
            data = json.load(fp)
            json.dump(data, out, indent=2)
        print(f"Copied ego json -> {dst}")
        return

    # CSV -> convert columns to lists
    if src.suffix.lower() == ".csv":
        with src.open("r", encoding="utf-8") as fp:
            rdr = csv.DictReader(fp)
            cols: Dict[str, List[float]] = {}
            for row in rdr:
                for k, v in row.items():
                    try:
                        f = float(v)
                    except Exception:
                        f = 0.0
                    cols.setdefault(k, []).append(f)
        dst.parent.mkdir(parents=True, exist_ok=True)
        with dst.open("w", encoding="utf-8") as out:
            json.dump(cols, out, indent=2)
        print(f"Converted ego csv -> {dst}")
        return

    print(f"Unsupported ego file format: {src}")


def find_sequence_dir(root: Path, seq_id: str) -> Optional[Path]:
    cand = root / seq_id
    if cand.exists() and cand.is_dir():
        return cand
    # fallback: try to find directory that contains seq_id as suffix/prefix
    for p in root.iterdir():
        if not p.is_dir():
            continue
        if p.name == seq_id or seq_id in p.name:
            return p
    return None


def main() -> None:
    parser = argparse.ArgumentParser(description="Write sequence label and ego_state files")
    parser.add_argument("--config", required=True)
    parser.add_argument("--split", required=True, choices=["train", "val", "test", "all"]) 
    parser.add_argument("--labels-csv", type=str, default=None,
                        help="CSV with columns: sequence_id,risk_level,ttc_estimate")
    parser.add_argument("--ego-dir", type=str, default=None,
                        help="Directory containing per-sequence ego CSV/JSON files named <sequence_id>.csv/.json")
    parser.add_argument("--overwrite", action="store_true")
    parser.add_argument("--dry-run", action="store_true")
    args = parser.parse_args()

    cfg = load_config(args.config)
    seq_root = Path(cfg.paths.sequences)

    splits = [args.split] if args.split != "all" else ["train", "val", "test"]

    # Process labels
    if args.labels_csv:
        rows = read_labels_csv(Path(args.labels_csv))
        for r in rows:
            seq_id = r.get("sequence_id") or r.get("sequence")
            if not seq_id:
                print("Row missing sequence_id, skipping:", r)
                continue
            try:
                risk = int(r.get("risk_level", 0))
            except Exception:
                risk = 0
            ttc_raw = r.get("ttc_estimate") or r.get("ttc")
            try:
                ttc = float(ttc_raw) if ttc_raw not in (None, "", "nan") else float("nan")
            except Exception:
                ttc = float("nan")

            # find sequence directory across requested splits
            found = False
            for sp in splits:
                root = seq_root / sp
                seq_dir = find_sequence_dir(root, seq_id)
                if seq_dir:
                    found = True
                    if args.dry_run:
                        print(f"DRY: would write label for {seq_id} -> {seq_dir}/label.json")
                    else:
                        write_label(seq_dir, seq_id, risk, ttc, overwrite=args.overwrite)
                    break
            if not found:
                print(f"Sequence {seq_id} not found under splits {splits}")

    # Process ego files
    if args.ego_dir:
        ego_root = Path(args.ego_dir)
        if not ego_root.exists():
            print("Ego dir not found:", ego_root)
        else:
            for f in sorted(ego_root.iterdir()):
                if not f.is_file():
                    continue
                seq_id = f.stem
                found = False
                for sp in splits:
                    root = seq_root / sp
                    seq_dir = find_sequence_dir(root, seq_id)
                    if seq_dir:
                        found = True
                        if args.dry_run:
                            print(f"DRY: would import ego {f} -> {seq_dir}/ego_state.json")
                        else:
                            import_ego_file(f, seq_dir, overwrite=args.overwrite)
                        break
                if not found:
                    print(f"Ego file {f.name} did not match any sequence under splits {splits}")

    if not args.labels_csv and not args.ego_dir:
        print("Nothing to do. Provide --labels-csv and/or --ego-dir.")


if __name__ == "__main__":
    main()
