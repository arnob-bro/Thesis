"""Run the full model and every ablation over all folds and seeds, then
collect a results table for the thesis.

    python -m tools.run_experiments --config configs/config.yaml
    python -m tools.run_experiments --config configs/config.yaml --seeds 0 1 2 --only full no_graph
    python -m tools.run_experiments --config configs/config.yaml --collect      # just rebuild the table

Runs that already have metrics_test.json are skipped, so the command can be
stopped and restarted. Output: runs/summary.csv and runs/summary.md
(mean +/- std over folds x seeds).
"""
from __future__ import annotations

import argparse
import csv
import json
from collections import defaultdict
from pathlib import Path
from typing import Dict, List

import numpy as np

from pipelines.training_pipeline import run_training
from utils.config import apply_overrides, load_config

EXPERIMENTS: Dict[str, List[str]] = {
    "full": [],
    "no_scene": ["ablation.use_vit=false"],
    "no_graph": ["ablation.use_gnn=false"],
    "no_ego_motion": ["ablation.use_ego=false"],
    "no_temporal": ["ablation.use_temporal=false"],
    "scene_only": ["ablation.use_gnn=false", "ablation.use_ego=false"],
    "graph_only": ["ablation.use_vit=false", "ablation.use_ego=false"],
    "no_rider_node": ["graph.include_ego_node=false"],
    "softmax_head": ["heads.type=softmax"],
}
# `no_rider_node` needs graphs rebuilt without the rider node, so it is not run by default.
DEFAULT = [e for e in EXPERIMENTS if e != "no_rider_node"]

COLUMNS = [
    ("accuracy", lambda m: m["accuracy"]),
    ("macro_f1", lambda m: m["f1"]),
    ("risky_recall", lambda m: m["risky_recall"]),
    ("high_recall", lambda m: m["per_class"]["high"]["recall"] if m["per_class"]["high"]["support"] else np.nan),
    ("ordinal_mae", lambda m: m["ordinal_mae"]),
    ("roc_auc", lambda m: m["roc_auc"]),
    ("ttc_mae_s", lambda m: m["ttc_mae_s"]),
    ("anticipation_ap", lambda m: m.get("anticipation", {}).get("average_precision", np.nan)),
    ("event_recall", lambda m: m.get("anticipation", {}).get("at_threshold", {}).get("event_recall", np.nan)),
    ("mean_tta_s", lambda m: m.get("anticipation", {}).get("at_threshold", {}).get("mean_tta_s", np.nan)),
    ("false_alarms_per_min", lambda m: m.get("anticipation", {}).get("at_threshold", {}).get("false_alarms_per_min", np.nan)),
]


def collect(runs_root: Path) -> None:
    table: Dict[str, List[Dict]] = defaultdict(list)
    for mfile in sorted(runs_root.glob("*/fold*_seed*/metrics_test.json")):
        m = json.loads(mfile.read_text(encoding="utf-8"))
        ant = mfile.with_name("anticipation_test.json")
        if ant.exists():
            m["anticipation"] = json.loads(ant.read_text(encoding="utf-8"))
        row = {}
        for name, fn in COLUMNS:
            try:
                v = fn(m)
                row[name] = np.nan if v is None else float(v)
            except (KeyError, TypeError):
                row[name] = np.nan
        table[mfile.parent.parent.name].append(row)
    if not table:
        print("No finished runs found.")
        return

    names = [c for c, _ in COLUMNS]
    with (runs_root / "summary.csv").open("w", newline="", encoding="utf-8") as fp:
        w = csv.writer(fp)
        w.writerow(["experiment", "runs"] + [f"{n}_{s}" for n in names for s in ("mean", "std")])
        for exp, rows in table.items():
            w.writerow([exp, len(rows)] + [f"{fn([r[n] for r in rows]):.4f}"
                                           for n in names for fn in (np.nanmean, np.nanstd)])
    lines = ["| experiment | runs | " + " | ".join(names) + " |", "|---|---|" + "---|" * len(names)]
    for exp, rows in table.items():
        cells = [f"{np.nanmean([r[n] for r in rows]):.3f} ± {np.nanstd([r[n] for r in rows]):.3f}" for n in names]
        lines.append(f"| {exp} | {len(rows)} | " + " | ".join(cells) + " |")
    (runs_root / "summary.md").write_text("\n".join(lines) + "\n", encoding="utf-8")
    print("\n".join(lines))
    print(f"\nWrote {runs_root / 'summary.csv'} and {runs_root / 'summary.md'}")


def main() -> None:
    p = argparse.ArgumentParser(description="Run model + ablations over folds and seeds")
    p.add_argument("--config", required=True)
    p.add_argument("--only", nargs="*", choices=list(EXPERIMENTS), default=DEFAULT)
    p.add_argument("--seeds", nargs="*", type=int, default=[0, 1, 2])
    p.add_argument("--folds", nargs="*", type=int, help="Default: all folds")
    p.add_argument("--collect", action="store_true", help="Only rebuild the summary table")
    args = p.parse_args()

    base = load_config(args.config)
    runs_root = Path(base.paths.runs)
    if not args.collect:
        folds = args.folds if args.folds is not None else list(range(int(base.cv.folds)))
        for exp in args.only:
            for fold in folds:
                for seed in args.seeds:
                    if (runs_root / exp / f"fold{fold}_seed{seed}" / "metrics_test.json").exists():
                        continue
                    cfg = apply_overrides(load_config(args.config), EXPERIMENTS[exp])
                    try:
                        run_training(cfg, fold=fold, seed=seed, name=exp)
                    except RuntimeError as exc:
                        print(f"{exp} fold {fold} seed {seed}: skipped ({exc})")
    collect(runs_root)


if __name__ == "__main__":
    main()
