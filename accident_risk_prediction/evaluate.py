"""Re-evaluate a finished run's best checkpoint.

    python evaluate.py --run runs/default/fold0_seed42
    python evaluate.py --run runs/default/fold0_seed42 --split val

Uses the config saved inside the run folder, so the model and the fold match
the training run exactly. Writes metrics_<split>.json, predictions_<split>.csv,
confusion_<split>.png and anticipation_<split>.json into the run folder.
"""
from __future__ import annotations

import argparse
import re
from pathlib import Path

from pipelines.training_pipeline import Trainer, make_loaders, write_test_artifacts
from utils.config import load_config
from utils.manifest_dataset import VideoStore, read_manifest, split_rows


def main() -> None:
    p = argparse.ArgumentParser(description="Evaluate a trained run")
    p.add_argument("--run", required=True, help="Run folder containing best.pt and config_used.yaml")
    p.add_argument("--split", choices=["val", "test"], default="test")
    args = p.parse_args()

    run_dir = Path(args.run)
    cfg = load_config(run_dir / "config_used.yaml")
    m = re.search(r"fold(\d+|X)_seed", run_dir.name)
    fold = int(m.group(1)) if m and m.group(1) != "X" else None

    parts = split_rows(read_manifest(cfg.paths.manifest), cfg, fold)
    store = VideoStore(cfg)
    loaders = make_loaders(cfg, {args.split: parts[args.split]}, store)
    trainer = Trainer(cfg)
    trainer.load(run_dir / "best.pt")
    result = write_test_artifacts(trainer, cfg, loaders, parts, store, run_dir, tag=args.split)
    if not result:
        raise SystemExit(f"Split '{args.split}' has no sequences.")
    print(f"[{args.split}] n={result['n']} accuracy={result['accuracy']:.3f} macro-F1={result['f1']:.3f} "
          f"AUC={result['roc_auc']:.3f} TTC-MAE={result['ttc_mae_s']:.2f}s")
    ant = result.get("anticipation")
    if ant:
        a = ant["at_threshold"]
        print(f"anticipation: AP={ant['average_precision']:.3f} event-recall={a['event_recall']:.2f} "
              f"mean-TTA={a['mean_tta_s']:.2f}s false-alarms/min={a['false_alarms_per_min']:.2f}")


if __name__ == "__main__":
    main()
