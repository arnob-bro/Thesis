"""Train one model.

    python train.py --config configs/config.yaml --fold 0
    python train.py --config configs/config.yaml --fold 0 --seed 1 --name no_ego --set ablation.use_ego=false

--fold k   test = fold k, validation = fold k+1, train = the rest (folds are per video).
           Without --fold the `split` column of the manifest is used.
--set      override any config value for this run, e.g. heads.type=softmax
Results go to runs/<name>/fold<k>_seed<s>/ .
"""
from __future__ import annotations

import argparse

from pipelines.training_pipeline import run_training
from utils.config import apply_overrides, load_config


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Train the accident-risk model")
    p.add_argument("--config", required=True)
    p.add_argument("--fold", type=int, default=None)
    p.add_argument("--seed", type=int, default=None)
    p.add_argument("--name", default="default", help="Experiment name (folder under runs/)")
    p.add_argument("--set", nargs="*", default=[], metavar="KEY=VALUE")
    return p.parse_args()


def main() -> None:
    args = parse_args()
    cfg = apply_overrides(load_config(args.config), args.set)
    run_training(cfg, fold=args.fold, seed=args.seed, name=args.name)


if __name__ == "__main__":
    main()
