"""Evaluation entrypoint.

Usage:
    python evaluate.py --config configs/config.yaml --checkpoint checkpoints/best.pt
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

from pipelines.training_pipeline import Trainer, make_loaders
from utils.config import load_config
from utils.logger import get_logger
from utils.seed import set_seed
from utils.visualization import plot_confusion_matrix


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Evaluate accident risk model")
    parser.add_argument("--config", type=str, required=True)
    parser.add_argument("--checkpoint", type=str, required=True)
    parser.add_argument(
        "--split", choices=["val", "test"], default="test",
        help="Which split to evaluate on",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    cfg = load_config(args.config)
    set_seed(int(cfg.seed))

    logger = get_logger("evaluate", log_dir=cfg.paths.logs)

    trainer = Trainer(cfg)
    trainer.load_checkpoint(args.checkpoint)

    loaders = make_loaders(cfg)
    if args.split not in loaders:
        raise RuntimeError(f"Split '{args.split}' not available.")

    metrics = trainer.evaluate(loaders[args.split])

    out_dir = Path(cfg.paths.logs)
    out_dir.mkdir(parents=True, exist_ok=True)

    metrics_path = out_dir / f"metrics_{args.split}.json"
    with metrics_path.open("w", encoding="utf-8") as fp:
        json.dump(metrics, fp, indent=2, default=str)

    cm = metrics.get("confusion_matrix")
    if cm:
        plot_confusion_matrix(cm, out_dir / f"confusion_{args.split}.png")

    logger.info(
        "[%s] accuracy=%.4f f1=%.4f auc=%s tta=%s early=%s",
        args.split,
        metrics["accuracy"], metrics["f1"],
        metrics["roc_auc"], metrics["time_to_accident"],
        metrics["early_risk_detection"],
    )
    logger.info("Wrote %s", metrics_path)


if __name__ == "__main__":
    main()
