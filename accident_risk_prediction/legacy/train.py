"""Reproducible training entrypoint.

Usage:
    python train.py --config configs/config.yaml
"""
from __future__ import annotations

import argparse
from pathlib import Path

from pipelines.training_pipeline import Trainer, make_loaders
from utils.config import load_config
from utils.logger import get_logger
from utils.seed import set_seed


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Train multimodal accident risk model")
    parser.add_argument("--config", type=str, required=True, help="Path to YAML config")
    parser.add_argument("--resume", type=str, default=None, help="Optional checkpoint")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    cfg = load_config(args.config)
    set_seed(int(cfg.seed))

    logger = get_logger("train", log_dir=cfg.paths.logs)
    logger.info("Loaded config from %s", args.config)

    trainer = Trainer(cfg)
    if args.resume:
        trainer.load_checkpoint(Path(args.resume))

    loaders = make_loaders(cfg)
    trainer.fit(loaders)
    logger.info("Training finished.")


if __name__ == "__main__":
    main()
