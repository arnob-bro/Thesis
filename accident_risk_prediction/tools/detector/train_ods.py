"""Step 2 - train YOLO26m on the ODS_New split, then evaluate it on the test split.

    python -m tools.detector.train_ods                 # full training (hours)
    python -m tools.detector.train_ods --resume        # continue after a crash / restart
    python -m tools.detector.train_ods --quick         # 1-epoch check on 5% of the data (minutes)

Settings live in configs/detector_ods.yaml (`train.args` goes straight to
Ultralytics). Results: runs/detector/<name>/ with weights/best.pt, the
training curves and, afterwards, the evaluation report in eval_test/.
"""
from __future__ import annotations

import argparse
from pathlib import Path

from ultralytics import YOLO

from utils.config import load_config, to_plain_dict


def main() -> None:
    p = argparse.ArgumentParser(description="Train YOLO26m on ODS_New")
    p.add_argument("--config", default="configs/detector_ods.yaml")
    p.add_argument("--name", help="Run name (default: train.name in the config)")
    p.add_argument("--weights", help="Start from these weights instead of train.base_weights")
    p.add_argument("--resume", action="store_true", help="Continue the interrupted run <name>")
    p.add_argument("--quick", action="store_true", help="Smoke test: 1 epoch on 5%% of the images")
    p.add_argument("--no-eval", action="store_true", help="Skip the evaluation after training")
    args = p.parse_args()
    cfg = load_config(args.config)
    t = cfg.train
    data = Path(cfg.split.out).resolve() / "data.yaml"
    if not data.exists():
        raise SystemExit(f"{data} not found. Run first: python -m tools.detector.prepare_ods")
    project = Path(t.project).resolve()
    name = args.name or (f"{t.name}_quick" if args.quick else t.name)

    if args.resume:
        last = project / name / "weights" / "last.pt"
        if not last.exists():
            raise SystemExit(f"Nothing to resume: {last} not found")
        model = YOLO(str(last))
        model.train(resume=True)
    else:
        if (project / name / "weights" / "best.pt").exists() and not args.quick:
            raise SystemExit(f"{project / name} already has a trained model. Use --resume to continue it "
                             f"or --name <new name> for a new run.")
        train_args = to_plain_dict(t.args)
        if args.quick:
            train_args.update(epochs=1, fraction=0.05, plots=False)
        model = YOLO(args.weights or t.base_weights)
        model.train(data=str(data), project=str(project), name=name, exist_ok=True, **train_args)

    save_dir = Path(model.trainer.save_dir)
    best = save_dir / "weights" / "best.pt"
    print(f"\nBest weights: {best}")
    if args.no_eval:
        print(f"Evaluate later: python -m tools.detector.evaluate_detector --weights {best.as_posix()}")
        return
    # release the trainer's GPU memory before evaluating
    import gc

    import torch
    del model
    gc.collect()
    torch.cuda.empty_cache()
    from tools.detector.evaluate_detector import evaluate
    evaluate(cfg, best, split=cfg.eval.split, quick=args.quick)


if __name__ == "__main__":
    main()
