"""Trainer + evaluator used by `train.py` and `evaluate.py`.

Implements the multimodal training loop described in
`arnob/project_documentation.md`:

    Forward pass: ViT (frozen) + GNN + Temporal + Ego  ->  Cross-Attn Fusion
    Loss:         CrossEntropy(class) + λ * SmoothL1(TTC)
"""
from __future__ import annotations

import json
import time
from dataclasses import asdict
from pathlib import Path
from typing import Dict, List, Optional

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader

from models.full_model import AccidentRiskModel
from utils.config import Config, save_config
from utils.dataset import SequenceDataset, collate_sequences
from utils.logger import get_logger
from utils.metrics import compute_all_metrics
from utils.visualization import plot_confusion_matrix

LOGGER = get_logger("training_pipeline")


def _move_batch_to_device(batch: Dict, device: torch.device) -> Dict:
    moved = dict(batch)
    moved["frames"] = batch["frames"].to(device, non_blocking=True)
    moved["ego_state"] = batch["ego_state"].to(device, non_blocking=True)
    moved["risk_level"] = batch["risk_level"].to(device, non_blocking=True)
    moved["ttc"] = batch["ttc"].to(device, non_blocking=True)
    moved["ttc_valid"] = batch["ttc_valid"].to(device, non_blocking=True)
    moved["graphs_per_t"] = [g.to(device) for g in batch["graphs_per_t"]]
    return moved


def make_loaders(cfg: Config) -> Dict[str, DataLoader]:
    loaders: Dict[str, DataLoader] = {}
    for split in ("train", "val", "test"):
        try:
            ds = SequenceDataset(cfg, split)
        except (FileNotFoundError, RuntimeError) as exc:
            LOGGER.warning("Split '%s' unavailable: %s", split, exc)
            continue
        loaders[split] = DataLoader(
            ds,
            batch_size=cfg.training.batch_size,
            shuffle=(split == "train"),
            num_workers=cfg.training.num_workers,
            pin_memory=cfg.training.pin_memory,
            collate_fn=collate_sequences,
            drop_last=(split == "train"),
        )
        LOGGER.info("Loader[%s]: %d sequences", split, len(ds))
    return loaders


# ---------------------------------------------------------------------------
# Loss
# ---------------------------------------------------------------------------

class MultimodalLoss(nn.Module):
    """L = CrossEntropy(cls) + lambda * SmoothL1(TTC)."""

    def __init__(self, lambda_ttc: float = 0.5, class_weights: Optional[List[float]] = None) -> None:
        super().__init__()
        weight = (
            torch.tensor(class_weights, dtype=torch.float32) if class_weights else None
        )
        self.cls = nn.CrossEntropyLoss(weight=weight)
        self.ttc = nn.SmoothL1Loss(reduction="none")
        self.lambda_ttc = float(lambda_ttc)

    def forward(
        self,
        logits: torch.Tensor,
        risk: torch.Tensor,
        pred_ttc: torch.Tensor,
        true_ttc: torch.Tensor,
        ttc_valid: torch.Tensor,
    ) -> Dict[str, torch.Tensor]:
        cls_loss = self.cls(logits, risk)
        ttc_per = self.ttc(pred_ttc, true_ttc) * ttc_valid
        denom = ttc_valid.sum().clamp_min(1.0)
        ttc_loss = ttc_per.sum() / denom
        total = cls_loss + self.lambda_ttc * ttc_loss
        return {"loss": total, "cls_loss": cls_loss, "ttc_loss": ttc_loss}


# ---------------------------------------------------------------------------
# Trainer
# ---------------------------------------------------------------------------

class Trainer:
    def __init__(self, cfg: Config) -> None:
        self.cfg = cfg
        self.device = torch.device(
            cfg.training.device if torch.cuda.is_available() else "cpu"
        )
        LOGGER.info("Using device: %s", self.device)

        self.model = AccidentRiskModel(cfg).to(self.device)
        self.criterion = MultimodalLoss(
            lambda_ttc=cfg.loss.lambda_ttc,
            class_weights=cfg.loss.class_weights,
        ).to(self.device)

        trainable = [p for p in self.model.parameters() if p.requires_grad]
        n_trainable = sum(p.numel() for p in trainable)
        n_total = sum(p.numel() for p in self.model.parameters())
        LOGGER.info("Trainable params: %d / %d", n_trainable, n_total)

        self.optimizer = torch.optim.AdamW(
            trainable,
            lr=cfg.training.learning_rate,
            weight_decay=cfg.training.weight_decay,
        )
        self.scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
            self.optimizer, T_max=cfg.training.epochs
        )

        self.ckpt_dir = Path(cfg.paths.checkpoints)
        self.ckpt_dir.mkdir(parents=True, exist_ok=True)
        save_config(cfg, self.ckpt_dir / "config_used.yaml")

    # ---------------------------------------------------------------- save
    def save_checkpoint(self, path: Path, epoch: int, best_val: float) -> None:
        torch.save(
            {
                "epoch": epoch,
                "model_state": self.model.state_dict(),
                "optimizer_state": self.optimizer.state_dict(),
                "best_val": best_val,
                "cfg": dict(self.cfg),
            },
            path,
        )

    def load_checkpoint(self, path: str | Path) -> None:
        ckpt = torch.load(path, map_location=self.device, weights_only=False)
        self.model.load_state_dict(ckpt["model_state"])
        LOGGER.info("Loaded checkpoint %s (epoch %s)", path, ckpt.get("epoch"))

    # ---------------------------------------------------------------- epoch
    def _run_epoch(self, loader: DataLoader, train: bool) -> Dict[str, float]:
        self.model.train(train)
        running = {"loss": 0.0, "cls_loss": 0.0, "ttc_loss": 0.0, "n": 0}

        for step, batch in enumerate(loader, start=1):
            batch = _move_batch_to_device(batch, self.device)
            with torch.set_grad_enabled(train):
                out = self.model(batch)
                loss_dict = self.criterion(
                    out["logits"], batch["risk_level"],
                    out["ttc"], batch["ttc"], batch["ttc_valid"],
                )
            if train:
                self.optimizer.zero_grad(set_to_none=True)
                loss_dict["loss"].backward()
                torch.nn.utils.clip_grad_norm_(
                    [p for p in self.model.parameters() if p.requires_grad],
                    self.cfg.training.grad_clip,
                )
                self.optimizer.step()

            bs = batch["risk_level"].size(0)
            running["loss"] += float(loss_dict["loss"]) * bs
            running["cls_loss"] += float(loss_dict["cls_loss"]) * bs
            running["ttc_loss"] += float(loss_dict["ttc_loss"]) * bs
            running["n"] += bs

            if train and step % self.cfg.training.log_interval == 0:
                LOGGER.info(
                    "step %d | loss %.4f (cls %.4f ttc %.4f)",
                    step, float(loss_dict["loss"]),
                    float(loss_dict["cls_loss"]), float(loss_dict["ttc_loss"]),
                )

        n = max(running["n"], 1)
        return {k: v / n for k, v in running.items() if k != "n"}

    # ---------------------------------------------------------------- eval
    @torch.no_grad()
    def evaluate(self, loader: DataLoader) -> Dict[str, object]:
        self.model.eval()
        y_true: List[int] = []
        y_pred: List[int] = []
        y_score: List[np.ndarray] = []
        pred_ttc: List[float] = []
        true_ttc: List[float] = []

        for batch in loader:
            batch = _move_batch_to_device(batch, self.device)
            out = self.model(batch)
            probs = F.softmax(out["logits"], dim=-1).cpu().numpy()
            preds = out["logits"].argmax(dim=-1).cpu().numpy()

            y_true.extend(batch["risk_level"].cpu().numpy().tolist())
            y_pred.extend(preds.tolist())
            y_score.extend(list(probs))
            pred_ttc.extend(out["ttc"].cpu().numpy().tolist())
            true_ttc.extend(
                [
                    t if v > 0.5 else float("nan")
                    for t, v in zip(
                        batch["ttc"].cpu().numpy().tolist(),
                        batch["ttc_valid"].cpu().numpy().tolist(),
                    )
                ]
            )

        result = compute_all_metrics(
            y_true=y_true,
            y_pred=y_pred,
            y_score=np.asarray(y_score) if y_score else None,
            pred_ttc=pred_ttc,
            true_ttc=true_ttc,
            num_classes=self.cfg.heads.num_classes,
            fps=float(self.cfg.frame_extraction.frame_rate),
        )
        return result.to_dict()

    # ---------------------------------------------------------------- fit
    def fit(self, loaders: Dict[str, DataLoader]) -> None:
        if "train" not in loaders:
            raise RuntimeError("Training requires a 'train' split.")

        best_val = float("inf")
        patience = self.cfg.training.early_stop_patience
        no_improve = 0

        for epoch in range(1, self.cfg.training.epochs + 1):
            t0 = time.time()
            train_metrics = self._run_epoch(loaders["train"], train=True)
            LOGGER.info(
                "Epoch %d/%d | train_loss %.4f (cls %.4f ttc %.4f) | %.1fs",
                epoch, self.cfg.training.epochs,
                train_metrics["loss"], train_metrics["cls_loss"],
                train_metrics["ttc_loss"], time.time() - t0,
            )

            val_metric_value: Optional[float] = None
            if "val" in loaders:
                val_metrics = self._run_epoch(loaders["val"], train=False)
                LOGGER.info(
                    "Epoch %d | val_loss %.4f (cls %.4f ttc %.4f)",
                    epoch, val_metrics["loss"], val_metrics["cls_loss"],
                    val_metrics["ttc_loss"],
                )
                val_metric_value = val_metrics["loss"]

            self.scheduler.step()

            # Checkpointing
            if epoch % self.cfg.training.ckpt_interval == 0:
                self.save_checkpoint(
                    self.ckpt_dir / f"epoch_{epoch:03d}.pt", epoch, best_val
                )

            if val_metric_value is not None:
                if val_metric_value < best_val - 1e-4:
                    best_val = val_metric_value
                    no_improve = 0
                    self.save_checkpoint(self.ckpt_dir / "best.pt", epoch, best_val)
                    LOGGER.info("New best val_loss=%.4f -> saved best.pt", best_val)
                else:
                    no_improve += 1
                    if no_improve >= patience:
                        LOGGER.info("Early stopping at epoch %d", epoch)
                        break

        # Final evaluation on test split (if present)
        if "test" in loaders:
            metrics = self.evaluate(loaders["test"])
            self._dump_eval_artifacts(metrics, tag="test")

    # ---------------------------------------------------------------- artifacts
    def _dump_eval_artifacts(self, metrics: Dict, tag: str) -> None:
        log_dir = Path(self.cfg.paths.logs)
        log_dir.mkdir(parents=True, exist_ok=True)
        with (log_dir / f"metrics_{tag}.json").open("w", encoding="utf-8") as fp:
            json.dump(metrics, fp, indent=2)
        cm = metrics.get("confusion_matrix")
        if cm:
            plot_confusion_matrix(cm, log_dir / f"confusion_{tag}.png")
        LOGGER.info("[%s] %s", tag, json.dumps(
            {k: v for k, v in metrics.items() if k != "confusion_matrix"},
            default=str,
        ))
