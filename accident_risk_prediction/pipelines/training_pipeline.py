"""Training, evaluation and sliding-window anticipation scoring.

One call to ``run_training`` = one run folder::

    runs/<name>/fold<k>_seed<s>/
        config_used.yaml        exact settings of this run
        history.csv             per-epoch train / validation curves
        best.pt                 checkpoint chosen on the validation set
        metrics_test.json       sequence-level metrics of best.pt on the test set
        predictions_test.csv    one row per test sequence
        confusion_test.png
        anticipation_test.json  event recall / time-to-accident / false alarms
"""
from __future__ import annotations

import csv
import json
import time
from pathlib import Path
from typing import Dict, List, Optional

import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader

from models.full_model import AccidentRiskModel
from utils.anticipation import anticipation_metrics
from utils.config import Config, save_config
from utils.events import load_events
from utils.logger import get_logger
from utils.manifest_dataset import ManifestDataset, VideoStore, collate, read_manifest, split_rows
from utils.metrics import compute_metrics
from utils.seed import set_seed
from utils.visualization import plot_confusion_matrix
from utils.windows import dense_windows

LOGGER = get_logger("training")


def to_device(batch: Dict, device: torch.device) -> Dict:
    return {k: (v.to(device) if hasattr(v, "to") else v) for k, v in batch.items()}


# ---------------------------------------------------------------------------
# Loss
# ---------------------------------------------------------------------------

class RiskLoss(nn.Module):
    """Risk loss (ordinal or softmax) + lambda * SmoothL1(time-to-collision)."""

    def __init__(self, cfg: Config, train_rows: List[Dict]) -> None:
        super().__init__()
        self.head_type = str(cfg.heads.type)
        self.n_cls = int(cfg.heads.num_classes)
        self.lambda_ttc = float(cfg.loss.lambda_ttc)
        counts = np.bincount([r["risk_level"] for r in train_rows], minlength=self.n_cls).astype(float)
        cw = cfg.loss.class_weights

        if self.head_type == "ordinal":
            pos_weight = None
            if cw == "auto":   # for each threshold k: weight of "risk > k" = negatives / positives
                pos = np.array([counts[k + 1:].sum() for k in range(self.n_cls - 1)])
                pos_weight = torch.tensor(np.clip((counts.sum() - pos) / np.maximum(pos, 1), 0.1, 10.0),
                                          dtype=torch.float32)
            self.cls = nn.BCEWithLogitsLoss(pos_weight=pos_weight)
        else:
            weight = None
            if cw == "auto":
                weight = torch.tensor(np.clip(counts.sum() / (self.n_cls * np.maximum(counts, 1)), 0.1, 10.0),
                                      dtype=torch.float32)
            elif cw:
                weight = torch.tensor(list(cw), dtype=torch.float32)
            self.cls = nn.CrossEntropyLoss(weight=weight)
        self.ttc = nn.SmoothL1Loss(reduction="none")

    def forward(self, out: Dict, batch: Dict) -> Dict[str, torch.Tensor]:
        y = batch["risk"]
        if self.head_type == "ordinal":
            ks = torch.arange(self.n_cls - 1, device=y.device)
            cls_loss = self.cls(out["logits"], (y.unsqueeze(1) > ks).float())
        else:
            cls_loss = self.cls(out["logits"], y)
        valid = batch["ttc_valid"]
        ttc_loss = (self.ttc(out["ttc"], batch["ttc"]) * valid).sum() / valid.sum().clamp_min(1.0)
        return {"loss": cls_loss + self.lambda_ttc * ttc_loss, "cls": cls_loss, "ttc": ttc_loss}


# ---------------------------------------------------------------------------
# Trainer
# ---------------------------------------------------------------------------

class Trainer:
    def __init__(self, cfg: Config, train_rows: Optional[List[Dict]] = None) -> None:
        self.cfg = cfg
        self.device = torch.device(cfg.training.device if torch.cuda.is_available() else "cpu")
        self.model = AccidentRiskModel(cfg).to(self.device)
        self.criterion = RiskLoss(cfg, train_rows or []).to(self.device)
        self.optimizer = torch.optim.AdamW(self.model.parameters(), lr=float(cfg.training.learning_rate),
                                           weight_decay=float(cfg.training.weight_decay))
        self.scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(self.optimizer, T_max=int(cfg.training.epochs))
        self.tte_scale = float(cfg.event_labeling.safe_min_tte)
        self.n_params = sum(p.numel() for p in self.model.parameters() if p.requires_grad)

    def load(self, path: str | Path) -> None:
        ckpt = torch.load(path, map_location=self.device, weights_only=False)
        self.model.load_state_dict(ckpt["model_state"])

    def save(self, path: Path, epoch: int, score: float) -> None:
        torch.save({"epoch": epoch, "score": score, "model_state": self.model.state_dict()}, path)

    def _epoch(self, loader: DataLoader, train: bool) -> Dict[str, float]:
        self.model.train(train)
        tot = {"loss": 0.0, "cls": 0.0, "ttc": 0.0}
        n = 0
        for batch in loader:
            batch = to_device(batch, self.device)
            with torch.set_grad_enabled(train):
                losses = self.criterion(self.model(batch), batch)
            if train:
                self.optimizer.zero_grad(set_to_none=True)
                losses["loss"].backward()
                nn.utils.clip_grad_norm_(self.model.parameters(), float(self.cfg.training.grad_clip))
                self.optimizer.step()
            bs = batch["risk"].size(0)
            n += bs
            for k in tot:
                tot[k] += float(losses[k].detach()) * bs
        return {k: v / max(n, 1) for k, v in tot.items()}

    @torch.no_grad()
    def predict(self, loader: DataLoader) -> Dict[str, object]:
        self.model.eval()
        probs, ttc, rows = [], [], []
        for batch in loader:
            out = self.model(to_device(batch, self.device))
            probs.append(out["probs"].cpu().numpy())
            ttc.append(out["ttc"].cpu().numpy() * self.tte_scale)
            rows.extend(batch["rows"])
        probs = np.concatenate(probs) if probs else np.zeros((0, int(self.cfg.heads.num_classes)))
        if self.model.head_type == "ordinal":   # count thresholds passed: P(risk > k) > 0.5
            pred = np.stack([probs[:, k + 1:].sum(axis=1) > 0.5 for k in range(probs.shape[1] - 1)], 1).sum(1)
        else:
            pred = probs.argmax(axis=1)
        return {"rows": rows, "probs": probs, "pred": pred,
                "ttc": np.concatenate(ttc) if ttc else np.zeros(0)}

    def evaluate(self, loader: DataLoader) -> Dict[str, object]:
        p = self.predict(loader)
        y = [r["risk_level"] for r in p["rows"]]
        true_ttc = [r["tte"] if (r["tte"] is not None and r["risk_level"] > 0) else float("nan") for r in p["rows"]]
        metrics = compute_metrics(y, p["pred"], p["probs"], p["ttc"], true_ttc, int(self.cfg.heads.num_classes))
        return {"metrics": metrics, "pred": p}

    def fit(self, loaders: Dict[str, DataLoader], run_dir: Path) -> None:
        tr = self.cfg.training
        use_f1 = str(tr.select_metric) == "f1" and "val" in loaders
        best, stale = -float("inf"), 0
        with (run_dir / "history.csv").open("w", newline="", encoding="utf-8") as fp:
            w = csv.writer(fp)
            w.writerow(["epoch", "train_loss", "train_cls", "train_ttc", "val_loss", "val_f1", "val_accuracy", "lr"])
            for epoch in range(1, int(tr.epochs) + 1):
                t0 = time.time()
                train = self._epoch(loaders["train"], True)
                val_loss = val_f1 = val_acc = float("nan")
                if "val" in loaders:
                    val_loss = self._epoch(loaders["val"], False)["loss"]
                    vm = self.evaluate(loaders["val"])["metrics"]
                    val_f1, val_acc = vm["f1"], vm["accuracy"]
                lr = self.optimizer.param_groups[0]["lr"]
                self.scheduler.step()
                w.writerow([epoch, train["loss"], train["cls"], train["ttc"], val_loss, val_f1, val_acc, lr])
                fp.flush()
                LOGGER.info("epoch %3d | train %.4f | val %.4f f1 %.3f acc %.3f | %.1fs",
                            epoch, train["loss"], val_loss, val_f1, val_acc, time.time() - t0)

                # higher is better; without a validation set keep the last epoch
                score = val_f1 if use_f1 else (-val_loss if "val" in loaders else float(epoch))
                if score > best + 1e-5:
                    best, stale = score, 0
                    self.save(run_dir / "best.pt", epoch, score)
                else:
                    stale += 1
                    if stale >= int(tr.early_stop_patience):
                        LOGGER.info("early stop at epoch %d", epoch)
                        break
        self.load(run_dir / "best.pt")   # all later evaluation uses the selected checkpoint


# ---------------------------------------------------------------------------
# Sliding-window (anticipation) evaluation
# ---------------------------------------------------------------------------

def dense_rows(cfg: Config, video_ids: List[str]) -> List[Dict]:
    """Every usable window of the given videos, labelled from the events file."""
    events = {e["video_id"]: e for e in load_events(cfg.event_labeling.events_file)["videos"].values()}
    fps, seq_len = float(cfg.frame_extraction.frame_rate), int(cfg.sequence.length)
    rows = []
    for vid in video_ids:
        meta = Path(cfg.paths.video_cache) / vid / "meta.json"
        e = events.get(vid)
        if e is None or not meta.exists():
            continue
        n = int(json.loads(meta.read_text(encoding="utf-8"))["n_frames"])
        for w in dense_windows(n, fps, seq_len, e.get("events", []), e.get("ignore", []), cfg.event_labeling):
            rows.append({"video_id": vid, "start": w.start, "end": w.end, "risk": w.risk,
                         "risk_level": w.risk or 0, "tte": w.tte, "event_time": w.event_time, "t_end": w.t_end})
    return rows


def anticipation_eval(trainer: Trainer, cfg: Config, video_ids: List[str], store: VideoStore) -> Dict:
    rows = dense_rows(cfg, video_ids)
    if not rows:
        return {}
    loader = DataLoader(ManifestDataset(cfg, rows, store), batch_size=int(cfg.training.batch_size) * 4,
                        collate_fn=collate)
    p = trainer.predict(loader)
    scores = p["probs"][:, int(cfg.evaluation.alarm_class):].sum(axis=1)
    records = [dict(r, score=float(s)) for r, s in zip(p["rows"], scores)]
    return anticipation_metrics(records, cfg)


# ---------------------------------------------------------------------------
# One full run
# ---------------------------------------------------------------------------

def make_loaders(cfg: Config, parts: Dict[str, List[Dict]], store: VideoStore) -> Dict[str, DataLoader]:
    loaders = {}
    for split, rows in parts.items():
        if not rows:
            LOGGER.warning("split '%s' is empty", split)
            continue
        loaders[split] = DataLoader(
            ManifestDataset(cfg, rows, store, augment=(split == "train")),
            batch_size=int(cfg.training.batch_size), shuffle=(split == "train"),
            num_workers=int(cfg.training.num_workers), collate_fn=collate,
        )
    return loaders


def write_test_artifacts(trainer: Trainer, cfg: Config, loaders: Dict, parts: Dict, store: VideoStore,
                         run_dir: Path, tag: str = "test") -> Dict:
    result: Dict[str, object] = {}
    if tag in loaders:
        ev = trainer.evaluate(loaders[tag])
        result = ev["metrics"]
        (run_dir / f"metrics_{tag}.json").write_text(json.dumps(result, indent=2), encoding="utf-8")
        plot_confusion_matrix(result["confusion_matrix"], run_dir / f"confusion_{tag}.png")
        p = ev["pred"]
        with (run_dir / f"predictions_{tag}.csv").open("w", newline="", encoding="utf-8") as fp:
            w = csv.writer(fp)
            w.writerow(["sequence_id", "video_id", "end", "tte", "risk_true", "risk_pred",
                        "p_safe", "p_slight", "p_moderate", "p_high", "ttc_pred_s"])
            for r, yhat, pr, t in zip(p["rows"], p["pred"], p["probs"], p["ttc"]):
                w.writerow([r["sequence_id"], r["video_id"], r["end"], r["tte"], r["risk_level"], int(yhat),
                            *[round(float(x), 4) for x in pr], round(float(t), 3)])
        ant = anticipation_eval(trainer, cfg, sorted({r["video_id"] for r in parts[tag]}), store)
        if ant:
            (run_dir / f"anticipation_{tag}.json").write_text(json.dumps(ant, indent=2), encoding="utf-8")
            result = dict(result, anticipation={k: v for k, v in ant.items() if k != "sweep"})
    return result


def run_training(cfg: Config, fold: Optional[int] = None, seed: Optional[int] = None,
                 name: str = "default") -> Dict:
    seed = int(cfg.seed if seed is None else seed)
    set_seed(seed)
    run_dir = Path(cfg.paths.runs) / name / f"fold{'X' if fold is None else fold}_seed{seed}"
    run_dir.mkdir(parents=True, exist_ok=True)
    save_config(cfg, run_dir / "config_used.yaml")

    parts = split_rows(read_manifest(cfg.paths.manifest), cfg, fold)
    if not parts["train"]:
        raise RuntimeError("No training sequences. Mark more videos or check the manifest.")
    store = VideoStore(cfg)
    loaders = make_loaders(cfg, parts, store)
    trainer = Trainer(cfg, parts["train"])
    LOGGER.info("run %s | train %d val %d test %d sequences | %d trainable params | %s",
                run_dir, len(parts["train"]), len(parts["val"]), len(parts["test"]),
                trainer.n_params, trainer.device)
    trainer.fit(loaders, run_dir)
    result = write_test_artifacts(trainer, cfg, loaders, parts, store, run_dir)
    if result:
        LOGGER.info("test | acc %.3f | macro-F1 %.3f | AUC %.3f | TTC MAE %.2fs",
                    result["accuracy"], result["f1"], result["roc_auc"], result["ttc_mae_s"])
    return result
