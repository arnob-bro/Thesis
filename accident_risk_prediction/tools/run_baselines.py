"""Simple baselines on the same folds as the main model.

    python -m tools.run_baselines --config configs/config.yaml

Baselines (all use the cached features, so this runs in a minute or two):

    majority       always predict the most common training class
    vit_logreg     mean ViT embedding of the sequence -> logistic regression
    handcrafted    graph + ego-motion summary statistics -> gradient boosting
    vit_gru        per-frame ViT embeddings -> small GRU
    tau_fixed      looming time-to-contact rule, thresholds = the TTE bins (no training)
    tau_tuned      same rule, thresholds chosen for macro-F1 on the training folds

For every fold: test = fold k, train = everything else. Writes
runs/baselines/results.csv (per fold) and prints mean +/- std. The two tau
rules are also slid over the whole test videos, like the main model, and
their anticipation metrics go to runs/baselines/anticipation.csv.
"""
from __future__ import annotations

import argparse
import csv
from collections import Counter, defaultdict
from pathlib import Path
from typing import Dict, List, Tuple

import numpy as np
import torch
import torch.nn as nn
from sklearn.ensemble import HistGradientBoostingClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

from features import looming_rule as lr
from features.graph_builder import window_stats
from pipelines.training_pipeline import dense_rows
from utils.anticipation import anticipation_metrics
from utils.config import load_config
from utils.manifest_dataset import VideoStore, read_manifest
from utils.metrics import compute_metrics
from utils.seed import set_seed

METRICS = ["accuracy", "f1", "risky_recall", "ordinal_mae", "roc_auc"]


def featurise(rows: List[Dict], store: VideoStore) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Returns (vit sequence [N,T,768], handcrafted [N,F], labels [N])."""
    seq, hand, y = [], [], []
    for r in rows:
        v = store.get(r["video_id"])
        s, e = r["start"], r["end"] + 1
        seq.append(np.asarray(v["vit"][s:e], dtype=np.float32))
        ego = np.asarray(v["ego"][s:e], dtype=np.float32)
        stats = window_stats([v["graphs"].get(k) for k in range(s, e)])
        hand.append(np.concatenate([stats, ego.mean(0), ego.std(0), ego[-1]]))
        y.append(r["risk_level"])
    return np.stack(seq), np.stack(hand), np.asarray(y)


def _probs(clf, X: np.ndarray, n_cls: int) -> np.ndarray:
    p = np.zeros((len(X), n_cls), dtype=np.float32)
    p[:, clf.classes_] = clf.predict_proba(X)
    return p


def gru_baseline(Xtr, ytr, Xte, n_cls: int, device: torch.device, epochs: int = 30) -> np.ndarray:
    class Net(nn.Module):
        def __init__(self) -> None:
            super().__init__()
            self.proj = nn.Linear(Xtr.shape[-1], 64)
            self.gru = nn.GRU(64, 64, batch_first=True)
            self.out = nn.Linear(64, n_cls)

        def forward(self, x):
            _, h = self.gru(torch.relu(self.proj(x)))
            return self.out(h[-1])

    net = Net().to(device)
    counts = np.bincount(ytr, minlength=n_cls).astype(float)
    weight = torch.tensor(np.clip(counts.sum() / (n_cls * np.maximum(counts, 1)), 0.1, 10), dtype=torch.float32)
    loss_fn = nn.CrossEntropyLoss(weight=weight.to(device))
    opt = torch.optim.AdamW(net.parameters(), lr=1e-3, weight_decay=0.01)
    X, y = torch.from_numpy(Xtr).to(device), torch.from_numpy(ytr).long().to(device)
    for _ in range(epochs):
        net.train()
        perm = torch.randperm(len(X), device=device)
        for i in range(0, len(X), 32):
            idx = perm[i:i + 32]
            opt.zero_grad()
            loss_fn(net(X[idx]), y[idx]).backward()
            opt.step()
    net.eval()
    with torch.no_grad():
        return torch.softmax(net(torch.from_numpy(Xte).to(device)), dim=-1).cpu().numpy()


def main() -> None:
    p = argparse.ArgumentParser(description="Run baselines on all folds")
    p.add_argument("--config", required=True)
    args = p.parse_args()
    cfg = load_config(args.config)
    set_seed(int(cfg.seed))
    n_cls = int(cfg.heads.num_classes)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    rows = read_manifest(cfg.paths.manifest)
    store = VideoStore(cfg)
    seq, hand, y = featurise(rows, store)
    folds = np.asarray([r["fold"] for r in rows])

    rule = lr.rule_settings(cfg)
    frame_tau: Dict[str, np.ndarray] = {}

    def taus(rs: List[Dict]) -> np.ndarray:
        for vid in {r["video_id"] for r in rs} - frame_tau.keys():
            frame_tau[vid] = lr.frame_tau(store.get(vid)["graphs"], cfg)
        return np.asarray([lr.window_tau(frame_tau[r["video_id"]], r["end"], rule["persist"]) for r in rs])

    tau = taus(rows)
    fixed_t = lr.bin_thresholds(cfg)
    alarm = int(cfg.evaluation.alarm_class)

    results: List[Dict] = []
    antic: List[Dict] = []
    for k in sorted(set(folds.tolist())):
        tr, te = folds != k, folds == k
        if len(set(y[tr].tolist())) < 2 or te.sum() == 0:
            continue
        preds: Dict[str, np.ndarray] = {}
        maj = np.zeros((int(te.sum()), n_cls), dtype=np.float32)
        maj[:, Counter(y[tr].tolist()).most_common(1)[0][0]] = 1.0
        preds["majority"] = maj
        logreg = make_pipeline(StandardScaler(), LogisticRegression(max_iter=2000, class_weight="balanced", C=0.1))
        preds["vit_logreg"] = _probs(logreg.fit(seq[tr].mean(1), y[tr]), seq[te].mean(1), n_cls)
        gb = HistGradientBoostingClassifier(max_depth=4, learning_rate=0.1, max_iter=200,
                                            class_weight="balanced", random_state=int(cfg.seed))
        preds["handcrafted"] = _probs(gb.fit(hand[tr], y[tr]), hand[te], n_cls)
        preds["vit_gru"] = gru_baseline(seq[tr], y[tr], seq[te], n_cls, device)

        tuned_t = lr.fit_thresholds(tau[tr], y[tr], n_cls)
        rule_t = {"tau_fixed": fixed_t, "tau_tuned": tuned_t}
        hard: Dict[str, np.ndarray] = {}
        for name, t in rule_t.items():
            preds[name] = lr.probabilities(tau[te], t, rule["softness"])
            hard[name] = lr.classify(tau[te], t)
        print(f"fold {k} tau thresholds (slight/moderate/high, s): fixed {fixed_t}  tuned {tuned_t}")

        test_videos = sorted({r["video_id"] for r, f in zip(rows, folds) if f == k})
        dense = dense_rows(cfg, test_videos)
        if dense:
            dense_tau = taus(dense)
            for name, t in rule_t.items():
                score = lr.probabilities(dense_tau, t, rule["softness"])[:, alarm:].sum(axis=1)
                a = anticipation_metrics([dict(r, score=float(s)) for r, s in zip(dense, score)], cfg)
                at = a["at_threshold"]
                antic.append({"baseline": name, "fold": k, "thresholds": "/".join(f"{v:g}" for v in t),
                              "n_events": a["n_events"], "average_precision": a["average_precision"],
                              "event_recall": at["event_recall"], "mean_tta_s": at["mean_tta_s"],
                              "false_alarms_per_min": at["false_alarms_per_min"]})

        for name, pr in preds.items():
            m = compute_metrics(y[te], hard.get(name, pr.argmax(1)), pr, num_classes=n_cls)
            results.append({"baseline": name, "fold": k, **{key: m[key] for key in METRICS}})
            print(f"fold {k} {name:<12} acc {m['accuracy']:.3f}  macro-F1 {m['f1']:.3f}  risky-recall {m['risky_recall']:.3f}")

    if not results:
        raise SystemExit("No fold had enough data for baselines.")
    out = Path(cfg.paths.runs) / "baselines" / "results.csv"
    out.parent.mkdir(parents=True, exist_ok=True)
    with out.open("w", newline="", encoding="utf-8") as fp:
        w = csv.DictWriter(fp, fieldnames=list(results[0].keys()))
        w.writeheader()
        w.writerows(results)

    by: Dict[str, List[Dict]] = defaultdict(list)
    for r in results:
        by[r["baseline"]].append(r)
    print(f"\n{'baseline':<12} " + " ".join(f"{m:>16}" for m in METRICS))
    for name, rs in by.items():
        cells = [f"{np.nanmean([r[m] for r in rs]):.3f} ± {np.nanstd([r[m] for r in rs]):.3f}" for m in METRICS]
        print(f"{name:<12} " + " ".join(f"{c:>16}" for c in cells))
    print(f"\nPer-fold results -> {out}")

    if antic:
        out_a = out.parent / "anticipation.csv"
        with out_a.open("w", newline="", encoding="utf-8") as fp:
            w = csv.DictWriter(fp, fieldnames=list(antic[0].keys()))
            w.writeheader()
            w.writerows(antic)
        cols = ["average_precision", "event_recall", "mean_tta_s", "false_alarms_per_min"]
        print(f"\nAnticipation (whole test videos, alarm = P(risk >= {alarm}) >= {cfg.evaluation.alarm_threshold})")
        print(f"{'baseline':<12} " + " ".join(f"{c:>22}" for c in cols))
        for name in ("tau_fixed", "tau_tuned"):
            rs = [r for r in antic if r["baseline"] == name]
            cells = [f"{np.nanmean([r[c] for r in rs]):.3f} ± {np.nanstd([r[c] for r in rs]):.3f}" for c in cols]
            print(f"{name:<12} " + " ".join(f"{c:>22}" for c in cells))
        print(f"Per-fold anticipation -> {out_a}")


if __name__ == "__main__":
    main()
