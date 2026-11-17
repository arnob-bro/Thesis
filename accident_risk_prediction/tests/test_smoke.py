"""Fast checks that need no videos, GPU or downloads.

    python -m tests.test_smoke          (or: pytest tests)
"""
from __future__ import annotations

import numpy as np
import torch

from features.graph_builder import VideoGraphs, build_video_graphs, filter_ego_detections, node_dim
from models.full_model import AccidentRiskModel, probs_from_logits
from pipelines.training_pipeline import RiskLoss
from utils.config import apply_overrides, load_config
from utils.manifest_dataset import collate
from utils.metrics import compute_metrics
from utils.windows import assign_folds, dense_windows, plan_windows

CFG = "configs/config.yaml"


def test_window_labels():
    cfg = load_config(CFG)
    lab = cfg.event_labeling
    events = [{"time_s": 40.0, "type": "collision"}]
    wins = plan_windows(300, 5.0, 12, events, [[0.0, 3.0]], lab)
    for w in wins:
        assert w.t_end < 40.0 or w.start / 5.0 > 40.0 + lab.post_event_skip   # never touches the crash
        assert w.start / 5.0 > 3.0                                            # ignore range respected
        if w.tte is not None and w.tte < 1.5:
            assert w.risk == 3
        if w.tte is not None and 4.5 <= w.tte < 6.0:
            raise AssertionError("ambiguous buffer must be skipped")
    assert {w.risk for w in wins} == {0, 1, 2, 3}
    # an edit cut between window and event hides the true gap -> those windows are dropped
    cut = plan_windows(300, 5.0, 12, events, [[38.0, 39.0]], lab)
    assert all(w.risk == 0 for w in cut if w.t_end < 38.0)
    assert len(dense_windows(300, 5.0, 12, events, [], lab)) > len(wins)


def test_folds_are_per_video():
    vids = [(f"v{i}", i % 3 != 0) for i in range(20)]
    folds = assign_folds(vids, 5, 42)
    assert set(folds.values()) == set(range(5)) and len(folds) == 20


def _fake_graphs(cfg, n_frames=12):
    tracks = {k: [
        {"track_id": 1, "class_id": 0, "bbox": [200 + 5 * k, 100, 60 + 4 * k, 40 + 3 * k]},
        {"track_id": 2, "class_id": 3, "bbox": [400 - 6 * k, 150, 30, 60]},
    ] for k in range(n_frames)}
    tracks[5] = []   # an empty frame
    return build_video_graphs(tracks, n_frames, None, cfg, (640, 384))


def test_graphs_and_ego_filter(tmp_path=None):
    cfg = load_config(CFG)
    arrays = _fake_graphs(cfg)
    assert arrays["x"].shape[1] == node_dim(cfg)
    assert np.isfinite(arrays["x"]).all() and np.isfinite(arrays["edge_attr"]).all()
    # growing box -> positive looming on the last frame of track 1 (node 1; node 0 is the rider)
    last = arrays["x"][arrays["node_ptr"][-2]:arrays["node_ptr"][-1]]
    assert last[0, 12] == 1.0 and last[1, 10] > 0
    own_bike = {"class_id": 1, "bbox": [100, 200, 400, 184], "confidence": 0.9}
    other = {"class_id": 1, "bbox": [300, 100, 50, 60], "confidence": 0.9}
    assert filter_ego_detections([own_bike, other], cfg, (640, 384)) == [other]


def _batch(cfg, B=3, T=12):
    import io
    buf = io.BytesIO()
    np.savez(buf, **_fake_graphs(cfg, T))
    buf.seek(0)
    vg = VideoGraphs(buf)
    items = []
    for b in range(B):
        items.append({
            "row": {"risk_level": b % 4, "tte": 1.0}, "vit": torch.randn(T, 768), "ego": torch.randn(T, 6),
            "graphs": [vg.get(k, flip=(b == 1)) for k in range(T)],
            "risk": torch.tensor(b % 4), "ttc": torch.tensor(0.2), "ttc_valid": torch.tensor(float(b % 4 > 0)),
        })
    return collate(items)


def test_model_forward_and_loss():
    for overrides in ([], ["heads.type=softmax"], ["ablation.use_vit=false", "ablation.use_temporal=false"],
                      ["ablation.use_gnn=false"], ["ablation.use_ego=false"]):
        cfg = apply_overrides(load_config(CFG), overrides)
        model = AccidentRiskModel(cfg)
        batch = _batch(cfg)
        out = model(batch)
        assert out["probs"].shape == (3, 4)
        assert torch.allclose(out["probs"].sum(1), torch.ones(3), atol=1e-5)
        loss = RiskLoss(cfg, [{"risk_level": c} for c in (0, 0, 0, 1, 2, 3)])(out, batch)["loss"]
        loss.backward()
        assert torch.isfinite(loss)


def test_ordinal_probs_and_metrics():
    p = probs_from_logits(torch.tensor([[5.0, -5.0, 9.0]]), "ordinal")   # non-monotone logits
    assert (p >= 0).all() and abs(float(p.sum()) - 1.0) < 1e-5
    m = compute_metrics([0, 1, 2, 3], [0, 1, 3, 3], np.eye(4)[[0, 1, 3, 3]], [1.0] * 4, [float("nan"), 2.0, 1.0, 0.5])
    assert m["accuracy"] == 0.75 and m["ordinal_mae"] == 0.25 and m["risky_recall"] == 1.0


def test_looming_rule():
    import io
    from features import looming_rule as lr
    cfg = load_config(CFG)
    buf = io.BytesIO()
    np.savez(buf, **_fake_graphs(cfg))
    buf.seek(0)
    per_frame = lr.frame_tau(VideoGraphs(buf), cfg)
    assert per_frame.shape == (12,) and (per_frame > 0).all()
    assert per_frame[-1] < float(cfg.features.tau_clamp)          # growing box ahead -> finite tau
    assert lr.window_tau(np.array([6.0, 1.0, 2.0]), 2, 2) == 2.0   # needs both frames low
    t = lr.bin_thresholds(cfg)
    assert t == (4.5, 3.0, 1.5)
    tau = np.array([0.5, 2.0, 4.0, 5.9])
    assert lr.classify(tau, t).tolist() == [3, 2, 1, 0]
    p = lr.probabilities(tau, t, 0.5)
    assert np.allclose(p.sum(1), 1.0) and p.argmax(1).tolist() == [3, 2, 1, 0]
    assert ((p[:, 1:].sum(1) >= 0.5) == (tau < t[0])).all()       # alarm threshold matches the rule
    y = np.array([3, 3, 2, 2, 1, 1, 0, 0])
    fit = lr.fit_thresholds(np.array([0.3, 0.4, 1.0, 1.1, 2.0, 2.1, 5.0, 5.5]), y)
    assert lr.classify(np.array([0.3, 1.0, 2.0, 5.0]), fit).tolist() == [3, 2, 1, 0]


if __name__ == "__main__":
    for name, fn in list(globals().items()):
        if name.startswith("test_") and callable(fn):
            fn()
            print(f"ok  {name}")
    print("all smoke tests passed")
