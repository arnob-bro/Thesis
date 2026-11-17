"""Predict risk over time for one video.

    python inference.py --run runs/full/fold0_seed0 --video path/to/ride.mp4

Processes the video (frames, detection, tracking, graphs, ViT), slides the
model over it one frame at a time, and writes next to --out:

    <name>_risk.json   risk class, probabilities, alarm score and predicted
                       time-to-collision for every moment of the video
    <name>_risk.png    risk-over-time curve
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

from torch.utils.data import DataLoader

from pipelines.build_video_cache import Models, process_video
from pipelines.training_pipeline import Trainer
from utils.config import load_config
from utils.events import make_video_id
from utils.manifest_dataset import ManifestDataset, VideoStore, collate
from utils.visualization import plot_risk_curve


def main() -> None:
    p = argparse.ArgumentParser(description="Risk prediction for one video")
    p.add_argument("--run", required=True, help="Run folder containing best.pt and config_used.yaml")
    p.add_argument("--video", required=True)
    p.add_argument("--out", default="outputs", help="Output folder")
    args = p.parse_args()

    run_dir, video = Path(args.run), Path(args.video)
    cfg = load_config(run_dir / "config_used.yaml")
    vid = make_video_id(video.name)
    cache_root = Path(cfg.paths.video_cache)
    process_video(video, vid, cache_root / vid, cfg, Models(cfg))

    meta = json.loads((cache_root / vid / "meta.json").read_text(encoding="utf-8"))
    fps, seq_len = float(meta["fps"]), int(cfg.sequence.length)
    rows = [{"video_id": vid, "start": e - seq_len + 1, "end": e, "risk_level": 0, "tte": None}
            for e in range(seq_len - 1, int(meta["n_frames"]))]
    if not rows:
        raise SystemExit("Video is shorter than one sequence.")

    trainer = Trainer(cfg)
    trainer.load(run_dir / "best.pt")
    loader = DataLoader(ManifestDataset(cfg, rows, VideoStore(cfg)), batch_size=128, collate_fn=collate)
    pred = trainer.predict(loader)
    k = int(cfg.evaluation.alarm_class)
    thr = float(cfg.evaluation.alarm_threshold)

    timeline = []
    for r, yhat, pr, ttc in zip(pred["rows"], pred["pred"], pred["probs"], pred["ttc"]):
        score = float(pr[k:].sum())
        timeline.append({
            "time_s": round(r["end"] / fps, 2), "risk_level": int(yhat),
            "probabilities": [round(float(x), 4) for x in pr],
            "alarm_score": round(score, 4), "alarm": score >= thr,
            "ttc_pred_s": round(float(ttc), 2) if yhat > 0 else None,
        })

    out_dir = Path(args.out)
    out_dir.mkdir(parents=True, exist_ok=True)
    out_json = out_dir / f"{vid}_risk.json"
    out_json.write_text(json.dumps({"video": video.name, "fps": fps, "timeline": timeline}, indent=2),
                        encoding="utf-8")
    plot_risk_curve([t["risk_level"] for t in timeline], out_dir / f"{vid}_risk.png",
                    fps=fps, title=f"Risk over time: {vid}")
    n_alarm = sum(t["alarm"] for t in timeline)
    print(f"{len(timeline)} time steps, {n_alarm} with alarm, max risk {max(t['risk_level'] for t in timeline)}")
    print(f"Wrote {out_json}")


if __name__ == "__main__":
    main()
