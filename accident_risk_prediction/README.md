# Multimodal Spatiotemporal Accident Risk Prediction

Research-grade implementation of a multimodal spatiotemporal accident risk
prediction system, following the design specification in `arnob/*.md`.

The system fuses four semantic modalities:

| Modality            | Source                  | Output shape |
| ------------------- | ----------------------- | ------------ |
| Scene               | ViT (frozen)            | `[B, 768]`   |
| Interaction         | GAT over per-frame KNN graphs (ego-excluded) | `[B, N, 256]` |
| Temporal evolution  | Transformer encoder     | `[B, 512]`   |
| Ego vehicle state   | MLP encoder over dashboard telemetry | `[B, 64]`   |

A cross-attention fusion module produces a `[B, 512]` representation that feeds
two heads:

- Classification: risk level `0..3` (safe / slight / moderate / high)
- Regression: approximate time-to-collision (TTC)

YOLOv8 and ViT are **frozen feature extractors**. The trainable parameters are
in: GAT, Temporal Transformer, Ego Encoder, Fusion, and Heads.

## 1. Setup

```bash
python -m venv .venv
source .venv/bin/activate      # Windows: .venv\Scripts\activate
pip install -r requirements.txt

# Install torch-geometric extension wheels matching your torch/CUDA build
# (see https://pytorch-geometric.readthedocs.io/en/latest/install/installation.html)
```

## 2. Dataset Preparation

Place raw videos using the naming convention `video_<location>_<weather>_<id>.mp4`:

```
data/raw_videos/{train,val,test}/video_dhaka_sunny_001.mp4
```

Per-sequence ego-state telemetry is required (synchronised to the extracted
frames) and stored as CSV/JSON in `data/features/ego_state/{train,val,test}/`.

Risk labels are **sequence-level** JSON files in
`data/annotations/risk_labels/`:

```json
{ "sequence_id": "0001", "risk_level": 2, "ttc_estimate": 1.8 }
```

Run the offline preprocessing pipeline (stages match the MD training strategy):

```bash
# Stage 1: extract frames at 5 FPS
python -m pipelines.frame_extractor --config configs/config.yaml --split train
python -m pipelines.frame_extractor --config configs/config.yaml --split val
python -m pipelines.frame_extractor --config configs/config.yaml --split test

# Stage 2: YOLOv8 detection (cached)
python -m pipelines.detection_pipeline --config configs/config.yaml --split train
python -m pipelines.detection_pipeline --config configs/config.yaml --split val
python -m pipelines.detection_pipeline --config configs/config.yaml --split test

# Stage 3: ByteTrack tracking (cached)
python -m pipelines.tracking_pipeline --config configs/config.yaml --split train
python -m pipelines.tracking_pipeline --config configs/config.yaml --split val
python -m pipelines.tracking_pipeline --config configs/config.yaml --split test

# Stage 4: per-frame node/edge features + KNN graphs (ego-excluded)
python -m pipelines.graph_pipeline --config configs/config.yaml --split train
python -m pipelines.graph_pipeline --config configs/config.yaml --split val
python -m pipelines.graph_pipeline --config configs/config.yaml --split test

# Stage 4b: ego-state alignment to sequences
python -m pipelines.ego_state_pipeline --config configs/config.yaml --split train
python -m pipelines.ego_state_pipeline --config configs/config.yaml --split val
python -m pipelines.ego_state_pipeline --config configs/config.yaml --split test
```

ViT scene embeddings are computed on the fly during training but cached to
`data/cache/embeddings/` after the first epoch.

## 3. Training

```bash
python train.py --config configs/config.yaml
```

Checkpoints land in `checkpoints/`, structured logs in `logs/`.

Ablations are toggled in `configs/config.yaml` under the `ablation:` section
(`use_vit`, `use_gnn`, `use_temporal`, `use_ego`).

## 4. Evaluation

```bash
python evaluate.py --config configs/config.yaml --checkpoint checkpoints/best.pt
```

Reports Accuracy, Precision, Recall, F1, ROC-AUC, Time-to-Accident (TTA), and
Early Risk Detection Score, plus a confusion matrix.

## 5. Inference (single video)

```bash
python inference.py \
    --config configs/config.yaml \
    --checkpoint checkpoints/best.pt \
    --video path/to/clip.mp4 \
    --ego_state path/to/ego.csv \
    --output prediction.json
```

The script runs the full pipeline (frame extraction → detection → tracking →
graphs → ViT → ego encoding → fusion → heads) and writes per-sequence risk
class + TTC estimate.

## 6. Project Layout

See top-level directory tree. Each module is independent and configurable.

## 7. Reproducibility

- Global seed in `configs/config.yaml` (`seed: 42`)
- Configs persisted alongside checkpoints
- Frame order strictly chronological
- Train/val/test split by **video** (never by frame)
- Caching layer prevents recomputation of expensive YOLO / ViT outputs
