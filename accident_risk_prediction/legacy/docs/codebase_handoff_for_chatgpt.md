# Accident Risk Prediction Codebase Handoff

This document is a compact map of the whole project for use in future ChatGPT
sessions. Give ChatGPT this file when asking for debugging, refactoring,
experiments, training help, or thesis-writing support.

## Project Summary

The project implements a multimodal spatiotemporal accident risk prediction
system. It turns dashcam videos into fixed-length frame sequences, detects road
objects with frozen YOLOv8, tracks them with ByteTrack, builds per-frame
interaction graphs, and trains a deep learning model that fuses four modalities:

- Scene context from a frozen Vision Transformer.
- Object interaction graphs from a GAT.
- Temporal evolution from a Transformer encoder.
- Ego-vehicle telemetry from an MLP encoder.

The final model predicts:

- `risk_level`: class `0..3` where `0=safe`, `1=slight`, `2=moderate`,
  `3=high`.
- `ttc_estimate`: approximate time to collision.

Main config: `configs/config.yaml`.

## Important Current State

- Source code is under `features/`, `models/`, `pipelines/`, `tools/`, `utils/`,
  plus top-level `train.py`, `evaluate.py`, and `inference.py`.
- Large generated/data artifacts exist under `data/`, `checkpoints/`, `logs/`,
  and `yolov8n.pt`.
- Python cache folders (`__pycache__/`) and `.venv/` are not source code.
- The current sequence folders are not only `train`, `val`, `test`; there are
  chunked folders such as `train 0-500`, `test(1-100)`, etc. The training
  loader, however, expects exactly `data/sequences/train`,
  `data/sequences/val`, and `data/sequences/test` unless the config/code is
  changed.
- Existing docs mention `tools.export_yolo_labels` and
  `tools.import_yolo_labels`, but those files are not present in this codebase.

## Runtime Dependencies

Declared in `requirements.txt`:

- Deep learning: `torch`, `torchvision`, `torch-geometric`.
- Detection/tracking: `ultralytics`, `supervision`.
- Vision transformer: `transformers`.
- Image/video: `opencv-python`, `Pillow`.
- Data/science: `numpy`, `scipy`, `pandas`, `scikit-learn`.
- Config/logging/progress/plots: `PyYAML`, `tqdm`, `matplotlib`.

## Configuration

`configs/config.yaml` is the single source of truth.

Key sections:

- `paths`: roots for raw videos, extracted frames, annotations, sequences,
  features, graphs, metadata, checkpoints, and logs.
- `frame_extraction`: default 5 FPS, resize `[640, 384]`.
- `sequence`: default length `12`, stride `12`.
- `yolo`: uses `yolov8n.pt`, confidence/iou thresholds, device, and COCO-to
  project class mapping.
- `tracking`: ByteTrack thresholds and `min_track_length`.
- `features`: node dim `8`, edge dim `4`, clamps for velocity, acceleration,
  and TTC.
- `graph`: KNN graph settings, max nodes.
- `vit`, `gnn`, `temporal`, `ego_encoder`, `fusion`, `heads`: model dimensions.
- `loss`: TTC loss weight and optional class weights.
- `training`: epochs, batch size, LR, workers, device, checkpointing, early stop.
- `ablation`: switches for `use_vit`, `use_gnn`, `use_temporal`, `use_ego`.

## Expected Data Schemas

Sequence directory expected by training:

```text
data/sequences/<split>/<sequence_id>/
  frames/*.jpg
  graph.pt
  ego_state.json
  label.json
  metadata.json
```

`label.json`:

```json
{
  "sequence_id": "1_seq0000",
  "risk_level": 0,
  "ttc_estimate": null
}
```

`ego_state.json`:

```json
{
  "speed": [0.0],
  "rpm_estimate": [0.0],
  "brake": [0.0],
  "acceleration": [0.0]
}
```

Detection item:

```json
{
  "frame_id": 1,
  "class_id": 0,
  "confidence": 0.91,
  "bbox": [x, y, w, h]
}
```

Track item adds `track_id`.

`graph.pt` contains `list[torch_geometric.data.Data]`, one graph per frame.
Each graph uses:

- `x`: node features `[x, y, w, h, vx, vy, acceleration, object_type]`.
- `edge_index`: directed KNN edges.
- `edge_attr`: `[proximity, rel_velocity_long, rel_velocity_lat, ttc]`.

## Recommended Pipeline Order

Per-sequence path, which matches the current training layout best:

```powershell
python -m pipelines.frame_extractor --config configs/config.yaml --split train
python -m pipelines.sequence_generator --config configs/config.yaml --split train
python -m pipelines.detection_per_sequence --config configs/config.yaml --split train
python -m pipelines.tracking_per_sequence --config configs/config.yaml --split train
python -m pipelines.graph_per_sequence --config configs/config.yaml --split train
python -m pipelines.ego_state_pipeline --config configs/config.yaml --split train
python train.py --config configs/config.yaml
python evaluate.py --config configs/config.yaml --checkpoint checkpoints/best.pt --split test
```

Repeat preprocessing for `val` and `test`.

Older/video-level path also exists:

```powershell
python -m pipelines.frame_extractor --config configs/config.yaml --split train
python -m pipelines.detection_pipeline --config configs/config.yaml --split train
python -m pipelines.tracking_pipeline --config configs/config.yaml --split train
python -m pipelines.graph_pipeline --config configs/config.yaml --split train
```

## Top-Level Entrypoints

### `train.py`

Purpose: reproducible training entrypoint.

Important functions:

- `parse_args()`: requires `--config`, optional `--resume`.
- `main()`: loads config, seeds RNGs, creates `Trainer`, optionally resumes,
  builds loaders, and trains.

Main dependency: `pipelines.training_pipeline.Trainer`.

### `evaluate.py`

Purpose: load a checkpoint and evaluate on `val` or `test`.

Important functions:

- `parse_args()`: requires `--config` and `--checkpoint`; `--split` defaults
  to `test`.
- `main()`: loads trainer/checkpoint, evaluates loader, writes metrics JSON and
  confusion matrix image under `logs/`.

### `inference.py`

Purpose: end-to-end prediction for one video.

Important functions:

- `_extract_frames()`: extracts temp frames from a video.
- `_detect()`: YOLO detections grouped by frame.
- `_track()`: ByteTrack output grouped by frame.
- `_tracks_to_points()`: converts dict tracks to `TrackPoint`.
- `_build_graph()`: builds a PyG graph for one frame.
- `_read_image_as_tensor()`: prepares ViT-normalized frame tensor.
- `_load_ego_csv()`: loads optional CSV/JSON ego telemetry or zeros.
- `run_inference()`: loads checkpoint, runs all stages, windows frames into
  sequences, predicts risk/TTC, writes output JSON.

## `utils/` Modules

### `utils/config.py`

- `Config`: recursive dict with attribute access.
- `load_config(path)`: reads YAML into `Config`.
- `save_config(cfg, path)`: writes config back to YAML.

### `utils/dataset.py`

- `_read_image()`: OpenCV load, resize, BGR-to-RGB.
- `_normalise_image()`: converts image to ImageNet-normalized tensor.
- `_load_ego_state()`: loads JSON-of-lists ego telemetry, pads/truncates.
- `SequenceDataset`: loads one sequence folder into frames, graphs, ego state,
  risk label, TTC, and TTC-valid mask.
- `collate_sequences()`: stacks frames/ego/labels and builds one PyG `Batch`
  per timestep.

Important: `SequenceDataset(cfg, split)` looks under
`Path(cfg.paths.sequences) / split`.

### `utils/metrics.py`

- `MetricResult`: dataclass for accuracy, precision, recall, F1, ROC-AUC, TTA,
  early risk detection, per-class metrics, confusion matrix.
- `classification_metrics()`: macro classification metrics and confusion.
- `time_to_accident()`: MAE between predicted and true TTC for risky cases.
- `early_risk_detection_score()`: lead/lag in seconds for detecting risk onset.
- `compute_all_metrics()`: combines classification/TTC/early-risk metrics.

### `utils/visualization.py`

- `draw_detections()`: overlays detection boxes.
- `draw_tracks()`: overlays track IDs.
- `draw_graph_edges()`: overlays graph nodes/edges and optional edge labels.
- `plot_risk_curve()`: saves risk-over-time plot.
- `plot_confusion_matrix()`: saves confusion matrix image.

### `utils/logger.py`

- `get_logger()`: creates console logger and optional rotating file logger.

### `utils/seed.py`

- `set_seed()`: seeds Python, NumPy, PyTorch, CUDA, and cuDNN determinism flags.

## `features/` Modules

### `features/object_features.py`

- `TrackPoint`: raw tracked object at one frame.
- `ObjectFeature`: computed object node feature with velocity and acceleration.
- `compute_object_features()`: estimates center velocity and acceleration per
  track across frames.
- `normalize_node_features()`: converts object features into normalized GNN
  node vectors of shape `[num_nodes, 8]`.

### `features/interaction_features.py`

- `pairwise_distance()`, `bbox_edge_distance()`, `bbox_overlap_ratio()`: geometry
  helpers.
- `bbox_proximity_score()`: proximity from edge distance and overlap.
- `relative_velocity_components()`: relative velocity along/across bbox
  separation direction.
- `geometry_ttc()`: rectangle-overlap TTC under constant relative velocity.
- `build_edges_knn()`: directed KNN edges and edge features.
- `normalize_edge_features()`: scales edge features for GNN input.
- `compute_ego_object_interactions()`: optional ego-to-object signals for
  debugging; ego is not a graph node.

### `features/scene_features.py`

- `cache_path_for()`, `save_scene_embedding()`, `load_scene_embedding()`: helper
  functions for `.npy` scene embedding caches. The active model computes ViT
  embeddings in `models/vit/vit_model.py`.

## `models/` Modules

### `models/full_model.py`

Defines `AccidentRiskModel`.

Flow:

1. `_encode_scene()`: frozen ViT over all frames, mean over time.
2. `_encode_graphs()`: GAT per timestep, pooled per-frame vectors, dense last
   frame node set for fusion.
3. `_encode_temporal()`: transformer over per-frame GNN embeddings.
4. `_encode_ego()`: ego telemetry encoder.
5. `CrossAttentionFusion`: temporal embedding queries ViT, GNN, and ego streams.
6. `cls_head`: risk logits.
7. `ttc_head`: scalar TTC prediction.

Supports ablation switches from config.

### `models/yolo/yolo_detector.py`

- `YoloDetector`: wrapper around Ultralytics YOLO.
- `predict_image()` and `predict_array()` produce project-format detections.
- `_results_to_detections()` maps COCO IDs to project IDs and returns absolute
  `[x, y, w, h]` boxes.

YOLO is frozen and not trained.

### `models/tracking/byte_tracker.py`

- `ByteTrackWrapper`: wrapper around `supervision.ByteTrack`.
- `update()`: converts detections to supervision format, runs tracker, returns
  tracked dicts.
- `filter_short_tracks()`: removes tracks shorter than `min_track_length`.

### `models/gnn/gat_model.py`

- `InteractionGAT`: GATv2 network over PyG batches.
- `forward()`: returns node embeddings and global mean-pooled per-graph vectors.
- `to_dense_node_set()`: pads variable node counts to `[B, max_nodes, D]` plus
  mask for fusion attention.

### `models/vit/vit_model.py`

- `FrozenViT`: HuggingFace ViT, frozen parameters, always eval mode.
- `encode()`: returns CLS token embedding projected to configured output dim.

### `models/temporal/temporal_transformer.py`

- `PositionalEncoding`: sinusoidal encoding.
- `TemporalTransformer`: projects per-frame embeddings, prepends learnable CLS
  token, returns CLS summary `[B, d_model]`.

### `models/ego_state/ego_encoder.py`

- `EgoEncoder`: MLP applied per timestep, mean-pooled over time, layer-normalized.

### `models/fusion/cross_attention_fusion.py`

- `_CrossAttentionBlock`: multihead cross-attention plus feedforward residual.
- `CrossAttentionFusion`: projects ViT/GNN/temporal/ego into shared `d_model`;
  temporal is the query; outputs fused `[B, output_dim]`.

## `pipelines/` Modules

### `pipelines/frame_extractor.py`

- `list_videos()`: finds videos under `data/raw_videos/<split>`.
- `extract_from_video()`: extracts frames at target FPS, resizes, writes
  `extraction_meta.json`.
- `run()`: processes all videos for a split.

Output: `data/extracted_frames/<split>/<video>/frame_<video>_<n>.jpg`.

### `pipelines/sequence_generator.py`

- Splits extracted frames into fixed-length sequence folders.
- Preserves original frame filenames.
- Writes `metadata.json` with `frame_ids`, `frame_names`, video, FPS, image size.

Output: `data/sequences/<split>/<video>_seqNNNN/`.

### `pipelines/detection_per_sequence.py`

- Runs YOLO over `data/sequences/<split>/*/frames/*.jpg`.
- Writes `detections.json` inside each sequence folder.

### `pipelines/tracking_per_sequence.py`

- Loads per-sequence `detections.json`.
- Runs ByteTrack independently for each sequence.
- Writes per-sequence `tracks.json`.

### `pipelines/graph_per_sequence.py`

- Loads per-sequence `tracks.json` and `metadata.json`.
- Builds object features and interaction graphs for metadata frame IDs.
- Saves `graph.pt` inside each sequence.
- Adds `track_id` tensor to graph nodes for visualization.

Note: this file does not create default `label.json` or `ego_state.json`; those
must exist already or be created with tools.

### `pipelines/ego_state_pipeline.py`

- Audits every sequence for `ego_state.json`.
- Writes `data/metadata/ego_state_audit_<split>.csv`.

### `pipelines/detection_pipeline.py`

Older video-level detection:

- Reads `data/extracted_frames/<split>/<video>`.
- Writes `data/annotations/detections/<split>/<video>/detections.json`.

### `pipelines/tracking_pipeline.py`

Older video-level tracking:

- Reads video-level detection JSON.
- Writes `data/annotations/tracking/<split>/<video>/tracks.json`.

### `pipelines/graph_pipeline.py`

Older video-level graph/sequence builder:

- Reads extracted frames and video-level tracking.
- Builds sequences and graphs.
- Materializes default labels and ego state if missing.
- Copies frames as `frame_0001.jpg`, etc.

### `pipelines/training_pipeline.py`

- `_move_batch_to_device()`: moves tensors and PyG batches.
- `make_loaders()`: creates `train`, `val`, `test` DataLoaders when available.
- `MultimodalLoss`: cross entropy plus weighted SmoothL1 TTC loss; TTC loss is
  masked by `ttc_valid`.
- `Trainer`: owns model, optimizer, scheduler, checkpointing, train loop,
  evaluation, and metric artifacts.

Important issue to know: in `_run_epoch()`, the loss call passes `out["ttc"]`
as both predicted and true TTC:

```python
out["ttc"], batch["ttc"], batch["ttc_valid"]
```

should be the intended pattern, but current code passes `out["ttc"]` twice.
That makes TTC loss always zero. Ask ChatGPT to verify/fix this before serious
training.

## `tools/` Modules

### `tools/init_sequence_annotations.py`

- Creates placeholder `label.json` and `ego_state.json` for sequence folders.
- Supports `--split train|val|test|all`, `--overwrite`, `--dry-run`, `--open`.

### `tools/write_sequence_annotations.py`

- Writes labels from CSV with columns like `sequence_id,risk_level,ttc_estimate`.
- Imports ego telemetry files from a directory.
- Converts CSV ego data to JSON-of-lists.
- Finds sequence directories by exact or substring match.

### `tools/export_labels_to_sequences.py`

- Reads a master JSON mapping sequence IDs to label payloads.
- Writes each sequence's `label.json`.
- Supports split/all, overwrite, dry run.

### `tools/export_ego_states_to_sequences.py`

- Reads a master JSON mapping sequence IDs to ego telemetry payloads.
- Writes each sequence's `ego_state.json`.
- Supports split/all, overwrite, dry run.

### `tools/visualize_detections.py`

- Video-level QA visualizer for `data/annotations/detections/<split>/<video>`.
- Draws boxes onto extracted frames.
- Writes images under `data/qa/detections/<split>/<video>`.

### `tools/visualize_detections_per_sequence.py`

- Per-sequence QA visualizer.
- Uses `metadata.json` to resolve exact original frame filenames.
- Writes annotated images under `qa_detections` inside the sequence, unless
  `--output` is provided.

### `tools/visualize_graphs.py`

- Loads `graph.pt` and overlays graph nodes/edges on sequence frames.
- Uses metadata to resolve frame order.
- Can denormalize labels using config values.
- Writes `qa_graphs` inside each sequence.

## Existing Docs

### `README.md`

High-level project description, setup, preprocessing, training, evaluation, and
inference commands.

### `docs/project_guide.md`

Long workflow guide for preparing videos, generating sequences, detecting,
tracking, graph construction, annotation, training, evaluation, and inference.
Some command descriptions are stale: it says per-sequence detection/tracking
outputs go to `data/annotations/...`, but current per-sequence scripts write
inside each sequence folder.

### `docs/to_visualize_and_correct_yolov8n_detections.md`

Describes a detection correction workflow. It references missing tools
`export_yolo_labels` and `import_yolo_labels`; only visualization tools exist.

## Known Risks And Fixes To Ask ChatGPT About

1. TTC loss bug in `pipelines/training_pipeline.py`: training currently compares
   predicted TTC to itself. It should compare predicted TTC to `batch["ttc"]`.
2. Folder mismatch: actual data is chunked into names like `train 0-500`, but
   `SequenceDataset` only reads `train`, `val`, `test`. Either consolidate
   folders or update the dataset loader/config.
3. Docs drift: existing guides mention output paths and tools that do not match
   current scripts.
4. `README.md` says ViT scene embeddings are cached, but active training code
   computes them on the fly and does not use `features/scene_features.py`.
5. `graph_per_sequence.py` does not create default label/ego files, while
   `graph_pipeline.py` does. Use annotation tools before training if using the
   per-sequence pipeline.
6. Raw video count and sequence split organization should be verified before a
   final training run.

## Good Prompts For Future ChatGPT Sessions

Use prompts like:

```text
Here is my codebase handoff document. First identify the exact files involved,
then propose a minimal fix. Do not invent files or commands not listed here.
```

```text
Based on this handoff, fix the TTC loss bug and explain how to verify it.
```

```text
Based on this handoff, update the dataset loader so it can train from chunked
sequence directories like "train 0-500" while keeping val/test behavior.
```

```text
Based on this handoff, write a thesis methodology section explaining the model
architecture and preprocessing pipeline.
```

```text
Based on this handoff, create a clean command checklist for rebuilding the
dataset from raw videos.
```

## Short Architecture Reference

```text
Raw video
  -> frame_extractor
  -> sequence_generator
  -> detection_per_sequence (YOLOv8 frozen)
  -> tracking_per_sequence (ByteTrack)
  -> graph_per_sequence (object + interaction features)
  -> SequenceDataset
  -> AccidentRiskModel
       frames -> FrozenViT -> scene embedding
       graphs -> InteractionGAT -> per-frame graph embeddings
       graph embeddings -> TemporalTransformer -> temporal embedding
       ego_state -> EgoEncoder -> ego embedding
       temporal query attends to scene/GNN/ego in CrossAttentionFusion
       -> classification head + TTC head
```

