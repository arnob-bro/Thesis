# Accident Risk Prediction — Complete Project Guide

> **Approach:** Run all pipeline stages first, then annotate sequences, then train.

---

## Prerequisites

| Component        | Your Setup       |
| ---------------- | ---------------- |
| Python           | 3.13             |
| PyTorch          | 2.12.0+cu126 ✅  |
| GPU              | RTX 3060 12GB ✅ |
| CPU              | i5-13400F ✅     |
| All dependencies | Installed ✅     |

---

## Step 0 — Prepare Raw Videos

Place your dashcam videos in `data/raw_videos/` organized by split:

```
data/raw_videos/
  train/
    1.mp4
    2.mp4
    ...
  val/
    3.mp4
  test/
    4.mp4
```

> [!TIP]
> Split roughly 70% train / 15% val / 15% test **by video** (never by frame).

---

## Step 1 — Extract Frames

Extract frames at 5 FPS from each video and store them by split.

```powershell
python -m pipelines.frame_extractor --config configs/config.yaml --split train
python -m pipelines.frame_extractor --config configs/config.yaml --split val
python -m pipelines.frame_extractor --config configs/config.yaml --split test
```

**Output:** `data/extracted_frames/<split>/<video_name>/frame_0001.jpg, frame_0002.jpg, ...`

**Verify:**

```powershell
ls data/extracted_frames/train/
```

---

## Step 2 — Generate Sequences

Split extracted frames into fixed-length sequence folders and preserve original frame filenames.

```powershell
python -m pipelines.sequence_generator --config configs/config.yaml --split train
python -m pipelines.sequence_generator --config configs/config.yaml --split val
python -m pipelines.sequence_generator --config configs/config.yaml --split test
```

**Output:** `data/sequences/<split>/<video>_seq<N>/`

```
data/sequences/train/1_seq0000/
    frames/          # 12 copied frames preserving original filenames
    metadata.json    # sequence metadata with frame_ids, frame_names, fps, image_size
```

> [!NOTE]
> Preserving original frame filenames is important for exact frame matching during later visualization and graph alignment.

---

## Step 3 — YOLOv8 Object Detection

Run YOLOv8 on each extracted frame.

```powershell
python -m pipelines.detection_per_sequence --config configs/config.yaml --split train
python -m pipelines.detection_per_sequence --config configs/config.yaml --split val
python -m pipelines.detection_per_sequence --config configs.config.yaml --split test
```

**Output:** `data/annotations/detections/<split>/<video_name>/` (per-frame detection JSONs)

> [!NOTE]
> This stage uses your GPU (`yolo.device: cuda` in config). The first run downloads `yolov8n.pt` automatically.

---

## Step 4 — ByteTrack Tracking

Link detections across frames to form object tracks.

```powershell
python -m pipelines.tracking_per_sequence --config configs/config.yaml --split train
python -m pipelines.tracking_per_sequence --config configs/config.yaml --split val
python -m pipelines.tracking_per_sequence --config configs/config.yaml --split test
```

**Output:** `data/annotations/tracking/<split>/<video_name>/tracks.json`

---

## Step 5 — Graph Construction

Build interaction graphs from tracked objects and save them inside each sequence.

```powershell
python -m pipelines.graph_per_sequence --config configs/config.yaml --split train
python -m pipelines.graph_per_sequence --config configs/config.yaml --split val
python -m pipelines.graph_per_sequence --config configs/config.yaml --split test
```

**Output:** Updated sequence folders at `data/sequences/<split>/<video>_seq<N>/`:

```
data/sequences/train/1_seq0000/
    frames/
    graph.pt         # per-frame interaction graphs
    label.json       # defaults to risk_level=0 until annotated
    ego_state.json   # ego vehicle telemetry or zeros if unavailable
    metadata.json    # sequence metadata including frame_names for exact frame matching
```

**How many sequences per video:**

- Total frames ÷ 12 (sequence length) = number of sequences
- Example: 339 frames → 28 sequences (seq0000–seq0027)
- Each sequence covers 2.4 seconds of video (12 frames ÷ 5 FPS)

**Detection visualization:**

```powershell
python -m tools.visualize_detections_per_sequence --split train --sequence 2_seq0000
```

**Graph QA visualization:**

```powershell
python -m tools.visualize_graphs --config configs/config.yaml --split train --sequence 1_seq0000
```

This tool now resolves exact original frame filenames from `metadata.json` and avoids duplicate numeric frame-ID ambiguity.

---

## Step 6 — Ego-State Audit

Verify ego-state files are present and correctly sized.

```powershell
python -m pipelines.ego_state_pipeline --config configs/config.yaml --split train
python -m pipelines.ego_state_pipeline --config configs/config.yaml --split val
python -m pipelines.ego_state_pipeline --config configs/config.yaml --split test
```

**Output:** `data/metadata/ego_state_audit_<split>.csv`

> [!NOTE]
> If you don't have ego-state telemetry, this stage fills zeros. Training still works, but ego-based features will be weaker. Disable it via `ablation.use_ego: false` in config.

---

## Step 7 — ✍️ Annotate Sequences

Label each sequence with a risk level and optional time-to-collision estimate.

### 7.1 Browse sequences

```powershell
ls data/sequences/train/
ls data/sequences/train/1_seq0000/frames/
```

### 7.2 Edit `label.json`

Each file begins as:

```json
{
  "sequence_id": "1_seq0000",
  "risk_level": 0,
  "ttc_estimate": NaN
}
```

Update based on the 2.4-second clip.

| risk_level | Meaning  | When to use                          |
| ---------- | -------- | ------------------------------------ |
| **0**      | Safe     | Normal driving, no hazards           |
| **1**      | Slight   | Minor risk, distant obstacles        |
| **2**      | Moderate | Approaching hazard, close proximity  |
| **3**      | High     | Imminent collision or very dangerous |

| Field          | Description                                                                                                      |
| -------------- | ---------------------------------------------------------------------------------------------------------------- |
| `ttc_estimate` | Estimated seconds until collision. Use `NaN` for safe sequences. For risky sequences, provide seconds to impact. |

**Example — safe sequence:**

```json
{ "sequence_id": "1_seq0000", "risk_level": 0, "ttc_estimate": NaN }
```

**Example — high-risk sequence:**

```json
{ "sequence_id": "1_seq0015", "risk_level": 3, "ttc_estimate": 0.8 }
```

> [!IMPORTANT]
> Annotate sequences for **train, val, and test**. Train on train labels, validate on val labels, and evaluate on test labels.

### 7.3 Map to original video time

Use `metadata.json` for frame-to-time mapping:

```json
{
  "start_frame": 1,
  "end_frame": 12,
  "fps": 5
}
```

- `start_frame=1, end_frame=12` at 5 FPS → **0.0s–2.4s**
- `start_frame=13, end_frame=24` → **2.4s–4.8s**

---

### 7.4 Bulk annotation helper

To help annotate many sequences at once the repository includes a helper script `tools/write_sequence_annotations.py`.

- Purpose: write `label.json` and/or `ego_state.json` into existing sequence folders from CSV/JSON sources.
- Location: `tools/write_sequence_annotations.py`

Usage examples:

```powershell
# Write labels from CSV (columns: sequence_id,risk_level,ttc_estimate)
python -m tools.write_sequence_annotations --config configs/config.yaml --split train --labels-csv labels.csv

# Import per-sequence ego files (.csv or .json) from a directory
python -m tools.write_sequence_annotations --config configs/config.yaml --split train --ego-dir path/to/ego_files

# Combine both, preview first with dry-run
python -m tools.write_sequence_annotations --config configs/config.yaml --split all --labels-csv labels.csv --ego-dir ego_dir --dry-run

# Overwrite existing files
python -m tools.write_sequence_annotations --config configs/config.yaml --split train --labels-csv labels.csv --overwrite
```

Notes:

- The script searches for sequence folders under `data/sequences/<split>/` by exact `sequence_id` or substring match.
- If `ttc_estimate` is missing or `NaN` in the CSV, the script writes `ttc_estimate: NaN` and the dataset will treat the TTC as invalid for loss masking.
- Use `--dry-run` to preview changes without writing files.

---

### 7.5 Initialize missing annotations & training compatibility

The helper script and the graph pipeline both support initializing missing
annotation files so training can run on the per-sequence layout.

- If a sequence has no `label.json`, the graph pipeline will create a default
  `label.json` (risk_level=0, `ttc_estimate`: NaN) when materialising sequences
  during `pipelines/graph_per_sequence.py`.
- If a sequence has no `ego_state.json`, the graph pipeline writes a zero-filled
  `ego_state.json` aligned to the sequence length. The `tools/write_sequence_annotations.py`
  script can also create or convert ego CSV/JSON files into `ego_state.json`.

Quick initialization options:

```powershell
# Run per-sequence graph construction which materialises default label/ego
python -m pipelines.graph_per_sequence --config configs/config.yaml --split train

# Or explicitly write labels and ego files with the helper script
python -m tools.write_sequence_annotations --config configs/config.yaml --split train --labels-csv labels.csv --ego-dir ego_dir
```

Training compatibility notes:

- The training loader `SequenceDataset` expects each sequence folder to contain
  `frames/`, `graph.pt`, `label.json`, `ego_state.json`, and `metadata.json`.
- `SequenceDataset._load_label()` reads `label.json` and uses the field
  `ttc_estimate` (or `ttc` fallback). If `ttc_estimate` is `NaN`, the dataset
  marks the TTC as invalid and the TTC loss is masked during training.
- `_load_ego_state()` loads `ego_state.json`, pads or truncates the series to
  the configured sequence length, and returns a `[T, D_ego]` tensor for the
  Ego encoder.

Therefore, either run the graph pipeline to materialise defaults or use the
helper script to write labels/ego files before running `train.py`.

## Additional helper: quick init for editable annotation files

If you want to create editable `label.json` and `ego_state.json` placeholders
for all sequences (so annotators can open and edit them later), use the
convenience initializer script:

```powershell
python -m tools.init_sequence_annotations --config configs/config.yaml --split train
```

Options:

- `--dry-run`: preview what would be created without writing files
- `--overwrite`: replace existing `label.json` / `ego_state.json`
- `--open`: try to open sequence folders after creating files (Windows only)

What it does:

- Writes `label.json` with `risk_level: 0` and `ttc_estimate: NaN` when missing.
- Writes `ego_state.json` filled with zeros for the configured ego feature keys.
- Leaves existing files untouched unless `--overwrite` is provided.

This script is handy for preparing large sets of sequences for manual annotation
and complements `tools/write_sequence_annotations.py` which imports existing
CSV/JSON label sources.

---

## Step 8 — Train

```powershell
python train.py --config configs/config.yaml
```

**What happens:**

- Loads sequences from `data/sequences/{train,val}/`
- Computes ViT scene embeddings on-the-fly and caches them to `data/cache/embeddings/`
- Saves checkpoints to `checkpoints/`
- Best model saved to `checkpoints/best.pt`
- Logs saved to `logs/`

**Key config settings** ([configs/config.yaml](file:///d:/Thesis/accident_risk_prediction/configs/config.yaml)):

```yaml
training:
  epochs: 50
  batch_size: 12
  learning_rate: 0.0001
  early_stop_patience: 10
```

**Resume training:**

```powershell
python train.py --config configs/config.yaml --resume checkpoints/epoch_10.pt
```

> [!WARNING]
> If you get CUDA OOM errors, reduce `batch_size` to `8` or `4` in `configs/config.yaml`.

---

## Step 9 — Evaluate

```powershell
python evaluate.py --config configs/config.yaml --checkpoint checkpoints/best.pt
```

**Validate on val split:**

```powershell
python evaluate.py --config configs/config.yaml --checkpoint checkpoints/best.pt --split val
```

**Output:**

- `logs/metrics_test.json` — accuracy, precision, recall, F1, ROC-AUC, TTA, early risk detection score
- `logs/confusion_test.png` — confusion matrix visualization

---

## Step 10 — Inference (Single Video)

```powershell
python inference.py --config configs/config.yaml --checkpoint checkpoints/best.pt --video path/to/new_clip.mp4 --ego_state path/to/ego.csv --output prediction.json
```

This runs the full pipeline on a new video and outputs per-sequence risk predictions.

---

## Quick Reference — All Commands

```powershell
# SETUP
python -m venv .venv
.venv\Scripts\activate
pip install torch torchvision --index-url https://download.pytorch.org/whl/cu126
pip install torch-geometric
pip install ultralytics supervision transformers opencv-python Pillow numpy scipy pandas PyYAML tqdm scikit-learn matplotlib

# PIPELINE
python -m pipelines.frame_extractor --config configs/config.yaml --split train
python -m pipelines.sequence_generator --config configs/config.yaml --split train
python -m pipelines.detection_pipeline --config configs/config.yaml --split train
python -m pipelines.tracking_pipeline --config configs/config.yaml --split train
python -m pipelines.graph_pipeline --config configs/config.yaml --split train
python -m pipelines.ego_state_pipeline --config configs/config.yaml --split train

# ANNOTATE
# edit data/sequences/train/*/label.json manually

# TRAIN
python train.py --config configs/config.yaml

# EVALUATE
python evaluate.py --config configs/config.yaml --checkpoint checkpoints/best.pt

# INFERENCE
python inference.py --config configs/config.yaml --checkpoint checkpoints/best.pt --video clip.mp4 --ego_state ego.csv --output prediction.json
```

---

## Ablation Experiments

Toggle modalities in `configs/config.yaml` to test their contribution:

```yaml
ablation:
  use_vit: true
  use_gnn: true
  use_temporal: true
  use_ego: true
```

Set any to `false` and retrain to measure impact.
