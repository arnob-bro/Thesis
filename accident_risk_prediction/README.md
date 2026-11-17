# Rider-Centric Accident Risk Prediction from Helmet-POV Video

Predicts how close a motorcyclist is to a collision, from helmet-camera video.
For every moment of a ride the model outputs a risk level
(0 safe, 1 slight, 2 moderate, 3 high) and an estimated time-to-collision.

Labels come from **collision times marked in the videos**: each 2.4 s sequence
is labelled by how long after its last frame the collision happens, so the
model is trained to *predict*, not to recognise a crash in progress.

## How it works

```
video ──► frames (5 fps)
      ──► YOLO detections (15 fps) ──► remove rider's own bike ──► ByteTrack
      ──► camera-motion estimate (optical flow)        = ego-motion modality
      ──► rider-centric graph per frame (looming, distance, bearing, rider node)
      ──► frozen ViT embedding per frame               = scene modality
                         │  (all cached once per video)
                         ▼
   scene tokens ─┐
   graph tokens ─┼─► Temporal Transformer ─► cross-attention fusion ─► risk (ordinal) + TTC
   ego tokens   ─┘
```

## Setup

```powershell
python -m venv .venv
.venv\Scripts\activate
pip install -r requirements-lock.txt
```

`requirements-lock.txt` holds the exact versions used; `requirements.txt` the minimum versions.

## Workflow

All commands run from this folder. Everything is controlled by `configs/config.yaml`.

### 1. Mark collision times

```powershell
python -m tools.mark_collisions --config configs/config.yaml --videos data/raw_videos
```

Keys: `space` play/pause, `a`/`d` one frame, `j`/`l` one second, `c` collision,
`n` near-miss, `x` loss of control, `[` `]` ignore range, `u` undo, `s` done + next.

- **Collision**: first frame of contact. **Near-miss**: moment of closest approach.
  **Loss of control**: first frame of the skid / fall.
- Mark intros, slow-motion replays and cuts as **ignore ranges**.
- A video saved with no events is valid: all of it is "safe" riding. Include some.
- Prefer typing times? Use `--import-csv times.csv` (`video,time,type`).
- `--list` prints all marked times; `--export-csv` saves them.

### 2. Process the videos (once)

```powershell
python -m pipelines.build_video_cache --config configs/config.yaml --videos data/raw_videos
```

Writes `data/video_cache/<video>/` (frames, detections, tracks, graphs, ViT
embeddings, camera motion). Finished steps are skipped on re-runs. After
changing feature settings: `--steps graphs --force`.

Check what the detector keeps and removes (green = tracked, red = removed as
your own bike, yellow = rider region) and tune `ego_filter` / `ego_box` if needed:

```powershell
python -m tools.visualize_cache --config configs/config.yaml --video-id 12
```

### 3. Build the sequence list

```powershell
python -m tools.make_event_sequences --config configs/config.yaml
```

Writes `data/manifests/sequences.csv`. Re-run it any time you change the
time-to-event bins or mark more videos; it takes seconds. `--dry-run` only
prints the class counts.

| Label | Time before collision |
|---|---|
| 3 high | 0 – 1.5 s |
| 2 moderate | 1.5 – 3 s |
| 1 slight | 3 – 4.5 s |
| skipped | 4.5 – 6 s |
| 0 safe | 6 s or more, or no collision ahead |

### 4. Train and evaluate

```powershell
python train.py --config configs/config.yaml --fold 0          # one run
python -m tools.run_baselines --config configs/config.yaml     # simple baselines, all folds
python -m tools.run_experiments --config configs/config.yaml   # model + ablations, all folds x 3 seeds
```

Folds are assigned **per video**: test = fold k, validation = fold k+1,
train = the rest. Each run writes `runs/<name>/fold<k>_seed<s>/` with the
config used, per-epoch `history.csv`, `best.pt`, test metrics, per-sequence
predictions, a confusion matrix and anticipation metrics.
`run_experiments` produces `runs/summary.md`: mean ± std for every experiment.

### 5. Use a trained model on a new video

```powershell
python inference.py --run runs/full/fold0_seed0 --video path\to\ride.mp4
```

## Metrics

- **Sequence level**: accuracy, macro-F1, recall of risky classes, ordinal
  error (how many classes off), ROC-AUC, TTC error in seconds.
- **Anticipation** (model slid over whole test videos): share of collisions
  warned about in advance, mean time-to-accident (how early), false alarms
  per minute of safe riding, average precision.

## Detector

The pipeline uses **YOLO26x** (`weights/yolo26x.pt`), the most accurate COCO
detector Ultralytics provides. COCO has no rickshaw, CNG or easy-bike class.
A Bangladesh detector trained on the ODS_New dataset (17 classes) replaces it:

```powershell
python -m tools.detector.prepare_ods        # split ../ODS_New by video clip
python -m tools.detector.train_ods          # YOLO26m, then evaluation report
```

Settings are in `configs/detector_ods.yaml`. The report
(`runs/detector/ods_yolo26m/eval_test/report.md`) ends with the lines to paste
into `configs/config.yaml`. The full guide, including labelling your own
helmet frames, is in [docs/detector_guide.md](docs/detector_guide.md).

## Folder map

| Path | Purpose |
|---|---|
| `configs/config.yaml` | all settings |
| `tools/mark_collisions.py` | mark / list collision times |
| `pipelines/build_video_cache.py` | per-video processing |
| `tools/make_event_sequences.py` | collision times → sequence manifest |
| `features/graph_builder.py` | rider-centric graph features, ego-bike filter |
| `features/ego_motion.py` | camera-motion estimation |
| `models/full_model.py` | the model |
| `pipelines/training_pipeline.py` | training, evaluation, anticipation scoring |
| `tools/run_baselines.py`, `tools/run_experiments.py` | baselines, ablations, results table |
| `tools/detector/prepare_ods.py`, `train_ods.py`, `evaluate_detector.py`, `configs/detector_ods.yaml` | Bangladesh detector trained on ODS_New |
| `tools/detector/` (other scripts), `configs/detector.yaml` | labelling your own helmet frames: sample, pre-label, merge, train |
| `tests/test_smoke.py` | quick checks: `python -m tests.test_smoke` |
| `legacy/` | the previous per-sequence pipeline and docs, for reference only |

`legacy/pre_upgrade_snapshot.zip` is a complete copy of the code before the
upgrade. The old hand-labelled sequences in `data/sequences/` and the old
checkpoints in `checkpoints/` only work with that code.
