# Codebase Assessment: Accident Risk Prediction for Motorcyclists (Helmet POV)

Date: 2026-11-16
Scope: `accident_risk_prediction/` (code, config, data under `data/sequences`, checkpoints, logs) and `docs/`.
Method: I read the code, then ran audit scripts over the current `train/val/test` sequence data (labels, ego files, `tracks.json`, `graph.pt`). Every number below comes from the data on disk. Nothing was retrained.

---

## 1. Executive summary

The engineering is solid. The pipeline is modular and config-driven, it has ablation switches, and it has QA visualisers. It runs end to end: video → frames → 12-frame sequences → YOLOv8 → ByteTrack → per-frame graphs → ViT + GAT + Temporal Transformer + Ego MLP → cross-attention fusion → risk class + TTC.

**The current results cannot go in the thesis yet.** Three problems each make the reported numbers unreliable:

1. **Train and test share videos.** All 12 test videos also appear in train. Neighbouring 2.4-second clips from the same ride end up on both sides of the split.
2. **The object graph is mostly the rider's own motorcycle.** About 72% of "motorcycle" tracks and many "person" tracks are wide boxes touching the bottom edge of the frame. That is the ego bike's handlebars or tank, or the rider's own arms. They are not other road users.
3. **The tracker discards 75% of detections.** 29,549 YOLO detections become 7,506 tracked objects. 40% of frames have an empty graph, and the average graph has 1.9 nodes.

There is also a problem with how the thesis is framed. The model classifies how risky the current 2.4 s window looks. It does not predict a future accident. You need to decide which claim the thesis makes (see §5).

Reported test result (`logs/metrics_test.json`): accuracy 0.52, macro-F1 0.47, ROC-AUC 0.79, TTC MAE 1.5 s, on 124 sequences. Problem #1 inflates this result, and it was computed with the wrong checkpoint (see §3.2).

---

## 2. What is good

- **Clear architecture and contracts.** Each modality is its own module (`models/vit`, `models/gnn`, `models/temporal`, `models/ego_state`, `models/fusion`). Tensor shapes are documented, and `configs/config.yaml` is the single source of truth.
- **Ablation switches** (`ablation.use_vit/gnn/temporal/ego`) are wired through `models/full_model.py`, with sensible fallbacks. Your ablation chapter depends on these.
- **Correct graph batching.** `utils/dataset.py:180` builds one PyG `Batch` per timestep, so the GAT runs per frame with proper batch indices.
- **TTC loss masking.** `MultimodalLoss` masks invalid TTC targets. The TTC-loss bug described in `docs/codebase_handoff_for_chatgpt.md` (prediction compared to itself) is **already fixed**: `pipelines/training_pipeline.py:160-163` now passes `batch["ttc"]`. Update that doc.
- **Geometric TTC on edges.** It uses rectangle-overlap TTC under constant velocity, which is better than centre-distance/speed.
- **QA tooling** to visualise detections and graphs per sequence.
- **Labelling guide** (`data dump/label_instruction.md`) with concrete scenarios per class. This is good for annotation consistency and for the thesis appendix.

---

## 3. Critical issues (fix before any reported result)

### 3.1 Data leakage: the split is per sequence, not per video
- `stratified_split.py` pools all sequences, shuffles each class, and re-splits them. This throws away the earlier video-level split (still on disk as `old_train/old_val/old_test`, which had **no** overlap).
- Current overlap: train/test share **12 of 12** test videos, train/val share 15, val/test share 12.
- Sequences are adjacent, non-overlapping 12-frame windows (for example `10_seq0001` is in train and `10_seq0002` is in test). Same rider, road, lighting, traffic and often the same event, so the test set measures memorisation.
- `config.yaml` even says `# Split (by video, not by frame)`, but the code does not follow it.

**Fix:** split by video. Only **17 labelled videos** exist, so one fixed split will be very noisy. Use **grouped k-fold cross-validation over videos** (e.g. 5 folds, `sklearn.model_selection.StratifiedGroupKFold` with `groups=video_id`) and report mean ± std.

### 3.2 Test metrics come from the last epoch, not the best checkpoint
`Trainer.fit()` (`pipelines/training_pipeline.py:279-282`) evaluates the test split with the model in memory after the loop. With early stopping (patience 10), that is the model 10 epochs **after** the best one. `checkpoints/` has `epoch_001…epoch_022`, so best was around epoch 12 and the test JSON was produced at epoch 22.
**Fix:** reload `best.pt` before the final test evaluation, or always report from `evaluate.py --checkpoint checkpoints/best.pt`.

### 3.3 The ego motorcycle and rider are being detected as road users (helmet-POV specific)
Measured on `data/sequences/train/*/tracks.json` (frame size 640×384):
- Boxes that touch the bottom 5% of the frame and are wider than 25% of the frame: **1,906 of 7,506 tracks (25%)**.
  - Class `motorcycle`: **1,428 of 1,974** motorcycle tracks (72%). This is almost certainly the rider's own handlebars, tank or dashboard.
  - Class `person`: 375. Probably the rider's own arms or knees.
- These "objects" never move relative to the camera, so they add constant nodes and edges that look like near-collisions (high proximity, overlaps). They swamp the real interactions.

**Fix (cheap):** apply a fixed ego-vehicle mask before graph building, i.e. drop detections whose box lies mostly in a bottom-centre region. Check it visually with `tools/visualize_detections_per_sequence.py`. **Better:** train or fine-tune a detector with an explicit "ego-vehicle" class to ignore.

### 3.4 Tracking keeps only 25% of detections; graphs are nearly empty
- 29,549 detections become 7,506 tracked points in train.
- 2,648 of 6,684 frames (40%) have **no tracked object**. Mean nodes per frame is 1.86, max 7. That is not a traffic graph for dense South-Asian traffic, and once the ego bike is removed (§3.3) it gets worse.
- Causes:
  - `tracking_per_sequence.py` restarts ByteTrack for **every 12-frame sequence**, so tracks have to re-activate each time and the first frames lose objects.
  - `track_thresh: 0.5` is higher than the YOLO `conf_threshold: 0.25`, and yolov8n gives low confidences on Bangladeshi traffic.
  - Video at 5 FPS from a moving head gives large frame-to-frame displacement, which hurts IoU matching.

**Fix:** track over the **whole video** first (the older `tracking_pipeline.py` already does this), then cut the tracks into sequences. Lower `track_thresh` to about 0.3. Consider 10 FPS for tracking even if you subsample later.

---

## 4. Significant issues (affect validity or quality)

### 4.1 Feature scaling saturates
`features/object_features.py` clamps velocity to ±50 px/s and acceleration to ±20 px/s², then divides by the same constants.
- At 5 FPS, 50 px/s is only 10 px per frame. **22% of velocity values are saturated** at ±1.
- **70% of acceleration values are saturated.** The acceleration feature is effectively just `sign(Δspeed)`. Another 20% are exactly zero (first frame of each track).

**Fix:** pick clamps from data percentiles (e.g. the 99th), or normalise velocity by box size (px/s ÷ box height), which also partly removes distance effects.

### 4.2 No ego-centric or ego-motion handling, which is central for helmet POV
- **Ego-motion.** The camera moves with the rider's head: yaw from looking around, roll from leaning, pitch from bumps. Image-space velocities therefore mix object motion with head motion. Nothing compensates for this (no global optical flow or homography subtraction).
- **Ego-relative risk.** The rider is the potential victim, but the graph has only object↔object edges (`include_ego_node: false`). `compute_ego_object_interactions()` exists but is never used.
- **Missing looming cue.** The standard monocular collision cue is box expansion: τ ≈ h / (dh/dt), the scale-change TTC. It needs no depth or speed and is exactly what helmet footage supports. It is not computed.

**Fix:**
- Add ego-relative node features: distance to bottom-centre, bearing, scale-change rate, τ.
- Or add an ego node connected to all objects. This is a strong, easy-to-explain contribution for the thesis.
- Optionally subtract the median optical flow of background pixels from track velocities.

### 4.3 Ego telemetry modality is weak and unnormalised
- `brake` is **0.0 in every sequence**, so it is a dead feature.
- `speed` and `rpm_estimate` are hand-entered (values such as 3000–11000 rpm). On a helmet cam the speedometer is rarely visible, so state how these values were obtained. If they were guessed by the annotator who also assigned the risk label, they may encode the label.
- The values go into the MLP **without normalisation**: rpm is about 10⁴ while speed is about 10¹, so rpm dominates the first linear layer.
- A few ego files have inconsistent list lengths (9, 11 or 13 instead of 12). The loader silently pads or truncates them.
- Speed differs by class (mean 40 for safe, 57 for high), so it carries signal. Make sure the ablation shows how much of the gain comes from this hand-entered input.

**Fix:** z-score the ego inputs (using train statistics), drop `brake`, document how the values were acquired, and run "with vs without ego" as a key ablation.

### 4.4 The TTC target is coarse and partly derived from the label
- `ttc_estimate` is an integer guess per sequence. Class 0 almost always uses the sentinel **6** (205 of 218 in train).
- Mean TTC per class goes 6.0 → 2.3 → 1.5 → 1.3, so TTC mostly re-encodes the class. "TTC MAE = 1.5 s" is largely driven by whether the class is right.
- `utils/metrics.py` calls this value `time_to_accident` (TTA). In the literature, TTA means something different (how early before an accident the model fires).

**Fix:** either report TTC only on risky classes with a clear note on how it was annotated, or drop it as a target and present it as an auxiliary loss. Rename the metric to `ttc_mae`.

### 4.5 Some reported metrics are placeholders
- `early_risk_detection` is **always 0.0**. `compute_all_metrics` only computes it when `y_true_seq`/`y_pred_seq` are passed, and `Trainer.evaluate` never passes them.
- There is no frame- or video-level evaluation. Every metric is per isolated sequence.

### 4.6 The model is large for the data
- 557 training sequences. Class 3 ("high") has **21 training and 6 test examples**.
- Rough trainable-parameter estimate: about 30M. That breaks down as a 4-layer d=512 temporal transformer (~8M), cross-attention fusion with 3×2 blocks (~19M), and heads. Overfitting is very likely.
- `class_weights: null`, so nothing handles the imbalance.

**Fix:** shrink the model (d_model 128–256, 1–2 layers), use class weights or focal loss, and add augmentation (horizontal flip with x-mirroring of graph features, colour jitter). Report per-class recall for "high".

### 4.7 Architecture details worth knowing before you write the method chapter
- **Two of the three cross-attention branches are not really attention.** The ViT and ego branches have a **single key token** (`[B,1,d]`). Softmax over one key always gives weight 1, so those branches reduce to a linear transform of the key plus an FFN. Only the GNN branch (over up to 32 nodes) actually attends. Either give them more tokens (per-frame ViT CLS tokens `[B,T,768]`, per-timestep ego `[B,T,64]`) or describe the fusion honestly.
- **ViT is averaged over time** (`_encode_scene`), so scene dynamics never reach the temporal model. The Temporal Transformer only sees pooled GNN vectors, and those are empty 40% of the time (§3.4).
- **The GNN branch in fusion uses only the last frame's nodes.**
- **ViT runs on every epoch** even though it is frozen. Caching the embeddings once (`features/scene_features.py` already has helpers) would make training several times faster and make k-fold CV practical.
- **Rickshaws are never detected.** `rickshaw` is in the class map but COCO has no such class, and CNG/auto-rickshaws and easy-bikes are mislabelled or missed. For Bangladeshi roads this is a real gap, and yolov8n is the weakest YOLOv8 variant.

---

## 5. Research-design issues for the thesis

1. **Prediction vs recognition.** Labels describe the risk *visible in the current window*. To claim *prediction*, labels must refer to the future, e.g. "an accident or near-miss occurs within the next N seconds". Alternatively, train on window *t* with the label of window *t+1*. Otherwise call it "risk assessment" or "risk recognition". Examiners will ask about this.
2. **Dataset size and diversity.** Only 17 videos are labelled and one video (`16`) contributes 132 of 557 training sequences. `data/raw_videos` holds 168 videos and `data dump/` holds about 5,000 more sequences that look prepared for labelling. Scaling labelled videos is the single biggest improvement available.
3. **Label reliability.** One annotator? Report inter-annotator agreement (Cohen's κ) on a double-labelled subset. "Slight" vs "moderate" will be the hardest boundary, and the confusion matrix confirms it.
4. **Baselines are missing.** Without them nobody can tell whether the multimodal model helps. Add:
   - a majority-class baseline;
   - frozen ViT + logistic regression or MLP on mean CLS;
   - ViT per-frame + GRU;
   - hand-crafted features (object count, min τ, max proximity, ego speed) + gradient boosting.
5. **Ablations with variance.** Run every ablation over k folds × ≥3 seeds and report mean ± std. With 124 test items, ±5% swings are noise.
6. **Positioning.** Cite and compare against helmet- or ego-view anticipation datasets and methods (e.g. DAD, CCD, DoTA for dashcams; Motorcycle-specific near-miss work). State clearly what is new: helmet POV, Bangladeshi mixed traffic, motorcycle-specific risk.

---

## 6. Code hygiene and reproducibility

- `__pycache__/*.pyc` files are tracked in git. Add a `.gitignore` for `__pycache__/`, `.venv/`, `data/`, `checkpoints/`, `logs/`, `*.pt`.
- `requirements.txt` uses only `>=` bounds. Pin exact versions (`pip freeze > requirements-lock.txt`) so the thesis results can be reproduced.
- `utils/dataset.py:108`: a label with `"ttc_estimate": null` would crash (`float(None)`). Not triggered by current data, but the documented schema allows null.
- `stratified_split.py`, `count_class_values_per_split` (no `.py`) and `check_json.py` sit at the project root as one-off scripts. Move them to `tools/`.
- Stale folders `data/sequences/old_*`, `new/accident_risk_prediction/` and a duplicate zip make it unclear which data is current.
- Docs drift: `README.md` says ViT embeddings are cached (they are not), and `docs/to_visualize_and_correct_yolov8n_detections.md` references `export_yolo_labels` / `import_yolo_labels`, which do not exist.
- `logs/train.log` has no per-epoch losses, because the training-pipeline logger writes only to the console. You will need training curves for the thesis, so log to file or TensorBoard/CSV.
- There are no tests. A few shape tests (dataset → collate → model forward on 2 fake sequences) would catch regressions cheaply.

---

## 7. Prioritised action plan

| Priority | Action | Where |
|---|---|---|
| **P0** | Video-level grouped k-fold split; delete the sequence-level split | `stratified_split.py`, `utils/dataset.py` |
| **P0** | Evaluate `best.pt`, not the last epoch | `training_pipeline.py:279` |
| **P0** | Mask out the ego motorcycle and rider detections | new filter before `graph_per_sequence.py` |
| **P0** | Track per video, then cut into sequences; lower `track_thresh` | `tracking_pipeline.py`, `config.yaml` |
| **P1** | Add ego-relative and looming (τ) features, or an ego node | `features/`, `graph_per_sequence.py` |
| **P1** | Data-driven feature clamps; normalise ego inputs; drop `brake` | `object_features.py`, `dataset.py` |
| **P1** | Class weights or focal loss; smaller model; cache ViT embeddings | `config.yaml`, `full_model.py` |
| **P1** | Baselines + ablations with k-fold × seeds; per-epoch logging | `train.py`, new scripts |
| **P2** | Decide prediction vs recognition; relabel with a future horizon if "prediction" | labelling protocol |
| **P2** | Label more videos (168 raw available); measure inter-annotator κ | data |
| **P2** | Fine-tune the detector on local classes (rickshaw, CNG, easy-bike, ego-vehicle) | `models/yolo` |
| **P2** | Real early-detection metric over consecutive windows; rename TTA → TTC MAE | `utils/metrics.py` |
| **P3** | `.gitignore`, pinned deps, tidy scripts, fix docs drift, smoke tests | repo |

---

## 8. Bottom line

The codebase is a capable research scaffold. Most of the modelling machinery a thesis needs is already there. The weak points are the **data pipeline and evaluation protocol**, and some of them are specific to helmet footage: the ego bike being detected, head motion, and the lack of ego-centric cues.

Fix the four P0 items and re-run with grouped cross-validation. The numbers will probably drop, but they will be defensible, and every later improvement (ego-centric features, more videos) can then be measured honestly.

---

## 9. Status after the upgrade (2026-11-17)

The codebase was restructured around collision-time labelling. What changed against this report:

| Issue | Status |
|---|---|
| §3.1 Train/test share videos | Fixed in code: folds and splits are assigned per video (`utils/windows.py`, `utils/events.py`). |
| §3.2 Test metrics from last epoch | Fixed: `best.pt` is reloaded before every evaluation. |
| §3.3 Rider's own bike detected | Mitigated: `ego_filter` drops bottom-edge wide motorcycle/person boxes. A fine-tuned detector with an `ego_vehicle` class is still the proper fix. |
| §3.4 Tracker drops detections | Improved: tracking runs once per video at 15 fps with YOLO26x. On 7 test videos this tracks about 3–4 times as many objects as the original setup. About half of the remaining detections are still not tracked, but most of those are tiny distant objects (median ~19×19 px) or low-confidence boxes, which matter little for collision risk. |
| §4.1 Feature saturation | Fixed: log-scaled velocities, no hard clamp. |
| §4.2 No ego-centric / ego-motion handling | Fixed: rider node, distance, bearing, looming, head-motion compensation. |
| §4.3 Ego telemetry | Replaced by camera-motion features estimated from the video. |
| §4.4 Coarse TTC target | Fixed: exact time-to-event from marked collision times. |
| §4.5 Placeholder metrics | Fixed: real anticipation metrics from sliding-window evaluation. |
| §4.6 Model too large | Fixed: about 1.1 M trainable parameters; class weighting; flip augmentation. |
| §4.7 Fusion / temporal design | Fixed: per-frame tokens for every modality; ViT embeddings cached; ordinal head. |
| §5.1 Prediction vs recognition | Fixed by the labelling rule (time measured from the sequence's last frame). |
| §5.4–5.5 Baselines, ablations | Tools added (`tools/run_baselines.py`, `tools/run_experiments.py`). Not yet run on real labels. |
| §6 Hygiene | `.gitignore`, pinned dependencies, smoke tests, new README; old code in `legacy/`. |

**Still open (needs you):** marking collision times on the real videos, labelling
frames and fine-tuning the Bangladesh detector, inter-annotator agreement, and
running the experiments. No result on real labels exists yet for the new pipeline.
