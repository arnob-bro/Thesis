# 📦 Dataset Documentation — Multimodal Accident Risk Prediction System

# 📌 Objective

Design a research-grade dataset structure for:

* accident risk prediction
* temporal reasoning
* interaction modeling
* multimodal learning

The dataset must support:

* object detection
* object tracking
* graph construction
* scene understanding
* temporal sequence learning
* early accident prediction

---

# 🏗️ Recommended Dataset Structure

```id="2drj35"
dataset/
│
├── raw_videos/
│   ├── train/
│   ├── val/
│   └── test/
│
├── extracted_frames/
│   ├── train/
│   ├── val/
│   └── test/
│
├── annotations/
│   ├── detections/
│   ├── tracking/
│   ├── risk_labels/
│   └── sequences/
│
├── features/
│   ├── object_features/
│   ├── interaction_features/
│   ├── scene_embeddings/
│   └── graph_embeddings/
│
├── graphs/
│
├── cache/
│   ├── detections/
│   ├── tracks/
│   ├── embeddings/
│   └── temporal/
│
└── metadata/
    ├── dataset_info.json
    ├── class_mapping.json
    └── splits.json
```

---

# 🎥 1️⃣ Raw Video Dataset

## Folder:

```id="6u49cl"
raw_videos/
```

## Purpose:

Stores original videos before preprocessing.

---

# Recommended Naming Convention

```id="ixn3o8"
video_<location>_<weather>_<id>.mp4
```

Example:

```id="v3x4ci"
video_dhaka_sunny_001.mp4
```

---

# Required Diversity

Dataset MUST include:

* urban roads
* rural roads
* highways
* intersections
* dense traffic
* sparse traffic
* overtaking
* sudden braking
* near misses
* pedestrians
* motorcycles
* bicycles
* night driving
* rain/fog (optional)

---

# ⚠️ IMPORTANT

DO NOT collect only safe driving videos.

Dataset MUST contain:

* risky interactions
* unsafe proximity
* abrupt motion
* chaotic traffic behavior

---

# 🖼️ 2️⃣ Extracted Frames

## Folder:

```id="yd8d11"
extracted_frames/
```

## Frame Extraction Rules

### Recommended FPS:

* 5–10 FPS

---

## Naming Convention

```id="mwl9vv"
frame_<videoid>_<frame_number>.jpg
```

Example:

```id="5b1ujg"
frame_001_000123.jpg
```

---

# 📦 3️⃣ Detection Annotations

## Folder:

```id="zc76v6"
annotations/detections/
```

---

# Format (YOLO)

Each frame should have:

```id="a6fgzn"
<class_id> <x_center> <y_center> <width> <height>
```

Example:

```id="n8kt1l"
0 0.51 0.48 0.20 0.18
1 0.22 0.63 0.08 0.12
```

---

# Recommended Classes

```id="ezs8zj"
0 = car
1 = motorcycle
2 = bicycle
3 = pedestrian
4 = bus
5 = truck
6 = rickshaw
```

---

# 🚗 4️⃣ Tracking Annotations

## Folder:

```id="ff6u9w"
annotations/tracking/
```

---

# Required Fields

```id="dlm0yq"
frame_id,
track_id,
class_name,
bbox_x,
bbox_y,
bbox_w,
bbox_h
```

---

# Important Constraints

Tracking MUST:

* preserve object identity
* avoid frequent ID switching
* maintain temporal consistency

---

# 🧠 5️⃣ Object Feature Dataset

## Folder:

```id="u9s9d6"
features/object_features/
```

---

# Features Per Object

Store:

* position
* bbox size
* velocity
* acceleration
* direction angle
* object type

---

# Example Format

```id="r2yd5p"
{
  "track_id": 17,
  "velocity": 5.4,
  "acceleration": 1.2,
  "direction": 32.1
}
```

---

# 🚘 6️⃣ Ego Vehicle State Dataset

## Folder:

```text
features/ego_state/
```

---

# Purpose

Store manually-provided dashboard telemetry synchronized with video frames.

This modality represents:

* ego vehicle dynamics
* internal vehicle behavior
* structured driving signals

---

# Supported Features

Recommended:

* speed
* rpm
* acceleration (optional)
* brake_state (optional)

---

# Example Format

```json
{
  "frame_id": 123,
  "timestamp": 12.4,
  "speed": 48,
  "rpm": 4200
}
```

---

# IMPORTANT

Dashboard values MUST:

* remain synchronized with extracted frames
* preserve temporal ordering

---

# Sequence Example

```text
Frame 001 → speed = 40
Frame 002 → speed = 44
Frame 003 → speed = 47
```

---

# Storage Recommendation

Store as:

```text
CSV
JSON
Parquet
```

---

# Recommended File Structure

```text
features/
└── ego_state/
    ├── sequence_001.csv
    ├── sequence_002.csv
    └── sequence_003.csv
```

---

# Example CSV

```text
frame_id,timestamp,speed,rpm
1,0.0,42,3800
2,0.2,44,3900
3,0.4,47,4100
```

---

# Why Ego-State Data Matters

Risk depends not only on:

* surrounding traffic

but also:

* ego vehicle behavior

Example:

```text
Close interaction + high speed
=
much higher collision probability
```

---

# Multimodal Learning Justification

This dataset is multimodal because it combines:

| Modality    | Representation               |
| ----------- | ---------------------------- |
| Scene       | visual context               |
| Interaction | graph relationships          |
| Temporal    | motion evolution             |
| Ego State   | structured vehicle telemetry |

---

# Important Assumption

Ego-state data is treated as:

* structured numerical modality

NOT:

* image modality
* OCR modality
* visual embedding


# 🔗 7️⃣ Interaction Feature Dataset

## Folder:

```id="h2g8g5"
features/interaction_features/
```

---

# Features Per Interaction Edge

For every object pair:

* distance
* relative velocity
* bearing angle
* time-to-collision (TTC)

---

# Example

```id="0y43ol"
{
  "source_track": 17,
  "target_track": 22,
  "distance": 14.3,
  "relative_velocity": 4.2,
  "ttc": 3.1
}
```

---

# 🧬 8️⃣ Graph Dataset

## Folder:

```id="z7xv8m"
graphs/
```

---

# Graph Structure

Each frame graph should contain:

* nodes = tracked objects
* edges = interactions

---

# Recommended Graph Constraints

## DO NOT:

* fully connect all objects

## MUST:

* use nearest-neighbor graph
  OR
* distance threshold graph

Recommended:

* K = 3–5 nearest neighbors

---

# 🌍 9️⃣ Scene Embedding Dataset

## Folder:

```id="g9cloz"
features/scene_embeddings/
```

---

# Source:

Generated using Vision Transformer (ViT)

---

# Embedding Storage

Store embeddings as:

```id="v3j2m4"
.npy
```

Example:

```id="3u4mcc"
scene_00123.npy
```

---

# ⏱️ 🔟 Temporal Sequence Dataset

## Folder:

```id="v0spq4"
annotations/sequences/
```

---

# Sequence Rules

Each sequence:

* fixed temporal length
* ordered frames
* preserved object IDs

---

# Recommended Sequence Length

```id="fsf49m"
8–16 frames
```

---

# Example

```id="hjn04q"
sequence_00012/
├── frame_001.jpg
├── frame_002.jpg
├── frame_003.jpg
└── metadata.json
```

---

# 🚨 1️⃣1️⃣ Risk Label Dataset

## Folder:

```id="wt0j1l"
annotations/risk_labels/
```

---

# Risk Levels

```id="sy7ghf"
0 = Safe
1 = Slight Risk
2 = Moderate Risk
3 = High Risk
```

---

# IMPORTANT

Labels MUST correspond to:

* temporal sequences
  NOT
* isolated frames

---

# Example Label File

```id="eh3cya"
{
  "sequence_id": "00012",
  "risk_level": 2,
  "ttc": 1.8,
  "notes": "Unsafe overtaking"
}
```

---

# ⚠️ 1️⃣2️⃣ Dataset Bias Prevention

Dataset MUST be balanced.

---

# Required Distribution

Include:

* safe scenes
* moderate risk scenes
* high-risk scenes

---

# DO NOT:

* overpopulate dataset with normal driving only

This causes:

* prediction collapse
* severe class imbalance

---

# 📈 1️⃣3️⃣ Train / Validation / Test Split

## Recommended Split

```id="4bav0f"
70% Train
15% Validation
15% Test
```

---

# IMPORTANT

Split MUST occur:

* by video
  NOT
* by frame

---

# Why?

Otherwise:

* same scene leaks into train/test
* evaluation becomes invalid

---

# 🧪 1️⃣4️⃣ Required Evaluation Scenarios

Test dataset should include:

* intersections
* lane changes
* overtaking
* dense traffic
* sudden stops
* pedestrians crossing
* motorcycles near vehicles

---

# ⚡ 1️⃣5️⃣ Caching Strategy

Cache:

* detections
* tracks
* embeddings
* graph tensors

---

# Example Cache Structure

```id="vjlwmj"
cache/
├── yolo/
├── tracking/
├── vit/
└── graphs/
```

---

# 🔒 1️⃣6️⃣ Reproducibility Requirements

Dataset pipeline MUST:

* preserve frame order
* preserve object IDs
* save preprocessing configs
* log extraction parameters

---

# 🚀 1️⃣7️⃣ Final Dataset Objective

The dataset should enable the model to:

* understand scene context
* reason about interactions
* model motion over time
* predict accidents BEFORE collision

---

# 💯 Final Recommendation

This dataset is fundamentally:

* multimodal
* temporal
* interaction-aware
* graph-structured

Design every preprocessing and annotation step with those assumptions in mind.
