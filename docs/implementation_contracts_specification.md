# 🚨 IMPLEMENTATION CONTRACTS & ENGINEERING SPECIFICATION
# Multimodal Spatiotemporal Accident Risk Prediction System

---

# 📌 PURPOSE OF THIS FILE

This file defines:

- exact implementation contracts
- tensor specifications
- data flow assumptions
- synchronization rules
- training stages
- graph formats
- dataset schemas
- batching requirements
- multimodal fusion contracts

This document is intended specifically for:

```text
Claude Code / AI coding agents
```

to reduce:
- ambiguity
- hallucinated implementations
- incompatible tensor dimensions
- broken training pipelines

---

# 🚨 CRITICAL IMPLEMENTATION PRINCIPLES

This project is:

- multimodal
- spatiotemporal
- graph-structured
- sequence-based
- interaction-aware

The implementation MUST preserve:

```text
space
+
time
+
interaction
+
ego behavior
```

throughout the pipeline.

---

# 🧠 GLOBAL SYSTEM FLOW

```text
Video Frames
    ↓
YOLOv8 Detection
    ↓
ByteTrack Tracking
    ↓
Object Features
    ↓
Interaction Features
    ↓
Graph Construction
    ↓
GNN / GAT
    ↓
Temporal Transformer
    ↓
Cross-Attention Fusion
    ↓
Risk Prediction + TTC
```

Additional parallel modalities:

```text
Frames → ViT → Scene Embeddings
Dashboard Data → Ego Encoder → Ego Embeddings
```

---

# ⚠️ IMPORTANT DESIGN ASSUMPTIONS

## 1. YOLOv8 IS FROZEN

YOLOv8 is used ONLY as:

- pretrained detector
- feature extraction source

DO NOT retrain YOLOv8.

---

## 2. ViT IS FROZEN

ViT is used ONLY as:

- pretrained scene embedding extractor

DO NOT retrain ViT.

---

## 3. TRAINABLE MODULES

Trainable modules include:

- GNN / GAT
- Temporal Transformer
- Ego-State Encoder
- Cross-Attention Fusion
- Output Heads

---

# 📦 DATASET ROOT STRUCTURE

```text
dataset/
│
├── raw_videos/
│
├── extracted_frames/
│
├── sequences/
│
├── detections/
│
├── tracking/
│
├── graphs/
│
├── ego_state/
│
├── embeddings/
│
├── cache/
│
└── metadata/
```

---

# 📂 SEQUENCE STRUCTURE (CRITICAL)

The PRIMARY training unit is:

```text
SEQUENCE
NOT FRAME
```

Each sequence MUST contain:

```text
sequence_0001/
│
├── frames/
│   ├── frame_0001.jpg
│   ├── frame_0002.jpg
│   └── ...
│
├── detections.json
├── tracks.json
├── ego_state.json
├── graph.pt
├── scene_embeddings.npy
├── label.json
└── metadata.json
```

---

# ⏱️ TEMPORAL ASSUMPTIONS

## REQUIRED

Frame order MUST remain:

```text
strictly chronological
```

---

# Sequence Length

Recommended:

```text
8–16 frames
```

Default:

```yaml
sequence_length: 12
```

---

# Frame Rate

Recommended extraction FPS:

```yaml
frame_rate: 5
```

---

# 📊 RISK LABEL CONTRACT

Risk labels are:

```text
SEQUENCE-LEVEL
NOT FRAME-LEVEL
```

---

# Risk Levels

```text
0 = Safe
1 = Slight Risk
2 = Moderate Risk
3 = High Risk
```

---

# Label Philosophy

Labels are assigned using:

- TTC trends
- unsafe proximity
- abrupt motion
- human semantic review

---

# Example label.json

```json
{
  "sequence_id": "0001",
  "risk_level": 2,
  "ttc_estimate": 1.8,
  "scenario": "unsafe overtaking",
  "notes": "motorcycle passed very close"
}
```

---

# 🚘 EGO-STATE DATA CONTRACT

Ego-state values are:

```text
approximate sequence-level telemetry estimates
```

NOT:
- precise automotive sensor streams

---

# Ego-State Features

Required:

```text
speed
rpm_estimate
```

Optional:

```text
brake_state
gear
acceleration
```

---

# Example ego_state.json

```json
{
  "sequence_id": "0001",
  "timestamps": [0.0,0.2,0.4,0.6],
  "speed": [40,42,44,47],
  "rpm_estimate": [2500,2700,3000,3300],
  "brake": [0,0,0,1]
}
```

---

# 🔍 DETECTION CONTRACT

YOLO detections format:

```json
[
  {
    "frame_id": 1,
    "class_id": 0,
    "class_name": "car",
    "confidence": 0.92,
    "bbox": [x, y, w, h]
  }
]
```

---

# 🚗 TRACKING CONTRACT

Tracking MUST preserve:

- temporal consistency
- stable object IDs

Reject:
- tracks shorter than 3 frames
- unstable tracks

---

# 🧠 NODE FEATURE CONTRACT

Node features MUST contain:

```text
[
 x_position,
 y_position,
 width,
 height,
 velocity,
 acceleration,
 direction_angle,
 object_type
]
```

---

# 🔗 EDGE FEATURE CONTRACT

Edge features MUST contain:

```text
[
 distance,
 relative_velocity,
 TTC,
 relative_angle
]
```

---

# 📈 TTC PHILOSOPHY

TTC values are:

```text
approximate visual-motion estimates
```

NOT:
- physically exact measurements

---

# 🌍 GRAPH CONTRACT

Graphs MUST use:

```text
PyTorch Geometric format
```

---

# GNN OUTPUT SHAPE

```text
[B, N, 256]
```

---

# 🌍 VISION TRANSFORMER CONTRACT

ViT OUTPUT SHAPE:

```text
[B, 768]
```

---

# ⏱️ TEMPORAL TRANSFORMER CONTRACT

Temporal Output Shape:

```text
[B, 512]
```

---

# 🚘 EGO-STATE ENCODER CONTRACT

Ego Encoder Output Shape:

```text
[B, 64]
```

---

# 🔥 MULTIMODAL FUSION CONTRACT

Fusion MUST use:

```text
cross-attention
```

DO NOT use:
- naive concatenation only

---

# REQUIRED INPUT SHAPES

```text
ViT             → [B, 768]
GNN             → [B, N, 256]
Temporal        → [B, 512]
Ego             → [B, 64]
```

---

# Fusion Output Shape

```text
[B, 512]
```

---

# 🎯 OUTPUT HEAD CONTRACT

Classification:

```text
[B, 4]
```

Regression:

```text
[B, 1]
```

---

# 📉 LOSS FUNCTIONS

## Classification

```python
CrossEntropyLoss
```

## TTC Regression

```python
SmoothL1Loss
```

---

# ⚡ TRAINING STRATEGY

## Stage 1

Run YOLO detection and cache outputs.

## Stage 2

Run ByteTrack and cache tracks.

## Stage 3

Generate:
- node features
- edge features
- TTC
- graphs

## Stage 4

Extract ViT embeddings.

## Stage 5

Train:
- GNN
- Temporal Transformer
- Ego Encoder
- Fusion
- Output Heads

---

# 📈 EVALUATION METRICS

Required:
- Accuracy
- Precision
- Recall
- F1-score
- ROC-AUC
- Time-to-Accident (TTA)
- Early Risk Detection Score

---

# ⚠️ MEMORY & HARDWARE ASSUMPTIONS

Target hardware:

```text
single GPU with limited memory
```

Implementation MUST:
- cache embeddings
- avoid recomputation
- use efficient batching

---

# 💯 FINAL ARCHITECTURAL PRINCIPLE

The system MUST jointly reason over:

```text
scene context
+
interaction reasoning
+
temporal evolution
+
ego vehicle behavior
```

to produce:

```text
early accident risk prediction
```
