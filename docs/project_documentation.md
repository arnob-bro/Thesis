# 🚀 Multimodal Accident Risk Prediction System (Research-Grade)

## 📌 Objective

Build a **multimodal spatiotemporal accident risk prediction system** using:

* YOLOv8 (object detection)
* ByteTrack (object tracking)
* Graph Neural Networks (interaction modeling)
* Vision Transformer (scene understanding)
* Temporal Transformer (sequence modeling)
* Cross-attention fusion

---

# 🏗️ Project Structure

```
project_root/
│
├── data/
│   ├── raw_videos/
│   ├── extracted_frames/
│   ├── annotations/
│   └── processed/
│
├── models/
│   ├── yolo/
│   ├── tracking/
│   ├── gnn/
│   ├── vit/
│   ├── temporal/
│   └── fusion/
│
├── features/
│   ├── object_features.py
│   ├── interaction_features.py
│   └── scene_features.py
│
├── pipelines/
│   ├── frame_extractor.py
│   ├── detection_pipeline.py
│   ├── tracking_pipeline.py
│   ├── graph_pipeline.py
│   └── training_pipeline.py
│
├── utils/
│   ├── config.py
│   ├── metrics.py
│   ├── visualization.py
│   └── logger.py
│
├── configs/
│   └── config.yaml
│
├── train.py
├── evaluate.py
└── README.md
```

---

# ⚙️ Core Modules Implementation Guide

## 1️⃣ Frame Extraction

**File:** `pipelines/frame_extractor.py`

### Tasks:

* Load video
* Extract frames at 5–10 FPS
* Save to `/data/extracted_frames`

---

## 2️⃣ Object Detection (YOLOv8)

**File:** `pipelines/detection_pipeline.py`

### Tasks:

* Load pretrained YOLOv8
* Run inference on frames
* Output:

  ```
  frame_id, object_id, class, bbox(x,y,w,h), confidence
  ```

---

## 3️⃣ Object Tracking (ByteTrack)

**File:** `pipelines/tracking_pipeline.py`

### Tasks:

* Assign consistent IDs to detected objects
* Track across frames
* Output:

  ```
  track_id, frame_id, bbox, class
  ```

---

## 4️⃣ Object Feature Extraction

**File:** `features/object_features.py`

### Extract:

* Center (x, y)
* Width, height
* Velocity (Δposition / Δtime)
* Acceleration
* Object type

---

## 5️⃣ Interaction Feature Engineering

**File:** `features/interaction_features.py`

### For each object pair:

* Distance
* Relative velocity
* Angle
* Approx Time-To-Collision (TTC)

---

## 6️⃣ Graph Construction

**File:** `pipelines/graph_pipeline.py`

### Tasks:

* Nodes = objects
* Edges = interactions
* Build adjacency matrix
* Prepare graph data for GNN

---

## 7️⃣ Graph Neural Network (GNN)

**Folder:** `models/gnn/`

### Requirements:

* Use Graph Attention Network (GAT)
* Input:

  * Node features
  * Edge features
* Output:

  * Interaction embedding

---

## 8️⃣ Scene Understanding (ViT)

**File:** `models/vit/vit_model.py`

### Tasks:

* Load pretrained Vision Transformer
* Extract scene embeddings
* Reduce dimension (MLP/PCA)

---

## 9️⃣ Temporal Modeling

**Folder:** `models/temporal/`

### Tasks:

* Input: sequence of embeddings (T = 8–16 frames)
* Use Transformer Encoder
* Output: temporal feature representation

---

# 🔟 Ego Vehicle State Modality (Structured Dashboard Data)

## Folder:

```text
models/ego_state/
```

---

# Objective

Integrate manually-provided dashboard values as an additional modality.

This modality represents:

* ego vehicle dynamics
* internal vehicle state
* driver behavior indicators

---

# IMPORTANT

Dashboard data is NOT treated as:

* image modality
* OCR task
* visual feature extraction

Instead, it is treated as:

* structured numerical modality

---

# Supported Dashboard Features

Recommended inputs:

```text
speed
rpm
acceleration (optional)
brake_state (optional)
gear (optional)
```

---

# Dataset Input Format

## Example

```json
{
  "timestamp": 12.4,
  "speed": 48,
  "rpm": 4200
}
```

---

# Storage Location

```text
data/
└── ego_state/
    ├── train/
    ├── val/
    └── test/
```

---

# Sequence Synchronization Requirement

Dashboard data MUST:

* align temporally with video frames
* use timestamps synchronized with extracted frames

---

# Example Sequence Alignment

```text
Frame 001 → speed = 42
Frame 002 → speed = 44
Frame 003 → speed = 47
```

---

# Ego State Encoder

## File

```text
models/ego_state/ego_encoder.py
```

---

# Tasks

Encode structured dashboard signals into:

* ego-state embeddings

---

# Recommended Architecture

Use:

* MLP encoder
  OR
* lightweight temporal encoder

---

# Example

```text
[speed, rpm]
        ↓
Linear
        ↓
ReLU
        ↓
Linear
        ↓
ego_embedding
```

---

# Purpose of Ego-State Modality

This modality captures:

* vehicle aggressiveness
* acceleration behavior
* speed-related risk
* driver motion dynamics

---

# Why This Matters

Accident risk depends not only on:

* external traffic interactions

but also:

* ego vehicle behavior

Example:

```text
High speed + close interaction
= much higher risk
```

---

# Multimodal Fusion Update

Fusion module MUST additionally fuse:

* ViT scene embeddings
* GNN interaction embeddings
* temporal embeddings
* ego-state embeddings

---

# Updated Fusion Goal

The fusion system should reason jointly over:

* scene context
* traffic interactions
* temporal evolution
* ego vehicle dynamics

---

# Important Multimodal Assumption

Even though dashboard values are numerical,
they represent a DIFFERENT semantic modality.

This qualifies as multimodal learning because:

* different modalities encode different aspects of risk.

---

# Example

| Modality             | Information        |
| -------------------- | ------------------ |
| ViT                  | scene context      |
| GNN                  | interactions       |
| Temporal Transformer | motion evolution   |
| Ego State            | speed/RPM dynamics |

---

# Final Objective

The model should learn relationships such as:

```text
High ego speed
+
dangerous interaction
+
dense traffic
=
high accident risk
```


## 1️⃣1️⃣ Fusion Module (Cross-Attention)

**Folder:** `models/fusion/`

### Tasks:

* Fuse:

  * GNN output
  * ViT embedding
  * Temporal features
* Use attention mechanism (not simple concat)

---

## 1️⃣2️⃣ Output Head

* Classification:

  * Risk level (0–3)
* Regression:

  * Time-to-Collision (TTC)

---

# 🧠 Training Pipeline

## File: `pipelines/training_pipeline.py`

### Steps:

1. Load processed dataset
2. Forward pass:

   * YOLO → Tracking → Features → GNN
   * Frames → ViT
   * Sequence → Temporal Transformer
3. Fusion
4. Prediction

---

# 📊 Loss Functions

### Classification:

* CrossEntropyLoss

### Regression:

* SmoothL1Loss

### Final Loss:

```
Loss = Classification + λ * TTC Loss
```

---

# 📈 Evaluation Metrics

Implement in `utils/metrics.py`:

* Accuracy
* F1 Score
* ROC-AUC
* Time-to-Accident (TTA)
* Early Risk Detection Score

---

# ⚙️ Config System

## File: `configs/config.yaml`

Include:

```
frame_rate: 5
sequence_length: 12
num_classes: 4
learning_rate: 0.0001
batch_size: 8
```

---

# 🚀 Training Entry Point

## File: `train.py`

### Tasks:

* Load config
* Initialize models
* Train loop
* Save checkpoints

---

# 📊 Evaluation Script

## File: `evaluate.py`

### Tasks:

* Load trained model
* Run on validation/test set
* Output metrics + confusion matrix

---

# 🧪 Expected Outputs

* Risk prediction per frame
* Risk prediction over time
* TTC estimation
* Visualization (optional)

---

# 🔥 Notes for Claude Code

* Use **PyTorch**
* Use **Ultralytics YOLOv8 API**
* Use **PyTorch Geometric for GNN**
* Use **HuggingFace Transformers for ViT & Temporal Transformer**
* Ensure modular design (each component independent)
* Add logging and checkpoint saving

---

# 💯 Final Goal

Deliver a system that:

* Learns object interactions
* Understands scene context
* Models temporal behavior
* Predicts accident risk early

---
