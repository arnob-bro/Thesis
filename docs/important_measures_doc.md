# ⚠️ Critical Engineering & Research Constraints (IMPORTANT)

This section defines critical implementation constraints, debugging checks, and research-quality requirements that MUST be followed during development.

---

# 🚨 1. Data Integrity Requirements

## DO NOT:

* Train on random single frames only
* Mix frame order
* Ignore temporal continuity

## MUST:

* Process sequential frames in chronological order
* Maintain temporal consistency across sequences
* Use fixed sequence lengths (e.g., 8–16 frames)

---

# 🧠 2. Labeling Constraints

## Labels must represent EARLY risk prediction:

* 0 = Safe
* 1 = Slight Risk
* 2 = Moderate Risk
* 3 = High Risk

## Important:

* Labels should correspond to sequences, NOT isolated frames
* Avoid inconsistent labeling across similar scenarios
* Store labels in structured format (JSON/CSV)

Example:

```id="qjlwmk"
{
  "sequence_id": 102,
  "risk_level": 2,
  "ttc": 1.8
}
```

---

# ⚙️ 3. Tracking Stability Constraints

Tracking consistency is CRITICAL.

## MUST:

* Use ByteTrack as primary tracker
* Preserve stable object IDs across frames
* Reject noisy tracks:

  * tracks shorter than 3 frames
  * low-confidence tracks

## Add Debugging Utilities:

* Visualize tracking IDs on frames
* Detect ID switching events
* Save tracking logs

---

# 📊 4. Feature Engineering Constraints

## ALL numerical features MUST be normalized.

Features include:

* distance
* velocity
* acceleration
* bbox dimensions
* TTC

---

## Numerical Stability Rules

### MUST:

* Avoid divide-by-zero
* Clamp extreme values
* Handle NaN/Inf safely

Example:

```id="9snx9v"
ttc = distance / (relative_velocity + 1e-6)
```

---

# 🔗 5. Graph Construction Constraints

Graph quality directly affects GNN performance.

## DO NOT:

* Fully connect all objects blindly

## MUST:

* Use nearest-neighbor graph OR distance threshold graph
* Keep graph sparse
* Recommended:

  * K = 3 to 5 nearest neighbors

---

# 🧬 6. Fusion Constraints

## DO NOT:

* Use naive feature concatenation as final fusion strategy

## MUST:

* Use attention-based fusion
  OR
* weighted multimodal fusion

Fusion should combine:

* Scene embeddings (ViT)
* Interaction embeddings (GNN)
* Temporal embeddings

---

# ⏱ 7. Temporal Modeling Constraints

Temporal reasoning is REQUIRED.

## MUST:

* Maintain ordered sequences
* Preserve object identity over time
* Support variable motion dynamics

---

## Recommended:

* Transformer Encoder
* Positional embeddings
* Sequence length:

  * 8–16 frames

---

# 🌍 8. Dataset Bias Prevention

The dataset MUST contain:

* normal driving
* risky interactions
* near misses
* dense traffic
* intersections
* overtaking
* braking events

## DO NOT:

* Train mostly on safe scenes

This will cause severe bias toward predicting “safe”.

---

# 📈 9. Evaluation Constraints

Accuracy alone is NOT sufficient.

## MUST compute:

* Accuracy
* Precision
* Recall
* F1-score
* ROC-AUC
* Time-to-Accident (TTA)
* Early Risk Detection Score

---

# 🧪 10. Ablation Study Support (REQUIRED)

The codebase MUST support disabling components individually.

## Required ablation modes:

* YOLO only
* YOLO + Temporal
* YOLO + GNN
* YOLO + ViT
* Full multimodal model

---

# ⚡ 11. Performance & Memory Optimization

Pipeline is computationally expensive.

## MUST:

* Cache detections
* Cache extracted features
* Avoid recomputing YOLO outputs during every epoch

---

## Recommended:

Create:

```id="k6vv89"
cache/
├── detections/
├── tracks/
├── features/
└── embeddings/
```

---

# 🧩 12. Debugging & Visualization Tools

The system MUST include visualization utilities.

## Required visualizations:

* Bounding boxes
* Tracking IDs
* Graph connections
* Attention maps (optional)
* Risk score over time

---

# 🔒 13. Reproducibility Requirements

## MUST:

* Set random seeds
* Save configs
* Save checkpoints
* Log hyperparameters

---

## Save:

* model weights
* optimizer states
* training metrics
* validation metrics

---

# 📂 14. Logging Requirements

Implement structured logging.

## Log:

* training loss
* validation loss
* GPU memory usage
* learning rate
* epoch timing

---

# 🚀 15. Research-Grade Code Quality Requirements

Code MUST:

* be modular
* support easy experimentation
* support future detector replacement
* support future dataset scaling

---

# 💯 16. Final System Objective

The final system should:

* reason over interactions
* understand scene context
* model temporal behavior
* predict accident risk EARLY
* generalize to chaotic real-world traffic

---

# 🔥 IMPORTANT FOR IMPLEMENTATION

This is NOT a simple classification project.

The system is fundamentally:

* multimodal
* spatiotemporal
* interaction-aware
* sequence-based

All modules should be implemented with this assumption.
