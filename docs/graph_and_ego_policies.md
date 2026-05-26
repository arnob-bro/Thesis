# 🚗 EGO VEHICLE GRAPH POLICY (CRITICAL)

The ego vehicle (carrier vehicle containing the camera)
MUST NOT be included as a graph node.

The traffic interaction graph MUST contain ONLY:

- external vehicles
- pedestrians
- motorcycles
- bicycles
- buses
- trucks
- rickshaws

The ego vehicle is modeled separately through:

ego-state modality

using:
- speed
- rpm_estimate
- acceleration behavior
- brake state

The ego vehicle representation MUST enter the system ONLY through:

ego_embedding

during multimodal fusion.

This avoids:
- duplicated ego representations
- graph ambiguity
- unstable ego localization
- camera-motion compensation issues

---

# 🚘 EGO-OBJECT INTERACTION POLICY

Even though the ego vehicle is NOT included
as a graph node, the system MUST compute:

ego-to-object interaction features

for all detected traffic participants.

These include:

- ego TTC
- ego-relative distance
- closing speed
- collision proximity

These features represent:
- direct collision risk to the carrier vehicle.

IMPORTANT:

The traffic graph models:

external object ↔ external object

interactions only.

Ego-relative interactions are computed separately
and fused through:

- ego-state encoder
- fusion module

to avoid:
- duplicated ego representation
- unstable ego graph construction
- camera-motion ambiguity

---

# 🔗 GRAPH EDGE CONSTRUCTION POLICY

Edges MUST be created using:

- spatial proximity
- nearest-neighbor relationships

Recommended:
- K-nearest neighbors
- K = 3–5

DO NOT:
- fully connect all objects

Edge creation should prioritize:
- nearby interactions
- possible collision relationships
- traffic interaction relevance

---

# ⏱️ GRAPH TEMPORAL POLICY

Graphs MUST be constructed independently for each frame.

Meaning:

1 frame = 1 graph

The Temporal Transformer is responsible for learning:

- graph evolution
- interaction evolution
- temporal risk progression

across consecutive frame-level graphs.

---

# 📊 GNN TEMPORAL AGGREGATION CONTRACT

For each frame:

Frame_t
→ Graph_t
→ GNN
→ Interaction Embedding_t

The sequence of interaction embeddings:

[GNN_t1, GNN_t2, ..., GNN_tn]

MUST be passed into the Temporal Transformer.

The Temporal Transformer learns:

- temporal interaction evolution
- motion progression
- evolving collision risk

---

# 📦 FRAME-LEVEL GRAPH STORAGE CONTRACT

Each sequence graph file MUST contain:

one graph per frame

Recommended structure:

[
  graph_t1,
  graph_t2,
  graph_t3
]

where each graph is a PyTorch Geometric Data object.

---

# 📈 EGO TTC COMPUTATION POLICY

The system MUST compute:

ego-to-object TTC

using:
- object motion
- bounding box expansion
- relative movement
- ego speed estimates

Because the system uses:
- monocular video
- approximate telemetry

all TTC estimates are considered:

approximate visual-motion TTC

NOT:
- physically exact measurements
- LiDAR-grade distance estimates

---

# 🎯 MULTIMODAL REASONING CONTRACT

The final prediction MUST jointly reason over:

- scene understanding (ViT)
- traffic interaction reasoning (GNN)
- temporal evolution (Temporal Transformer)
- ego behavior dynamics (Ego Encoder)
- ego-object collision risk

The system should learn relationships such as:

high ego speed
+
dangerous interaction
+
small TTC
+
dense traffic
=
high accident risk

---

# ⚠️ IMPLEMENTATION RESTRICTIONS (IMPORTANT)

DO NOT implement:

- dynamic graph neural networks
- fully connected graphs
- ego vehicle graph nodes
- online end-to-end YOLO training
- graph construction across multiple frames simultaneously

Graphs MUST remain:

frame-level

Temporal reasoning MUST remain:

sequence-level

through the Temporal Transformer.
