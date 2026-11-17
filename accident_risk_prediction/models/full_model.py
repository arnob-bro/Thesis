"""Rider-centric multimodal accident-risk model.

    ViT embeddings [B,T,768] (pre-computed, frozen) -> scene tokens   [B,T,d]
    interaction graphs (B*T graphs) -> GAT -> pool + rider node -> graph tokens [B,T,d]
    ego-motion [B,T,6] -> MLP                                       -> ego tokens   [B,T,d]

    per-frame mix of the tokens -> Temporal Transformer -> sequence summary [B,d]
    summary cross-attends over each modality's T tokens -> fused [B,d]
    fused -> risk head (ordinal or softmax) + time-to-collision head

Each modality and the temporal encoder can be switched off in `ablation`.
"""
from __future__ import annotations

from typing import Dict

import torch
import torch.nn as nn

from features.ego_motion import EGO_DIM
from features.graph_builder import EDGE_DIM, node_dim
from models.fusion.cross_attention_fusion import CrossAttentionFusion
from models.gnn.gat_model import InteractionGAT
from models.temporal.temporal_transformer import TemporalTransformer
from utils.config import Config


def probs_from_logits(logits: torch.Tensor, head_type: str) -> torch.Tensor:
    """Class probabilities [B, C] for either head type."""
    if head_type == "softmax":
        return torch.softmax(logits, dim=-1)
    # ordinal: logit k is P(risk > k); make it monotone, then take differences
    p_gt = torch.cummin(torch.sigmoid(logits), dim=-1).values
    ones = torch.ones_like(p_gt[:, :1])
    upper = torch.cat([ones, p_gt], dim=-1)
    lower = torch.cat([p_gt, torch.zeros_like(ones)], dim=-1)
    return upper - lower


class AccidentRiskModel(nn.Module):
    def __init__(self, cfg: Config) -> None:
        super().__init__()
        m, ab = cfg.model, cfg.ablation
        d, drop = int(m.d_model), float(m.dropout)
        self.head_type = str(cfg.heads.type)
        self.has_ego_node = bool(cfg.graph.include_ego_node)
        self.streams = [s for s, on in (("scene", ab.use_vit), ("graph", ab.use_gnn), ("ego", ab.use_ego)) if on]
        if not self.streams:
            raise ValueError("All modalities are switched off - nothing to learn from.")

        if "scene" in self.streams:
            self.scene_proj = nn.Sequential(nn.Linear(int(cfg.vit.output_dim), d), nn.GELU(), nn.Dropout(drop))
        if "graph" in self.streams:
            self.gnn = InteractionGAT(
                node_in_dim=node_dim(cfg), edge_in_dim=EDGE_DIM, hidden_dim=int(m.gnn.hidden_dim),
                output_dim=d, num_heads=int(m.gnn.num_heads), num_layers=int(m.gnn.num_layers), dropout=drop,
            )
            self.graph_proj = nn.Sequential(nn.Linear(d * (2 if self.has_ego_node else 1), d), nn.GELU())
        if "ego" in self.streams:
            self.ego_enc = nn.Sequential(nn.Linear(EGO_DIM, d), nn.GELU(), nn.Dropout(drop), nn.Linear(d, d))

        self.mix = nn.Linear(d * len(self.streams), d)
        self.temporal = (
            TemporalTransformer(input_dim=d, d_model=d, num_heads=int(m.temporal.num_heads),
                                num_layers=int(m.temporal.num_layers),
                                dim_feedforward=int(m.temporal.dim_feedforward), dropout=drop)
            if ab.use_temporal else None
        )
        self.fusion = CrossAttentionFusion(d, int(m.fusion.num_heads), int(m.fusion.num_layers), drop, self.streams)

        n_cls = int(cfg.heads.num_classes)
        self.cls_head = nn.Linear(d, n_cls - 1 if self.head_type == "ordinal" else n_cls)
        self.ttc_head = nn.Sequential(nn.Linear(d, d // 2), nn.GELU(), nn.Linear(d // 2, 1))

    def forward(self, batch: Dict) -> Dict[str, torch.Tensor]:
        B, T = batch["ego"].shape[:2]
        tokens: Dict[str, torch.Tensor] = {}
        if "scene" in self.streams:
            tokens["scene"] = self.scene_proj(batch["vit"])
        if "graph" in self.streams:
            g = batch["graphs"]
            node_emb, pooled = self.gnn(g)
            if self.has_ego_node:   # node 0 of every graph is the rider
                pooled = torch.cat([pooled, node_emb[g.ptr[:-1]]], dim=-1)
            tokens["graph"] = self.graph_proj(pooled).view(B, T, -1)
        if "ego" in self.streams:
            tokens["ego"] = self.ego_enc(batch["ego"])

        mixed = self.mix(torch.cat([tokens[s] for s in self.streams], dim=-1))    # [B, T, d]
        summary = self.temporal(mixed) if self.temporal is not None else mixed.mean(dim=1)
        fused = self.fusion(summary, tokens)

        logits = self.cls_head(fused)
        return {
            "logits": logits,
            "probs": probs_from_logits(logits, self.head_type),
            "ttc": self.ttc_head(fused).squeeze(-1),
        }
