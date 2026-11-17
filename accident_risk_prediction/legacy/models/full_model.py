"""End-to-end multimodal accident risk model.

Architecture follows arnob/*.md:

    [B,T,3,H,W]   ──► ViT (frozen) ──────────►  scene  [B, 768]
                                                              │
    list[T] graphs ──► GAT ─► per-frame pool ─► [B,T,256] ─► Temporal TF ─► [B,512]
                                       │
                       last-frame node set + mask ─────────►  GNN set [B,N,256]
                                                              │
    [B,T,D_ego]   ──► EgoEncoder ────────────────────────►  ego  [B, 64]
                                                              │
                                          Cross-Attention Fusion
                                                              │
                                                          [B, 512]
                                                              │
                                                       ┌──────┴──────┐
                                                       │             │
                                                Classification    TTC head
                                                  [B, 4]            [B, 1]
"""
from __future__ import annotations

from typing import Dict, List, Optional

import torch
import torch.nn as nn
from torch_geometric.data import Batch

from models.ego_state.ego_encoder import EgoEncoder
from models.fusion.cross_attention_fusion import CrossAttentionFusion
from models.gnn.gat_model import InteractionGAT
from models.temporal.temporal_transformer import TemporalTransformer
from models.vit.vit_model import FrozenViT
from utils.config import Config


class AccidentRiskModel(nn.Module):
    def __init__(self, cfg: Config) -> None:
        super().__init__()
        self.cfg = cfg
        self.ablation = cfg.ablation

        # --- Frozen encoders ---
        self.vit: Optional[FrozenViT] = (
            FrozenViT(model_name=cfg.vit.model_name, output_dim=cfg.vit.output_dim)
            if self.ablation.use_vit else None
        )

        # --- Trainable encoders ---
        self.gnn: Optional[InteractionGAT] = (
            InteractionGAT(
                node_in_dim=cfg.features.node_dim,
                edge_in_dim=cfg.features.edge_dim,
                hidden_dim=cfg.gnn.hidden_dim,
                output_dim=cfg.gnn.output_dim,
                num_heads=cfg.gnn.num_heads,
                num_layers=cfg.gnn.num_layers,
                dropout=cfg.gnn.dropout,
            )
            if self.ablation.use_gnn else None
        )

        self.temporal: Optional[TemporalTransformer] = (
            TemporalTransformer(
                input_dim=cfg.temporal.input_dim,
                d_model=cfg.temporal.d_model,
                num_heads=cfg.temporal.num_heads,
                num_layers=cfg.temporal.num_layers,
                dim_feedforward=cfg.temporal.dim_feedforward,
                dropout=cfg.temporal.dropout,
                max_seq_len=cfg.temporal.max_seq_len,
            )
            if self.ablation.use_temporal else None
        )

        self.ego: Optional[EgoEncoder] = (
            EgoEncoder(
                input_dim=cfg.ego_encoder.input_dim,
                hidden_dim=cfg.ego_encoder.hidden_dim,
                output_dim=cfg.ego_encoder.output_dim,
                dropout=cfg.ego_encoder.dropout,
            )
            if self.ablation.use_ego else None
        )

        # --- Fusion ---
        self.fusion = CrossAttentionFusion(
            d_model=cfg.fusion.d_model,
            num_heads=cfg.fusion.num_heads,
            num_layers=cfg.fusion.num_layers,
            dropout=cfg.fusion.dropout,
            vit_dim=cfg.fusion.vit_dim,
            gnn_dim=cfg.fusion.gnn_dim,
            temporal_dim=cfg.fusion.temporal_dim,
            ego_dim=cfg.fusion.ego_dim,
            output_dim=cfg.fusion.output_dim,
        )

        # --- Heads ---
        self.cls_head = nn.Linear(cfg.fusion.output_dim, cfg.heads.num_classes)
        self.ttc_head = nn.Sequential(
            nn.Linear(cfg.fusion.output_dim, cfg.fusion.output_dim // 2),
            nn.ReLU(inplace=True),
            nn.Linear(cfg.fusion.output_dim // 2, cfg.heads.ttc_output_dim),
        )

        self.max_nodes = int(cfg.graph.max_nodes)

    # ------------------------------------------------------------------
    # Modality forward passes
    # ------------------------------------------------------------------
    def _encode_scene(self, frames: torch.Tensor) -> Optional[torch.Tensor]:
        """frames: [B, T, 3, H, W]  ->  [B, 768] (mean over T)."""
        if self.vit is None:
            return None
        B, T, C, H, W = frames.shape
        flat = frames.view(B * T, C, H, W)
        with torch.no_grad():
            emb = self.vit(flat)                                    # [B*T, 768]
        emb = emb.view(B, T, -1).mean(dim=1)                        # [B, 768]
        return emb

    def _encode_graphs(
        self, graphs_per_t: List[Batch], device: torch.device,
    ):
        """Run GAT on every (b, t) graph in the batch.

        Returns
            per_frame:   [B, T, 256]  (per-frame mean-pooled GNN embedding)
            last_dense:  [B, N, 256]  (per-node embeddings for the *last* frame)
            last_mask:   [B, N] bool
        """
        if self.gnn is None:
            return None, None, None

        T = len(graphs_per_t)
        B = graphs_per_t[0].num_graphs
        per_frame = torch.zeros(B, T, self.gnn.output_dim, device=device)

        last_dense: Optional[torch.Tensor] = None
        last_mask: Optional[torch.Tensor] = None

        for t, batch in enumerate(graphs_per_t):
            batch = batch.to(device)
            node_emb, pooled = self.gnn(batch)                      # [N,256],[B,256]
            per_frame[:, t, :] = pooled
            if t == T - 1:
                last_dense, last_mask = InteractionGAT.to_dense_node_set(
                    node_emb, batch.batch, self.max_nodes,
                )
        return per_frame, last_dense, last_mask

    def _encode_temporal(
        self, per_frame_gnn: Optional[torch.Tensor],
    ) -> Optional[torch.Tensor]:
        if self.temporal is None or per_frame_gnn is None:
            return None
        return self.temporal(per_frame_gnn)                         # [B, 512]

    def _encode_ego(self, ego_state: torch.Tensor) -> Optional[torch.Tensor]:
        if self.ego is None:
            return None
        return self.ego(ego_state)                                  # [B, 64]

    # ------------------------------------------------------------------
    # Forward
    # ------------------------------------------------------------------
    def forward(self, sample: Dict) -> Dict[str, torch.Tensor]:
        frames: torch.Tensor = sample["frames"]
        graphs_per_t: List[Batch] = sample["graphs_per_t"]
        ego_state: torch.Tensor = sample["ego_state"]
        device = frames.device

        vit_emb = self._encode_scene(frames)
        per_frame_gnn, gnn_node_set, gnn_mask = self._encode_graphs(
            graphs_per_t, device
        )

        # Temporal needs per-frame embeddings; fall back to mean of ViT if GNN
        # is ablated. If both are off, the temporal branch is also off.
        if per_frame_gnn is None and self.temporal is not None and vit_emb is not None:
            # Repeat the (already temporally pooled) ViT embedding across T to
            # give the temporal encoder a coherent input shape.
            T = frames.size(1)
            per_frame_gnn = vit_emb.unsqueeze(1).expand(-1, T, -1)
        temporal_emb = self._encode_temporal(per_frame_gnn)

        ego_emb = self._encode_ego(ego_state)

        # The fusion module requires the temporal query. If temporal is ablated
        # use ego or scene as the fallback query (still cross-attention based).
        if temporal_emb is None:
            if ego_emb is not None:
                temporal_emb = ego_emb.new_zeros(
                    ego_emb.size(0), self.fusion.d_model
                )
                temporal_emb[:, : ego_emb.size(-1)] = ego_emb
            elif vit_emb is not None:
                temporal_emb = vit_emb.new_zeros(
                    vit_emb.size(0), self.fusion.d_model
                )
                temporal_emb[:, : vit_emb.size(-1)] = vit_emb
            else:
                raise RuntimeError("All modalities ablated - nothing to fuse.")

        fused = self.fusion(
            vit_embed=vit_emb,
            gnn_node_set=gnn_node_set,
            gnn_mask=gnn_mask,
            temporal_embed=temporal_emb,
            ego_embed=ego_emb,
        )

        logits = self.cls_head(fused)                               # [B, 4]
        ttc = self.ttc_head(fused).squeeze(-1)                      # [B]

        return {"logits": logits, "ttc": ttc, "fused": fused}
