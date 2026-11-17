"""Graph Attention Network (GAT) for traffic interaction reasoning.

Output contract (MD):
    forward(Batch) -> node_embeddings of dim 256, with the per-graph mean
    pooled vector also returned (used as per-frame interaction embedding for
    the Temporal Transformer).

Edge features are 4-D per MD (distance, rel_velocity, ttc, rel_angle) and are
passed to `GATv2Conv` via its `edge_dim` argument.
"""
from __future__ import annotations

from typing import Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.data import Batch
from torch_geometric.nn import GATv2Conv, global_mean_pool


class InteractionGAT(nn.Module):
    def __init__(
        self,
        node_in_dim: int = 8,
        edge_in_dim: int = 4,
        hidden_dim: int = 128,
        output_dim: int = 256,
        num_heads: int = 4,
        num_layers: int = 2,
        dropout: float = 0.1,
    ) -> None:
        super().__init__()
        if num_layers < 1:
            raise ValueError("GAT must have at least one layer")
        self.dropout = dropout
        self.output_dim = output_dim

        self.input_proj = nn.Linear(node_in_dim, hidden_dim)

        self.layers = nn.ModuleList()
        in_dim = hidden_dim
        for i in range(num_layers):
            is_last = i == num_layers - 1
            out_per_head = output_dim // num_heads if is_last else hidden_dim
            self.layers.append(
                GATv2Conv(
                    in_channels=in_dim,
                    out_channels=out_per_head,
                    heads=num_heads,
                    concat=True,
                    edge_dim=edge_in_dim,
                    dropout=dropout,
                )
            )
            in_dim = out_per_head * num_heads

        if in_dim != output_dim:
            self.output_proj: nn.Module = nn.Linear(in_dim, output_dim)
        else:
            self.output_proj = nn.Identity()

    def forward(self, batch: Batch) -> Tuple[torch.Tensor, torch.Tensor]:
        """Returns (node_embeddings [N_total, 256], per_graph_pooled [B, 256])."""
        x = self.input_proj(batch.x)
        edge_attr = batch.edge_attr
        for i, layer in enumerate(self.layers):
            x = layer(x, batch.edge_index, edge_attr=edge_attr)
            if i < len(self.layers) - 1:
                x = F.elu(x)
                x = F.dropout(x, p=self.dropout, training=self.training)
        x = self.output_proj(x)
        pooled = global_mean_pool(x, batch.batch)
        return x, pooled

    @staticmethod
    def to_dense_node_set(
        node_embeddings: torch.Tensor,
        batch_index: torch.Tensor,
        max_nodes: int,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """Pack variable-size graphs into a dense `[B, max_nodes, D]` tensor.

        Returns (dense, mask) where `mask` is `[B, max_nodes]` with 1 for real
        nodes and 0 for padding. Used by the cross-attention fusion module.
        """
        device = node_embeddings.device
        B = int(batch_index.max().item()) + 1 if batch_index.numel() > 0 else 0
        D = node_embeddings.size(-1)
        dense = node_embeddings.new_zeros(B, max_nodes, D)
        mask = torch.zeros(B, max_nodes, device=device, dtype=torch.bool)

        for b in range(B):
            idx = (batch_index == b).nonzero(as_tuple=False).view(-1)
            n = min(int(idx.numel()), max_nodes)
            if n > 0:
                dense[b, :n] = node_embeddings[idx[:n]]
                mask[b, :n] = True
        return dense, mask
