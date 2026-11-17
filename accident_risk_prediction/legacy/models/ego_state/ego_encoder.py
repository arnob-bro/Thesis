"""Ego-State Encoder (MLP) over structured dashboard telemetry.

Contract (MD):
    forward(ego: [B, T, D_in]) -> [B, 64]

The MD recommends an MLP or *lightweight temporal encoder*. We use an MLP
applied per-timestep followed by mean-pooling over T - matching the
"speed/RPM dynamics" semantics described in the spec.
"""
from __future__ import annotations

import torch
import torch.nn as nn


class EgoEncoder(nn.Module):
    def __init__(
        self,
        input_dim: int = 4,
        hidden_dim: int = 64,
        output_dim: int = 64,
        dropout: float = 0.1,
    ) -> None:
        super().__init__()
        self.mlp = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.ReLU(inplace=True),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, output_dim),
        )
        self.norm = nn.LayerNorm(output_dim)
        self.output_dim = output_dim

    def forward(self, ego: torch.Tensor) -> torch.Tensor:
        """
        ego: [B, T, D_in]  or  [B, D_in]
        returns: [B, 64]
        """
        if ego.dim() == 2:
            x = self.mlp(ego)
            return self.norm(x)
        B, T, _ = ego.shape
        x = self.mlp(ego.view(B * T, -1)).view(B, T, -1)
        x = x.mean(dim=1)
        return self.norm(x)
