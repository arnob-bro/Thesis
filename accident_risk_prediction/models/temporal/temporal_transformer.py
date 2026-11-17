"""Temporal Transformer encoder over per-frame interaction embeddings.

Contract (MD):
    forward(seq: [B, T, input_dim]) -> [B, 512]

Sequence length T = 8..16 (default 12). Positional embeddings + learnable
CLS token; we return the CLS state as the sequence summary.
"""
from __future__ import annotations

import math

import torch
import torch.nn as nn


class PositionalEncoding(nn.Module):
    def __init__(self, d_model: int, max_len: int = 64) -> None:
        super().__init__()
        pe = torch.zeros(max_len, d_model)
        pos = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)
        div = torch.exp(
            torch.arange(0, d_model, 2).float() * (-math.log(10000.0) / d_model)
        )
        pe[:, 0::2] = torch.sin(pos * div)
        pe[:, 1::2] = torch.cos(pos * div)
        self.register_buffer("pe", pe.unsqueeze(0))     # [1, max_len, d_model]

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return x + self.pe[:, : x.size(1)]


class TemporalTransformer(nn.Module):
    def __init__(
        self,
        input_dim: int = 256,
        d_model: int = 512,
        num_heads: int = 8,
        num_layers: int = 4,
        dim_feedforward: int = 1024,
        dropout: float = 0.1,
        max_seq_len: int = 32,
    ) -> None:
        super().__init__()
        self.input_proj = (
            nn.Linear(input_dim, d_model) if input_dim != d_model else nn.Identity()
        )
        self.cls_token = nn.Parameter(torch.zeros(1, 1, d_model))
        nn.init.trunc_normal_(self.cls_token, std=0.02)

        self.pos = PositionalEncoding(d_model, max_len=max_seq_len + 1)

        encoder_layer = nn.TransformerEncoderLayer(
            d_model=d_model,
            nhead=num_heads,
            dim_feedforward=dim_feedforward,
            dropout=dropout,
            batch_first=True,
            activation="gelu",
        )
        self.encoder = nn.TransformerEncoder(encoder_layer, num_layers=num_layers)
        self.norm = nn.LayerNorm(d_model)
        self.output_dim = d_model

    def forward(self, seq: torch.Tensor) -> torch.Tensor:
        B = seq.size(0)
        x = self.input_proj(seq)                        # [B, T, d_model]
        cls = self.cls_token.expand(B, -1, -1)          # [B, 1, d_model]
        x = torch.cat([cls, x], dim=1)
        x = self.pos(x)
        x = self.encoder(x)
        x = self.norm(x)
        return x[:, 0]                                  # CLS -> [B, d_model]
