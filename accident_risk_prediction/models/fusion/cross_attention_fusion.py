"""Cross-attention multimodal fusion.

The temporal summary of the sequence is the query. It attends separately
over each modality's per-frame tokens ([B, T, d]): scene (ViT), interaction
graph, and ego-motion. Every stream has T tokens, so each attention is a
real choice of *which frames* of that modality matter.
"""
from __future__ import annotations

from typing import Dict, List

import torch
import torch.nn as nn


class _CrossAttentionBlock(nn.Module):
    def __init__(self, d_model: int, num_heads: int, dropout: float) -> None:
        super().__init__()
        self.cross = nn.MultiheadAttention(d_model, num_heads, dropout=dropout, batch_first=True)
        self.norm1 = nn.LayerNorm(d_model)
        self.ffn = nn.Sequential(
            nn.Linear(d_model, d_model * 2), nn.GELU(), nn.Dropout(dropout),
            nn.Linear(d_model * 2, d_model),
        )
        self.norm2 = nn.LayerNorm(d_model)
        self.drop = nn.Dropout(dropout)

    def forward(self, query: torch.Tensor, key: torch.Tensor) -> torch.Tensor:
        attn_out, _ = self.cross(query, key, key, need_weights=False)
        x = self.norm1(query + self.drop(attn_out))
        return self.norm2(x + self.drop(self.ffn(x)))


class CrossAttentionFusion(nn.Module):
    def __init__(self, d_model: int, num_heads: int, num_layers: int, dropout: float,
                 streams: List[str]) -> None:
        super().__init__()
        self.streams = list(streams)
        self.key_norm = nn.ModuleDict({s: nn.LayerNorm(d_model) for s in self.streams})
        self.blocks = nn.ModuleDict({
            s: nn.ModuleList([_CrossAttentionBlock(d_model, num_heads, dropout) for _ in range(num_layers)])
            for s in self.streams
        })
        self.fuse = nn.Sequential(
            nn.Linear(d_model * (1 + len(self.streams)), d_model * 2), nn.GELU(), nn.Dropout(dropout),
            nn.Linear(d_model * 2, d_model),
        )
        self.norm_out = nn.LayerNorm(d_model)

    def forward(self, query: torch.Tensor, tokens: Dict[str, torch.Tensor]) -> torch.Tensor:
        """query: [B, d]; tokens[name]: [B, T, d] -> fused [B, d]."""
        q = query.unsqueeze(1)
        outs = [query]
        for s in self.streams:
            x = q
            key = self.key_norm[s](tokens[s])
            for blk in self.blocks[s]:
                x = blk(x, key)
            outs.append(x.squeeze(1))
        return self.norm_out(self.fuse(torch.cat(outs, dim=-1)))
