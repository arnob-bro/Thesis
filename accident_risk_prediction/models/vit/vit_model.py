"""Frozen Vision Transformer used for global scene embeddings.

Contract (MD):
    forward(images: [B, 3, H, W]) -> [B, 768]

ViT is frozen — its parameters never receive gradients.
"""
from __future__ import annotations

import torch
import torch.nn as nn
from transformers import ViTModel


class FrozenViT(nn.Module):
    def __init__(
        self,
        model_name: str = "google/vit-base-patch16-224-in21k",
        output_dim: int = 768,
    ) -> None:
        super().__init__()
        self.backbone = ViTModel.from_pretrained(model_name, add_pooling_layer=False)
        for p in self.backbone.parameters():
            p.requires_grad = False
        self.backbone.eval()

        hidden = self.backbone.config.hidden_size
        if hidden != output_dim:
            self.proj: nn.Module = nn.Linear(hidden, output_dim, bias=False)
        else:
            self.proj = nn.Identity()
        self.output_dim = output_dim

    def train(self, mode: bool = True) -> "FrozenViT":
        super().train(mode)
        self.backbone.eval()                            # always eval (frozen)
        return self

    @torch.no_grad()
    def encode(self, images: torch.Tensor) -> torch.Tensor:
        """images: [B, 3, H, W] -> [B, 768] using the [CLS] token."""
        out = self.backbone(pixel_values=images).last_hidden_state
        cls = out[:, 0, :]
        return self.proj(cls)

    def forward(self, images: torch.Tensor) -> torch.Tensor:
        return self.encode(images)
