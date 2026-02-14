"""Custom refinement layers for Point Set Refinement."""

import torch
import torch.nn as nn


class RefinementBlock(nn.Module):
    """Refinement block with cross-attn, self-attn, and MLP."""

    def __init__(self, dim: int, n_heads: int, dropout: float = 0.0):
        super().__init__()
        self.cross_attn = nn.MultiheadAttention(
            embed_dim=dim,
            num_heads=n_heads,
            dropout=dropout,
            batch_first=True,
        )
        self.norm1 = nn.LayerNorm(dim)

        self.self_attn = nn.MultiheadAttention(
            embed_dim=dim,
            num_heads=n_heads,
            dropout=dropout,
            batch_first=True,
        )
        self.norm2 = nn.LayerNorm(dim)

        self.mlp = nn.Sequential(
            nn.Linear(dim, dim * 2),
            nn.ReLU(),
            nn.Linear(dim * 2, dim),
        )
        self.norm3 = nn.LayerNorm(dim)

    def forward(self, point_features: torch.Tensor, image_features: torch.Tensor) -> torch.Tensor:
        """Refine point features using image context and point interactions."""
        attn_out, _ = self.cross_attn(
            query=point_features,
            key=image_features,
            value=image_features,
            need_weights=False,
        )
        point_features = self.norm1(point_features + attn_out)

        attn_out, _ = self.self_attn(
            query=point_features,
            key=point_features,
            value=point_features,
            need_weights=False,
        )
        point_features = self.norm2(point_features + attn_out)

        point_features = self.norm3(point_features + self.mlp(point_features))
        return point_features
