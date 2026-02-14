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


class SinePositionalEncoding2D(nn.Module):
    """2D sine/cosine positional encoding for feature grids."""

    def __init__(self, channels: int, height: int, width: int):
        super().__init__()
        if channels % 4 != 0:
            raise ValueError("channels must be divisible by 4")

        y_vals = torch.linspace(0, 1, height).view(height, 1).repeat(1, width)
        x_vals = torch.linspace(0, 1, width).view(1, width).repeat(height, 1)

        dim_t = torch.arange(0, channels // 2, 2).float()
        inv_freq = 10000 ** (dim_t / (channels // 2))

        pos_x = x_vals.unsqueeze(-1) / inv_freq
        pos_y = y_vals.unsqueeze(-1) / inv_freq

        pe_x = torch.stack((pos_x.sin(), pos_x.cos()), dim=3).flatten(2)
        pe_y = torch.stack((pos_y.sin(), pos_y.cos()), dim=3).flatten(2)

        pe = torch.cat([pe_x, pe_y], dim=2).unsqueeze(0)
        self.register_buffer("positional_encoding", pe)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        pe = self.positional_encoding.permute(0, 3, 1, 2)
        return x + pe.to(x.device)
