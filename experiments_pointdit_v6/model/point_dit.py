"""Point-DiT V6 Scaled: Fourier-enhanced stippling model with 5M parameters.

V6 Scaling Upgrade:
- Increased from 1M params (dim=128, n_layers=4) to 5M params (dim=256, n_layers=6)
- Provides ~10x more computational capacity for learning complex N-body interactions
- Better equipped to learn crystalline point spacing through self-attention
- Training: ~1.5 hours/epoch (vs 0.75 hours for v5)

Key design:
- Shallow CNN encoder with GroupNorm (no BatchNorm) to avoid instability on binary masks
- Fourier positional embeddings (high-frequency grid) for precise spatial conditioning
- LayerNorm fusion to balance image and position features
- Predicts x_start (clean points) instead of noise for stable geometric learning
"""

import math
from typing import Tuple

import torch
import torch.nn as nn


class FourierEmbedder(nn.Module):
    """Fourier feature embedder for high-frequency positional encoding.
    
    Used to give each spatial location a unique high-frequency fingerprint,
    preventing mode collapse and enabling precise spatial control.
    """
    
    def __init__(self, num_freqs: int = 32, temperature: float = 100.0):
        super().__init__()
        self.num_freqs = num_freqs
        self.temperature = temperature
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x: (..., 1) coordinates in range [-1, 1]
        Returns:
            (..., num_freqs * 2) sinusoidal embeddings
        """
        # Generate frequencies: [temp^0, temp^(1/N), ..., temp^((N-1)/N)]
        exponent = torch.arange(self.num_freqs, dtype=x.dtype, device=x.device)
        freqs = self.temperature ** (exponent / self.num_freqs)
        
        # Create args: x * freq
        args = x * freqs.view(1, -1)  # (..., num_freqs)
        
        # Return [sin, cos]
        return torch.cat([torch.sin(args), torch.cos(args)], dim=-1)


class SinusoidalTimeEmbedding(nn.Module):
    """Sinusoidal time embedding (standard for diffusion models)."""

    def __init__(self, dim: int):
        super().__init__()
        self.dim = dim

    def forward(self, t: torch.Tensor) -> torch.Tensor:
        if t.dim() == 1:
            t = t.unsqueeze(-1)  # (B, 1)

        half_dim = self.dim // 2
        emb = math.log(10000) / (half_dim - 1)
        emb = torch.exp(torch.arange(half_dim, device=t.device) * -emb)
        emb = t * emb.unsqueeze(0)
        emb = torch.cat([torch.sin(emb), torch.cos(emb)], dim=-1)
        return emb


class PointDiT(nn.Module):
    """Point Diffusion Transformer with Fourier positional features (V6 Scaled).
    
    Upgraded from 1M to 5M parameters with dim=256, n_layers=6, n_heads=8.
    This provides sufficient capacity to learn complex N-body interactions.
    """

    def __init__(
        self,
        n_points: int = 2048,
        dim: int = 128,
        n_layers: int = 4,
        n_heads: int = 4,
        image_size: int = 512,
        dropout: float = 0.0,
    ):
        super().__init__()

        self.n_points = n_points
        self.dim = dim
        self.image_size = image_size

        # 1) Stable image encoder: shallow CNN with GroupNorm; outputs 64x64 grid for 512x512 input
        self.img_encoder = nn.Sequential(
            nn.Conv2d(1, 32, kernel_size=4, stride=2, padding=1),  # 512 -> 256
            nn.GroupNorm(8, 32),
            nn.SiLU(),
            nn.Conv2d(32, 64, kernel_size=4, stride=2, padding=1),  # 256 -> 128
            nn.GroupNorm(16, 64),
            nn.SiLU(),
            nn.Conv2d(64, dim, kernel_size=4, stride=2, padding=1),  # 128 -> 64
            nn.GroupNorm(32, dim),
            nn.SiLU(),
        )
        
        # 2) Fourier positional embeddings for spatial grid
        self.fourier_embed = FourierEmbedder(num_freqs=32, temperature=100.0)
        grid_dim = 32 * 2 * 2  # 32 freqs * 2 (sin/cos) * 2 (x/y) = 128
        
        # 3) Context fusion with LayerNorm (balances image and position features)
        self.fusion = nn.Sequential(
            nn.Linear(dim + grid_dim, dim),
            nn.LayerNorm(dim),
            nn.SiLU()
        )

        # 4) Point + time embeddings
        self.point_embed = nn.Sequential(
            nn.Linear(2, dim),
            nn.SiLU(),
        )

        self.time_embed = nn.Sequential(
            SinusoidalTimeEmbedding(dim),
            nn.Linear(dim, dim),
            nn.SiLU(),
            nn.Linear(dim, dim),
        )

        # 5) Transformer decoder (points query image tokens)
        decoder_layer = nn.TransformerDecoderLayer(
            d_model=dim,
            nhead=n_heads,
            dim_feedforward=dim * 2,
            dropout=dropout,
            activation="gelu",
            batch_first=True,
            norm_first=True,
        )
        self.transformer = nn.TransformerDecoder(decoder_layer, num_layers=n_layers)

        # 6) Output head predicts x_start (clean coordinates)
        self.final_head = nn.Sequential(
            nn.LayerNorm(dim),
            nn.Linear(dim, dim // 2),
            nn.SiLU(),
            nn.Linear(dim // 2, 2),
        )

        self._init_weights()

    def _init_weights(self):
        for module in self.modules():
            if isinstance(module, nn.Linear):
                nn.init.xavier_uniform_(module.weight, gain=0.5)
                if module.bias is not None:
                    nn.init.zeros_(module.bias)
            elif isinstance(module, nn.Conv2d):
                nn.init.kaiming_normal_(module.weight, mode="fan_out", nonlinearity="relu")
                if module.bias is not None:
                    nn.init.zeros_(module.bias)

    def _encode_image(self, image: torch.Tensor) -> Tuple[torch.Tensor, int, int]:
        # A. Encode image features
        img_feats = self.img_encoder(image)  # (B, dim, 64, 64) for 512x512 input
        B, C, H, W = img_feats.shape

        # B. Generate Fourier positional grid with explicit 'ij' indexing
        y_raw = torch.linspace(-1, 1, H, device=image.device)
        x_raw = torch.linspace(-1, 1, W, device=image.device)
        grid_y, grid_x = torch.meshgrid(y_raw, x_raw, indexing='ij')
        
        # Scale by π to ensure sine waves complete full cycles (sharper signal)
        emb_y = self.fourier_embed(grid_y.unsqueeze(-1) * 3.14159)
        emb_x = self.fourier_embed(grid_x.unsqueeze(-1) * 3.14159)
        
        # Combine x and y embeddings: (H, W, 128) -> (B, 128, H, W)
        grid = torch.cat([emb_x, emb_y], dim=-1).permute(2, 0, 1).unsqueeze(0).expand(B, -1, -1, -1)
        
        # C. Fuse image features with positional grid
        combined = torch.cat([img_feats, grid], dim=1)  # (B, dim+128, H, W)
        context = combined.flatten(2).transpose(1, 2)    # (B, H*W, dim+128)
        memory = self.fusion(context)                     # (B, H*W, dim)
        
        return memory, H, W

    def forward(self, x_t: torch.Tensor, t: torch.Tensor, image: torch.Tensor) -> torch.Tensor:
        B, N, _ = x_t.shape
        memory, _, _ = self._encode_image(image)

        point_emb = self.point_embed(x_t)
        if t.dim() == 2 and t.shape[1] == 1:
            t = t.squeeze(1)
        time_emb = self.time_embed(t).unsqueeze(1).expand(-1, N, -1)

        query = point_emb + time_emb
        out = self.transformer(tgt=query, memory=memory)
        pred_x0 = self.final_head(out)
        return pred_x0

    def get_num_params(self) -> int:
        return sum(p.numel() for p in self.parameters())


if __name__ == "__main__":
    # V6 Scaled configuration: dim=256, n_layers=6, n_heads=8
    model = PointDiT(n_points=2048, dim=256, n_layers=6, n_heads=8)
    print(f"Point-DiT V6 Scaled (Fourier features, 5M params)")
    print(f"Model parameters: {model.get_num_params():,}")

    x_t = torch.randn(2, 2048, 2)
    t = torch.randint(0, 1000, (2,))
    image = torch.randn(2, 1, 512, 512)

    with torch.no_grad():
        pred = model(x_t, t, image)

    print(f"Input shape: {x_t.shape}")
    print(f"Output shape: {pred.shape}")
    print("✓ Model forward pass successful")
