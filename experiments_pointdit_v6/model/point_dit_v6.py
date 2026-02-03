"""Point-DiT V6: Enhanced capacity model for better blue noise quality.

Key improvements over V5:
1. Deeper image encoder (5 layers → richer spatial features)
2. Larger transformer (dim=256, 6 layers, 8 heads)
3. Multi-scale image features (skip connections from encoder)
4. Improved cross-attention with more capacity
5. Gradient checkpointing option for memory efficiency

V6.1 Enhancements (Tactile Density Sensors):
6. Direct density input: Points receive local pixel intensity directly
   - Concatenate image intensity at point location to point coordinates
   - Input becomes (x, y, intensity) instead of just (x, y)
   - Points immediately know "I am in a dark/light region"
   - Creates stronger gradient for local spacing adjustment

Target: Close the gap between direct optimization (0.75) and learned model (0.67)
"""

import math
from typing import Tuple, Optional

import torch
import torch.nn as nn
import torch.nn.functional as F


class FourierEmbedder(nn.Module):
    """Fourier feature embedder for high-frequency positional encoding."""
    
    def __init__(self, num_freqs: int = 32, temperature: float = 100.0):
        super().__init__()
        self.num_freqs = num_freqs
        self.temperature = temperature
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        exponent = torch.arange(self.num_freqs, dtype=x.dtype, device=x.device)
        freqs = self.temperature ** (exponent / self.num_freqs)
        args = x * freqs.view(1, -1)
        return torch.cat([torch.sin(args), torch.cos(args)], dim=-1)


class SinusoidalTimeEmbedding(nn.Module):
    """Sinusoidal time embedding."""

    def __init__(self, dim: int):
        super().__init__()
        self.dim = dim

    def forward(self, t: torch.Tensor) -> torch.Tensor:
        if t.dim() == 1:
            t = t.unsqueeze(-1)
        half_dim = self.dim // 2
        emb = math.log(10000) / (half_dim - 1)
        emb = torch.exp(torch.arange(half_dim, device=t.device) * -emb)
        emb = t * emb.unsqueeze(0)
        return torch.cat([torch.sin(emb), torch.cos(emb)], dim=-1)


class ConvBlock(nn.Module):
    """Convolutional block with GroupNorm and residual connection."""
    
    def __init__(self, in_ch: int, out_ch: int, stride: int = 1, use_residual: bool = True):
        super().__init__()
        self.use_residual = use_residual and (in_ch == out_ch and stride == 1)
        
        self.conv = nn.Sequential(
            nn.Conv2d(in_ch, out_ch, kernel_size=3, stride=stride, padding=1),
            nn.GroupNorm(min(32, out_ch), out_ch),
            nn.SiLU(),
            nn.Conv2d(out_ch, out_ch, kernel_size=3, stride=1, padding=1),
            nn.GroupNorm(min(32, out_ch), out_ch),
        )
        self.act = nn.SiLU()
        
        if stride > 1 or in_ch != out_ch:
            self.downsample = nn.Conv2d(in_ch, out_ch, kernel_size=1, stride=stride)
        else:
            self.downsample = None
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        out = self.conv(x)
        if self.downsample is not None:
            x = self.downsample(x)
        if self.use_residual:
            out = out + x
        return self.act(out)


class MultiScaleImageEncoder(nn.Module):
    """
    Deep multi-scale image encoder with skip connections.
    
    Provides rich hierarchical features at multiple resolutions.
    """
    
    def __init__(self, out_dim: int = 256, base_ch: int = 64):
        super().__init__()
        
        # Progressive downsampling: 512 → 256 → 128 → 64 → 32
        self.stem = nn.Sequential(
            nn.Conv2d(1, base_ch, kernel_size=7, stride=2, padding=3),  # 512 → 256
            nn.GroupNorm(8, base_ch),
            nn.SiLU(),
        )
        
        self.stage1 = ConvBlock(base_ch, base_ch * 2, stride=2)      # 256 → 128
        self.stage2 = ConvBlock(base_ch * 2, base_ch * 4, stride=2)  # 128 → 64
        self.stage3 = ConvBlock(base_ch * 4, base_ch * 4, stride=1)  # 64 → 64 (deeper)
        self.stage4 = ConvBlock(base_ch * 4, out_dim, stride=1)      # 64 → 64, final dim
        
        # Multi-scale fusion: combine features from different scales
        self.scale_proj = nn.ModuleList([
            nn.Conv2d(base_ch * 2, out_dim, kernel_size=1),  # From stage1 (128x128)
            nn.Conv2d(base_ch * 4, out_dim, kernel_size=1),  # From stage2 (64x64)
        ])
    
    def forward(self, x: torch.Tensor) -> Tuple[torch.Tensor, list]:
        """
        Returns:
            main_features: (B, out_dim, 64, 64) - primary features
            multi_scale: list of features at different scales for skip connections
        """
        x = self.stem(x)           # (B, 64, 256, 256)
        s1 = self.stage1(x)        # (B, 128, 128, 128)
        s2 = self.stage2(s1)       # (B, 256, 64, 64)
        s3 = self.stage3(s2)       # (B, 256, 64, 64)
        main = self.stage4(s3)     # (B, out_dim, 64, 64)
        
        # Project multi-scale features to same dimension
        multi_scale = [
            F.interpolate(self.scale_proj[0](s1), size=main.shape[2:], mode='bilinear', align_corners=False),
            self.scale_proj[1](s2),
        ]
        
        return main, multi_scale


class EnhancedTransformerBlock(nn.Module):
    """
    Enhanced transformer block with:
    - Pre-LayerNorm (more stable training)
    - Larger feedforward (4x instead of 2x)
    - Separate self-attention and cross-attention
    """
    
    def __init__(self, dim: int, n_heads: int, dropout: float = 0.0, ff_mult: int = 4):
        super().__init__()
        
        # Self-attention (points attend to each other)
        self.norm1 = nn.LayerNorm(dim)
        self.self_attn = nn.MultiheadAttention(dim, n_heads, dropout=dropout, batch_first=True)
        
        # Cross-attention (points attend to image features)
        self.norm2 = nn.LayerNorm(dim)
        self.cross_attn = nn.MultiheadAttention(dim, n_heads, dropout=dropout, batch_first=True)
        self.norm_memory = nn.LayerNorm(dim)
        
        # Feedforward
        self.norm3 = nn.LayerNorm(dim)
        self.ff = nn.Sequential(
            nn.Linear(dim, dim * ff_mult),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(dim * ff_mult, dim),
            nn.Dropout(dropout),
        )
    
    def forward(self, x: torch.Tensor, memory: torch.Tensor) -> torch.Tensor:
        # Self-attention
        x_norm = self.norm1(x)
        x = x + self.self_attn(x_norm, x_norm, x_norm, need_weights=False)[0]
        
        # Cross-attention
        x_norm = self.norm2(x)
        mem_norm = self.norm_memory(memory)
        x = x + self.cross_attn(x_norm, mem_norm, mem_norm, need_weights=False)[0]
        
        # Feedforward
        x = x + self.ff(self.norm3(x))
        
        return x


class PointDiTV6(nn.Module):
    """
    Point-DiT V6: Enhanced capacity model with Tactile Density Sensors.
    
    Changes from V5:
    - dim: 128 → 256 (2x)
    - n_layers: 4 → 6 (1.5x)
    - n_heads: 4 → 8 (2x)
    - Image encoder: 3 stages → 5 stages with multi-scale
    - Feedforward: 2x → 4x dim
    - Separate self/cross attention
    
    V6.1 Enhancement: Tactile Density Input
    - Points receive local pixel intensity directly concatenated to coordinates
    - Input: (x, y, intensity) → dim=3 instead of dim=2
    - Benefit: Points immediately know their local density requirement
    - Result: Much stronger gradient for spacing adjustment
    
    Expected params: ~4M (vs 1M in V5)
    """

    def __init__(
        self,
        n_points: int = 5000,
        dim: int = 256,
        n_layers: int = 6,
        n_heads: int = 8,
        image_size: int = 512,
        dropout: float = 0.0,
        use_multi_scale: bool = True,
        use_density_input: bool = True,  # NEW: Tactile density sensors
    ):
        super().__init__()

        self.n_points = n_points
        self.dim = dim
        self.image_size = image_size
        self.use_multi_scale = use_multi_scale
        self.use_density_input = use_density_input  # NEW

        # 1) Deep multi-scale image encoder
        self.img_encoder = MultiScaleImageEncoder(out_dim=dim, base_ch=64)
        
        # 2) Fourier positional embeddings
        self.fourier_embed = FourierEmbedder(num_freqs=32, temperature=100.0)
        grid_dim = 32 * 2 * 2  # 128
        
        # 3) Context fusion
        # If using multi-scale, we have main + 2 skip connections = 3x dim
        fusion_in = dim * 3 + grid_dim if use_multi_scale else dim + grid_dim
        self.fusion = nn.Sequential(
            nn.Linear(fusion_in, dim),
            nn.LayerNorm(dim),
            nn.GELU(),
            nn.Linear(dim, dim),
        )

        # 4) Point embedding (with larger capacity)
        # NEW: Accept 3D input (x, y, intensity) if using density sensors
        point_input_dim = 3 if use_density_input else 2
        self.point_embed = nn.Sequential(
            nn.Linear(point_input_dim, dim),
            nn.LayerNorm(dim),
            nn.GELU(),
            nn.Linear(dim, dim),
        )

        # 5) Time embedding (larger MLP)
        self.time_embed = nn.Sequential(
            SinusoidalTimeEmbedding(dim),
            nn.Linear(dim, dim * 2),
            nn.GELU(),
            nn.Linear(dim * 2, dim),
        )

        # 6) Enhanced transformer layers
        self.transformer_layers = nn.ModuleList([
            EnhancedTransformerBlock(dim, n_heads, dropout=dropout, ff_mult=4)
            for _ in range(n_layers)
        ])

        # 7) Output head (larger)
        self.final_head = nn.Sequential(
            nn.LayerNorm(dim),
            nn.Linear(dim, dim),
            nn.GELU(),
            nn.Linear(dim, dim // 2),
            nn.GELU(),
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

    def _encode_image(self, image: torch.Tensor) -> torch.Tensor:
        B = image.shape[0]
        
        # Get multi-scale features
        main_feats, multi_scale = self.img_encoder(image)  # (B, dim, 64, 64)
        _, _, H, W = main_feats.shape

        # Fourier positional grid
        y_raw = torch.linspace(-1, 1, H, device=image.device)
        x_raw = torch.linspace(-1, 1, W, device=image.device)
        grid_y, grid_x = torch.meshgrid(y_raw, x_raw, indexing='ij')
        
        emb_y = self.fourier_embed(grid_y.unsqueeze(-1) * 3.14159)
        emb_x = self.fourier_embed(grid_x.unsqueeze(-1) * 3.14159)
        grid = torch.cat([emb_x, emb_y], dim=-1).permute(2, 0, 1).unsqueeze(0).expand(B, -1, -1, -1)
        
        # Combine all features
        if self.use_multi_scale:
            combined = torch.cat([main_feats, multi_scale[0], multi_scale[1], grid], dim=1)
        else:
            combined = torch.cat([main_feats, grid], dim=1)
        
        context = combined.flatten(2).transpose(1, 2)  # (B, H*W, fusion_in)
        memory = self.fusion(context)  # (B, H*W, dim)
        
        return memory
    
    def _sample_density_at_points(self, x_t: torch.Tensor, image: torch.Tensor) -> torch.Tensor:
        """Sample image intensity at each point location.
        
        This is the "Tactile Sensor" - each point immediately knows the local density.
        
        Args:
            x_t: (B, N, 2) point coordinates in [-1, 1]
            image: (B, 1, H, W) image in [0, 1]
        
        Returns:
            intensity: (B, N, 1) sampled intensity at each point
        """
        # grid_sample expects (B, N, 1, 2) for 2D points, output (B, 1, N, 1)
        grid = x_t.unsqueeze(2)  # (B, N, 1, 2)
        
        # Sample using bilinear interpolation
        # Note: grid_sample expects (x, y) in [-1, 1] which matches our coordinate system
        intensity = F.grid_sample(
            image, grid, 
            mode='bilinear', 
            padding_mode='border', 
            align_corners=True
        )  # (B, 1, N, 1)
        
        # Reshape to (B, N, 1)
        intensity = intensity.squeeze(-1).transpose(1, 2)  # (B, N, 1)
        
        return intensity

    def forward(self, x_t: torch.Tensor, t: torch.Tensor, image: torch.Tensor) -> torch.Tensor:
        B, N, _ = x_t.shape
        
        # Encode image
        memory = self._encode_image(image)

        # NEW: Tactile Density Input
        # Sample image intensity at each point and concatenate to coordinates
        if self.use_density_input:
            intensity = self._sample_density_at_points(x_t, image)  # (B, N, 1)
            x_t_input = torch.cat([x_t, intensity], dim=-1)  # (B, N, 3)
        else:
            x_t_input = x_t  # (B, N, 2)
        
        # Encode points (now with density info if enabled)
        point_emb = self.point_embed(x_t_input)
        
        # Encode time
        if t.dim() == 2 and t.shape[1] == 1:
            t = t.squeeze(1)
        time_emb = self.time_embed(t).unsqueeze(1).expand(-1, N, -1)

        # Initial query
        query = point_emb + time_emb
        
        # Apply transformer layers
        for layer in self.transformer_layers:
            query = layer(query, memory)
        
        # Predict clean coordinates
        pred_x0 = self.final_head(query)
        return pred_x0

    def get_num_params(self) -> int:
        return sum(p.numel() for p in self.parameters())


# Also provide a "medium" variant that's between V5 and V6
class PointDiTV5_Medium(nn.Module):
    """
    Medium capacity model (V5.5): Safer upgrade from V5.
    
    Changes from V5:
    - dim: 128 → 192 (1.5x)
    - n_layers: 4 → 5
    - n_heads: 4 → 6
    - Feedforward: 2x → 3x
    
    Expected params: ~2M (vs 1M in V5, 4M in V6)
    """

    def __init__(
        self,
        n_points: int = 5000,
        dim: int = 192,
        n_layers: int = 5,
        n_heads: int = 6,
        image_size: int = 512,
        dropout: float = 0.0,
    ):
        super().__init__()

        self.n_points = n_points
        self.dim = dim
        self.image_size = image_size

        # Same image encoder as V5 but with larger output
        self.img_encoder = nn.Sequential(
            nn.Conv2d(1, 48, kernel_size=4, stride=2, padding=1),
            nn.GroupNorm(8, 48),
            nn.SiLU(),
            nn.Conv2d(48, 96, kernel_size=4, stride=2, padding=1),
            nn.GroupNorm(16, 96),
            nn.SiLU(),
            nn.Conv2d(96, dim, kernel_size=4, stride=2, padding=1),
            nn.GroupNorm(32, dim),
            nn.SiLU(),
        )
        
        self.fourier_embed = FourierEmbedder(num_freqs=32, temperature=100.0)
        grid_dim = 128
        
        self.fusion = nn.Sequential(
            nn.Linear(dim + grid_dim, dim),
            nn.LayerNorm(dim),
            nn.SiLU()
        )

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

        decoder_layer = nn.TransformerDecoderLayer(
            d_model=dim,
            nhead=n_heads,
            dim_feedforward=dim * 3,  # 3x instead of 2x
            dropout=dropout,
            activation="gelu",
            batch_first=True,
            norm_first=True,
        )
        self.transformer = nn.TransformerDecoder(decoder_layer, num_layers=n_layers)

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

    def _encode_image(self, image: torch.Tensor) -> torch.Tensor:
        B = image.shape[0]
        img_feats = self.img_encoder(image)
        _, _, H, W = img_feats.shape

        y_raw = torch.linspace(-1, 1, H, device=image.device)
        x_raw = torch.linspace(-1, 1, W, device=image.device)
        grid_y, grid_x = torch.meshgrid(y_raw, x_raw, indexing='ij')
        
        emb_y = self.fourier_embed(grid_y.unsqueeze(-1) * 3.14159)
        emb_x = self.fourier_embed(grid_x.unsqueeze(-1) * 3.14159)
        grid = torch.cat([emb_x, emb_y], dim=-1).permute(2, 0, 1).unsqueeze(0).expand(B, -1, -1, -1)
        
        combined = torch.cat([img_feats, grid], dim=1)
        context = combined.flatten(2).transpose(1, 2)
        return self.fusion(context)

    def forward(self, x_t: torch.Tensor, t: torch.Tensor, image: torch.Tensor) -> torch.Tensor:
        B, N, _ = x_t.shape
        memory = self._encode_image(image)

        point_emb = self.point_embed(x_t)
        if t.dim() == 2 and t.shape[1] == 1:
            t = t.squeeze(1)
        time_emb = self.time_embed(t).unsqueeze(1).expand(-1, N, -1)

        query = point_emb + time_emb
        out = self.transformer(tgt=query, memory=memory)
        return self.final_head(out)

    def get_num_params(self) -> int:
        return sum(p.numel() for p in self.parameters())


if __name__ == "__main__":
    print("Model Comparison:")
    print("-" * 50)
    
    # V5 (original)
    from point_dit import PointDiT
    v5 = PointDiT(n_points=5000, dim=128, n_layers=4, n_heads=4)
    print(f"V5 (Original):  {v5.get_num_params():>10,} params")
    
    # V5 Medium
    v5m = PointDiTV5_Medium(n_points=5000, dim=192, n_layers=5, n_heads=6)
    print(f"V5.5 (Medium):  {v5m.get_num_params():>10,} params")
    
    # V6 (full) - without density input for comparison
    v6_no_density = PointDiTV6(n_points=5000, dim=256, n_layers=6, n_heads=8, use_density_input=False)
    print(f"V6 (No Density): {v6_no_density.get_num_params():>10,} params")
    
    # V6.1 (full with density input)
    v6 = PointDiTV6(n_points=5000, dim=256, n_layers=6, n_heads=8, use_density_input=True)
    print(f"V6.1 (Density):  {v6.get_num_params():>10,} params")
    
    # Test forward pass
    print("\nTesting forward pass (V6.1 with density input)...")
    x = torch.randn(2, 5000, 2)
    t = torch.randint(0, 1000, (2,))
    img = torch.rand(2, 1, 512, 512)  # Use rand for [0,1] range
    out = v6(x, t, img)
    print(f"Output shape: {out.shape}")
    
    # Verify density sampling works
    print("\nTesting density sampling...")
    grid = x[:, :10, :].unsqueeze(2)  # First 10 points
    intensity = torch.nn.functional.grid_sample(
        img, grid, mode='bilinear', padding_mode='border', align_corners=True
    ).squeeze(-1).transpose(1, 2)
    print(f"Sampled intensities shape: {intensity.shape}")
    print(f"Sample values: {intensity[0, :5, 0].tolist()}")
