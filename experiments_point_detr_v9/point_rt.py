"""Point-RT: Point Refinement Transformer architecture.

Core innovation: Learn 50-step Lloyd's relaxation in a single forward pass.
- Backbone: ResNet18 (pretrained, robust features)
- Encoder: Fourier positional embeddings + point features
- Decoder: Transformer (single pass, not iterative)
- Output: Point displacements (dx, dy)
"""

import math
from typing import Tuple, Optional

import torch
import torch.nn as nn
import torchvision.models as models


class FourierEmbedder(nn.Module):
    """Fourier feature embedder for high-frequency positional encoding.
    
    Gives each spatial location a unique high-frequency fingerprint,
    preventing mode collapse and enabling precise spatial control.
    
    Reused from Point-DiT V5.
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


class PointRT(nn.Module):
    """Point Refinement Transformer - Single-Pass Point Cloud Refinement.
    
    Learns to approximate 50 steps of Lloyd's relaxation in one forward pass.
    
    Comparison to V5:
    - V5: Diffusion model (50 timesteps) → Iterative refinement
    - V7: Direct refinement model (1 pass) → Amortized learning
    
    Architecture:
    1. Image Feature Extraction: ResNet18 backbone (pretrained)
    2. Spatial Context: Fourier grid embeddings
    3. Point Encoding: (x, y, intensity) → embeddings
    4. Transformer Decoder: Single pass refinement
    5. Output Head: Predict (dx, dy) displacement
    """
    
    def __init__(
        self,
        n_points: int = 2048,
        dim: int = 256,
        n_layers: int = 6,
        n_heads: int = 8,
        image_size: int = 512,
        use_resnet_backbone: bool = True,
        resnet_pretrained: bool = True,
        use_displacement: bool = True,
        fourier_freqs: int = 32,
        fourier_temperature: float = 100.0,
        dropout: float = 0.0,
    ):
        super().__init__()
        
        self.n_points = n_points
        self.dim = dim
        self.image_size = image_size
        self.use_displacement = use_displacement
        
        # ===== 1. IMAGE ENCODER: ResNet18 Backbone =====
        if use_resnet_backbone:
            # Pretrained ResNet18 is significantly better than training from scratch
            resnet = models.resnet18(pretrained=resnet_pretrained)
            
            # Remove final pooling and classification layers
            # ResNet18 structure: conv1 + layer1-4 + avgpool + fc
            # We keep up to layer4, which outputs 512 channels
            self.backbone = nn.Sequential(*list(resnet.children())[:-2])
            
            # Project from 512 to dim
            self.feature_proj = nn.Conv2d(512, dim, kernel_size=1)
            backbone_output_channels = dim
        else:
            # Fallback: shallow CNN (similar to V5)
            self.backbone = nn.Sequential(
                nn.Conv2d(1, 32, kernel_size=4, stride=2, padding=1),  # 512 → 256
                nn.GroupNorm(8, 32),
                nn.SiLU(),
                nn.Conv2d(32, 64, kernel_size=4, stride=2, padding=1),  # 256 → 128
                nn.GroupNorm(16, 64),
                nn.SiLU(),
                nn.Conv2d(64, dim, kernel_size=4, stride=2, padding=1),  # 128 → 64
                nn.GroupNorm(32, dim),
                nn.SiLU(),
            )
            self.feature_proj = None
            backbone_output_channels = dim
        
        # ===== 2. POSITIONAL EMBEDDINGS: Fourier =====
        self.fourier_embed = FourierEmbedder(num_freqs=fourier_freqs, temperature=fourier_temperature)
        fourier_dim = fourier_freqs * 2 * 2  # 2(sin/cos) * 2(x/y) * num_freqs
        
        # ===== 3. CONTEXT FUSION WITH LAYERNORM =====
        # Combine backbone features with Fourier positional grids
        self.fusion = nn.Sequential(
            nn.Linear(backbone_output_channels + fourier_dim, dim),
            nn.LayerNorm(dim),
            nn.SiLU(),
        )
        
        # ===== 4. POINT ENCODER =====
        # Input: (x, y, intensity_at_location) → 3-D vector
        self.point_embed = nn.Sequential(
            nn.Linear(3, dim),
            nn.LayerNorm(dim),
            nn.SiLU(),
        )
        
        # ===== 5. TRANSFORMER DECODER (Single Pass) =====
        # Unlike V5 which runs 50 times, this runs ONCE
        # Points are "queries" that attend to image context "memory"
        decoder_layer = nn.TransformerDecoderLayer(
            d_model=dim,
            nhead=n_heads,
            dim_feedforward=dim * 4,
            dropout=dropout,
            activation="gelu",
            batch_first=True,
            norm_first=True,
        )
        self.transformer = nn.TransformerDecoder(decoder_layer, num_layers=n_layers)
        
        # ===== 6. PREDICTION HEAD =====
        # Predicts either:
        # - Displacements (dx, dy) if use_displacement=True
        # - Absolute coordinates if use_displacement=False
        self.final_head = nn.Sequential(
            nn.LayerNorm(dim),
            nn.Linear(dim, dim // 2),
            nn.SiLU(),
            nn.Linear(dim // 2, 2),
        )
        
        # ===== 7. WEIGHT INITIALIZATION =====
        self._init_weights()
    
    def _init_weights(self):
        """Initialize weights for stable training."""
        for module in self.modules():
            if isinstance(module, nn.Linear):
                # Xavier uniform with smaller gain for stability
                nn.init.xavier_uniform_(module.weight, gain=0.5)
                if module.bias is not None:
                    nn.init.zeros_(module.bias)
            elif isinstance(module, nn.Conv2d):
                nn.init.kaiming_normal_(module.weight, mode="fan_out", nonlinearity="relu")
                if module.bias is not None:
                    nn.init.zeros_(module.bias)
    
    def _encode_image(self, image: torch.Tensor) -> Tuple[torch.Tensor, int, int]:
        """Encode image to context features.
        
        Args:
            image: (B, 1, H, W) grayscale image
        
        Returns:
            memory: (B, H*W, dim) image context features
            feat_h, feat_w: Spatial dimensions of feature grid
        """
        B, _, H, W = image.shape
        
        # Step A: Handle ResNet, which expects 3-channel input
        if isinstance(self.backbone, nn.Sequential) and hasattr(self.backbone[0], 'in_channels'):
            # Check first layer (would be Conv2d for ResNet)
            first_layer = self.backbone[0]
            if hasattr(first_layer, 'in_channels') and first_layer.in_channels == 3:
                # ResNet expects 3 channels, repeat grayscale
                image = image.repeat(1, 3, 1, 1)
        
        # Step B: Extract image features via backbone
        # For ResNet18: output is (B, 512, H/32, W/32)
        # For shallow CNN: output is (B, dim, H/8, W/8)
        img_feats = self.backbone(image)
        
        # Project if using ResNet
        if self.feature_proj is not None:
            img_feats = self.feature_proj(img_feats)  # (B, dim, ?, ?)
        
        B, C, feat_h, feat_w = img_feats.shape
        
        # Step C: Generate Fourier positional embeddings for feature grid
        # Create normalized coordinates for each feature location
        y_raw = torch.linspace(-1, 1, feat_h, device=image.device)
        x_raw = torch.linspace(-1, 1, feat_w, device=image.device)
        grid_y, grid_x = torch.meshgrid(y_raw, x_raw, indexing='ij')
        
        # Scale by π to ensure sine waves complete cycles (sharper spatial signal)
        emb_y = self.fourier_embed(grid_y.unsqueeze(-1) * 3.14159)  # (feat_h, feat_w, fourier_dim)
        emb_x = self.fourier_embed(grid_x.unsqueeze(-1) * 3.14159)  # (feat_h, feat_w, fourier_dim)
        
        # Combine x and y embeddings
        grid = torch.cat([emb_x, emb_y], dim=-1)  # (feat_h, feat_w, fourier_dim*2)
        grid = grid.permute(2, 0, 1).unsqueeze(0).expand(B, -1, -1, -1)  # (B, fourier_dim*2, feat_h, feat_w)
        
        # Step D: Fuse image features with positional embeddings
        combined = torch.cat([img_feats, grid], dim=1)  # (B, C+fourier_dim*2, feat_h, feat_w)
        
        # Flatten spatial dimensions
        context = combined.flatten(2).transpose(1, 2)  # (B, feat_h*feat_w, C+fourier_dim*2)
        
        # Apply fusion layer (learn weighted combination)
        memory = self.fusion(context)  # (B, feat_h*feat_w, dim)
        
        return memory, feat_h, feat_w
    
    def forward(
        self,
        x_init: torch.Tensor,
        image: torch.Tensor,
    ) -> torch.Tensor:
        """Single-pass refinement of point clouds.
        
        Args:
            x_init: (B, N, 3) initialized points with structure [x, y, intensity]
                   where intensity is sampled from image at (x, y)
            image: (B, 1, H, W) conditioning image context
        
        Returns:
            refined_points: (B, N, 2) refined point coordinates
                           - If use_displacement: x_init[:, :, :2] + displacement
                           - If not: direct coordinates
        """
        B, N, _ = x_init.shape
        
        # Step A: Encode image to context
        memory, feat_h, feat_w = self._encode_image(image)
        
        # Step B: Encode points
        query = self.point_embed(x_init)  # (B, N, dim)
        
        # Step C: Single transformer pass (THE KEY DIFFERENCE FROM V5)
        # No timestep loop! Just one forward pass.
        refined_features = self.transformer(tgt=query, memory=memory)  # (B, N, dim)
        
        # Step D: Predict displacements or coordinates
        output = self.final_head(refined_features)  # (B, N, 2)
        
        # Step E: Compose final result
        if self.use_displacement:
            # Add displacement to initial points
            refined_points = x_init[:, :, :2] + output
        else:
            # Use predicted coordinates directly (then clamp to valid range)
            refined_points = torch.clamp(output, min=-1.0, max=1.0)
        
        return refined_points
    
    def get_num_params(self) -> int:
        """Return total number of learnable parameters."""
        return sum(p.numel() for p in self.parameters())
    
    def freeze_backbone(self, freeze: bool = True):
        """Optionally freeze ResNet backbone to preserve pretrained features."""
        if hasattr(self, 'backbone'):
            for param in self.backbone.parameters():
                param.requires_grad = not freeze
            if self.feature_proj is not None:
                for param in self.feature_proj.parameters():
                    param.requires_grad = not freeze


if __name__ == "__main__":
    print("Testing Point-RT model...")
    
    # Test configuration
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Using device: {device}")
    
    # Create model
    model = PointRT(
        n_points=2048,
        dim=256,
        n_layers=6,
        n_heads=8,
        use_resnet_backbone=True,
        resnet_pretrained=False,  # Don't download weights for quick test
    )
    model = model.to(device)
    
    print(f"Model Parameters: {model.get_num_params():,}")
    print(f"Model: {model}")
    
    # Create mock inputs
    B, N, H, W = 2, 2048, 512, 512
    
    # Initialize points with fast_density_initialization (simulate)
    x_init = torch.randn(B, N, 2, device=device) * 0.5  # Random init
    x_init = torch.clamp(x_init, -1.0, 1.0)
    
    # Add intensity (simulate sampling from image)
    intensities = torch.rand(B, N, 1, device=device)
    x_init_with_intensity = torch.cat([x_init, intensities], dim=-1)
    
    # Create mock image
    image = torch.rand(B, 1, H, W, device=device)
    
    # Forward pass
    print("\nForward pass test:")
    with torch.no_grad():
        output = model(x_init_with_intensity, image)
    
    print(f"Input shape: {x_init_with_intensity.shape}")
    print(f"Image shape: {image.shape}")
    print(f"Output shape: {output.shape}")
    print(f"Output range: [{output.min():.3f}, {output.max():.3f}]")
    
    # Check computation
    print("\n✓ Model forward pass successful!")
    print("\nKey differences from V5:")
    print("  - Single transformer pass (no diffusion loop)")
    print("  - ResNet18 backbone (pretrained features)")
    print("  - Input: (x, y, intensity) instead of (x, y, t)")
    print("  - Output: displacement (dx, dy) instead of x_t prediction")
