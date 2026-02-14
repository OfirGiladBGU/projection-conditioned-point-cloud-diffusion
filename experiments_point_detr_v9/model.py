"""Point Set Refinement model (iterative, physics-inspired)."""

from typing import Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision.models as models

from layers import RefinementBlock


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


class StippleRefiner(nn.Module):
    """Iterative point set refinement with unrolled physics steps."""

    def __init__(
        self,
        n_points: int = 2048,
        dim: int = 256,
        n_layers: int = 6,
        n_heads: int = 8,
        image_size: int = 512,
        use_resnet_backbone: bool = True,
        resnet_pretrained: bool = True,
        fourier_freqs: int = 32,
        fourier_temperature: float = 100.0,
        dropout: float = 0.0,
        n_refine_steps: int = 5,
        share_weights: bool = True,
        delta_scale: float = 0.1,
    ):
        super().__init__()

        self.n_points = n_points
        self.dim = dim
        self.image_size = image_size
        self.n_refine_steps = n_refine_steps
        self.share_weights = share_weights
        self.delta_scale = delta_scale

        # ===== 1. IMAGE ENCODER =====
        if use_resnet_backbone:
            resnet = models.resnet18(pretrained=resnet_pretrained)
            self.backbone = nn.Sequential(*list(resnet.children())[:-2])
            self.feature_proj = nn.Conv2d(512, dim, kernel_size=1)
            backbone_output_channels = dim
        else:
            self.backbone = nn.Sequential(
                nn.Conv2d(1, 32, kernel_size=4, stride=2, padding=1),
                nn.GroupNorm(8, 32),
                nn.SiLU(),
                nn.Conv2d(32, 64, kernel_size=4, stride=2, padding=1),
                nn.GroupNorm(16, 64),
                nn.SiLU(),
                nn.Conv2d(64, dim, kernel_size=4, stride=2, padding=1),
                nn.GroupNorm(32, dim),
                nn.SiLU(),
            )
            self.feature_proj = None
            backbone_output_channels = dim

        # ===== 2. POSITIONAL EMBEDDINGS =====
        self.fourier_embed = FourierEmbedder(num_freqs=fourier_freqs, temperature=fourier_temperature)
        fourier_dim = fourier_freqs * 2 * 2

        self.fusion = nn.Sequential(
            nn.Linear(backbone_output_channels + fourier_dim, dim),
            nn.LayerNorm(dim),
            nn.SiLU(),
        )

        # ===== 3. POINT EMBEDDING =====
        self.point_embed = nn.Sequential(
            nn.Linear(3, dim),
            nn.LayerNorm(dim),
            nn.SiLU(),
        )

        # ===== 4. REFINEMENT BLOCKS =====
        if share_weights:
            self.blocks = nn.ModuleList(
                [RefinementBlock(dim=dim, n_heads=n_heads, dropout=dropout) for _ in range(n_layers)]
            )
            self.step_blocks = None
        else:
            self.blocks = None
            self.step_blocks = nn.ModuleList(
                [
                    nn.ModuleList(
                        [RefinementBlock(dim=dim, n_heads=n_heads, dropout=dropout) for _ in range(n_layers)]
                    )
                    for _ in range(n_refine_steps)
                ]
            )

        # ===== 5. DELTA HEAD =====
        self.to_delta = nn.Sequential(
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
        B, _, H, W = image.shape

        if isinstance(self.backbone, nn.Sequential) and hasattr(self.backbone[0], "in_channels"):
            first_layer = self.backbone[0]
            if hasattr(first_layer, "in_channels") and first_layer.in_channels == 3:
                image = image.repeat(1, 3, 1, 1)

        img_feats = self.backbone(image)
        if self.feature_proj is not None:
            img_feats = self.feature_proj(img_feats)

        B, C, feat_h, feat_w = img_feats.shape

        y_raw = torch.linspace(-1, 1, feat_h, device=image.device)
        x_raw = torch.linspace(-1, 1, feat_w, device=image.device)
        grid_y, grid_x = torch.meshgrid(y_raw, x_raw, indexing="ij")

        emb_y = self.fourier_embed(grid_y.unsqueeze(-1) * 3.14159)
        emb_x = self.fourier_embed(grid_x.unsqueeze(-1) * 3.14159)

        grid = torch.cat([emb_x, emb_y], dim=-1)
        grid = grid.permute(2, 0, 1).unsqueeze(0).expand(B, -1, -1, -1)

        combined = torch.cat([img_feats, grid], dim=1)
        context = combined.flatten(2).transpose(1, 2)
        memory = self.fusion(context)

        return memory, feat_h, feat_w

    def _sample_intensity(self, points: torch.Tensor, image: torch.Tensor) -> torch.Tensor:
        grid = points.unsqueeze(2)
        intensities = F.grid_sample(
            image,
            grid,
            mode="bilinear",
            padding_mode="border",
            align_corners=True,
        ).squeeze(-1).squeeze(1)
        return intensities.unsqueeze(-1)

    def forward(self, x_init: torch.Tensor, image: torch.Tensor) -> torch.Tensor:
        if x_init.shape[-1] == 3:
            current_points = x_init[:, :, :2]
        else:
            current_points = x_init

        memory, _, _ = self._encode_image(image)

        for step in range(self.n_refine_steps):
            intensities = self._sample_intensity(current_points, image)
            point_input = torch.cat([current_points, intensities], dim=-1)
            point_features = self.point_embed(point_input)

            if self.share_weights:
                blocks = self.blocks
            else:
                blocks = self.step_blocks[step]

            for block in blocks:
                point_features = block(point_features, memory)

            delta = self.to_delta(point_features)
            if self.delta_scale is not None:
                delta = delta * self.delta_scale
            current_points = current_points + delta
            current_points = torch.clamp(current_points, min=-1.0, max=1.0)

        return current_points

    def get_num_params(self) -> int:
        return sum(p.numel() for p in self.parameters())

    def freeze_backbone(self, freeze: bool = True):
        if hasattr(self, "backbone"):
            for param in self.backbone.parameters():
                param.requires_grad = not freeze
            if self.feature_proj is not None:
                for param in self.feature_proj.parameters():
                    param.requires_grad = not freeze
