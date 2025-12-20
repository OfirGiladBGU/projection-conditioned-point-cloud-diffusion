"""
Lightweight PyTorch3D compatibility layer.

If the real `pytorch3d` package is installed with working C++/CUDA extensions,
this module simply re-exports its classes. If the build is unavailable, we
provide minimal, CUDA-capable Python fallbacks implemented with plain PyTorch
so the rest of the codebase can run without PyTorch3D.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Optional, Sequence, Tuple

import torch

try:  # Prefer the real library when available
    from pytorch3d.structures import Pointclouds as _Pointclouds
    from pytorch3d.renderer import (
        PointsRasterizer as _PointsRasterizer,
        PointsRasterizationSettings as _PointsRasterizationSettings,
    )
    from pytorch3d.renderer.cameras import CamerasBase as _CamerasBase
    from pytorch3d.renderer.cameras import PerspectiveCameras as _PerspectiveCameras
    from pytorch3d.implicitron.dataset.data_loader_map_provider import FrameData as _FrameData

    PYTORCH3D_AVAILABLE = True

    Pointclouds = _Pointclouds  # type: ignore
    PointsRasterizer = _PointsRasterizer  # type: ignore
    PointsRasterizationSettings = _PointsRasterizationSettings  # type: ignore
    CamerasBase = _CamerasBase  # type: ignore
    PerspectiveCameras = _PerspectiveCameras  # type: ignore
    FrameData = _FrameData  # type: ignore

except Exception:  # pragma: no cover - executed when PyTorch3D is missing
    PYTORCH3D_AVAILABLE = False

    class CamerasBase:
        """Very small camera container used for simple projection math."""

        def __init__(
            self,
            R: Optional[torch.Tensor] = None,
            T: Optional[torch.Tensor] = None,
            image_size: Optional[Sequence[int] | torch.Tensor] = None,
            device: Optional[torch.device | str] = None,
        ) -> None:
            device = torch.device(device) if device is not None else None
            self.R = torch.eye(3, device=device).unsqueeze(0) if R is None else R
            self.T = torch.zeros(1, 3, device=device) if T is None else T
            if image_size is None:
                self.image_size = None
            else:
                if isinstance(image_size, torch.Tensor):
                    self.image_size = image_size.to(device)
                else:
                    self.image_size = torch.tensor(image_size, device=device)
            self.device = device if device is not None else self.R.device

        def to(self, device: torch.device | str) -> "CamerasBase":
            device = torch.device(device)
            return self.__class__(
                R=self.R.to(device),
                T=self.T.to(device),
                image_size=self.image_size.to(device) if self.image_size is not None else None,
                device=device,
            )

        def clone(self) -> "CamerasBase":
            return self.to(self.device)

    class PerspectiveCameras(CamerasBase):
        """Alias for compatibility with the original API."""

        pass

    class Pointclouds:
        """Minimal Pointcloud container supporting GPU tensors."""

        def __init__(self, points, features: Optional[torch.Tensor] = None) -> None:
            if isinstance(points, (list, tuple)):
                points = torch.stack(points, dim=0)
            if points.dim() == 2:
                points = points.unsqueeze(0)
            self._points = points

            if features is not None and isinstance(features, (list, tuple)):
                features = torch.stack(features, dim=0)
            if features is not None and features.dim() == 2:
                features = features.unsqueeze(0)
            self._features = features

        def to(self, device: torch.device | str) -> "Pointclouds":
            device = torch.device(device)
            feats = self._features.to(device) if self._features is not None else None
            return Pointclouds(self._points.to(device), feats)

        def points_padded(self) -> torch.Tensor:
            return self._points

        def points_packed(self) -> torch.Tensor:
            return self._points.reshape(-1, self._points.shape[-1])

        def features_padded(self) -> Optional[torch.Tensor]:
            return self._features

    @dataclass
    class FrameData:
        sequence_point_cloud: Pointclouds
        camera: Optional[CamerasBase] = None
        image_rgb: Optional[torch.Tensor] = None
        fg_probability: Optional[torch.Tensor] = None

    @dataclass
    class PointsRasterizationSettings:
        image_size: Tuple[int, int]
        radius: float
        points_per_pixel: int
        bin_size: int = 0

    class PointsRasterizer:
        """Very small point rasterizer approximation using pure PyTorch.

        The goal is API compatibility, not accurate rendering. Points are mapped
        to pixel centers after min-max normalization in XY and indexed per pixel.
        """

        def __init__(self, cameras: CamerasBase, raster_settings: PointsRasterizationSettings) -> None:
            self.cameras = cameras
            self.raster_settings = raster_settings

        def __call__(self, point_clouds: Pointclouds):
            points = point_clouds.points_padded()  # (B, N, 3)
            B, _, _ = points.shape
            H, W = self.raster_settings.image_size
            R = self.raster_settings.points_per_pixel
            device = points.device

            idx = torch.full((B, H, W, R), -1, device=device, dtype=torch.long)
            for b in range(B):
                pts = points[b]
                xy = pts[:, :2]
                mins = xy.min(0).values
                maxs = xy.max(0).values
                scale = torch.clamp(maxs - mins, min=1e-6)
                xy_norm = (xy - mins) / scale
                px = torch.clamp((xy_norm[:, 0] * (W - 1)).long(), 0, W - 1)
                py = torch.clamp(((1.0 - xy_norm[:, 1]) * (H - 1)).long(), 0, H - 1)
                lin = py * W + px
                flat = idx[b].view(-1, R)
                flat[lin, 0] = torch.arange(len(pts), device=device)

            class _Fragments:
                def __init__(self, idx_tensor: torch.Tensor) -> None:
                    self.idx = idx_tensor

            return _Fragments(idx)

__all__ = [
    "PYTORCH3D_AVAILABLE",
    "Pointclouds",
    "PointsRasterizer",
    "PointsRasterizationSettings",
    "CamerasBase",
    "PerspectiveCameras",
    "FrameData",
]
