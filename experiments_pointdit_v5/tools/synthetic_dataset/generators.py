"""Synthetic stippling dataset generators using Voronoi/Lloyd algorithms.

These are for GENERATING synthetic training data, not for loading real datasets.
"""

import torch
from torch.utils.data import Dataset
from pathlib import Path
from PIL import Image
import numpy as np


class VoronoiStipplingDataset(Dataset):
    """Generate ground truth stippling using Lloyd's relaxation (Voronoi)."""
    
    def __init__(
        self,
        source_dir: str,
        image_size: int = 512,
        num_points: int = 2048,
        lloyd_iterations: int = 20,
        density_power: float = 2.0,
    ):
        """
        Args:
            source_dir: Directory with grayscale images
            image_size: Resize images to this size
            num_points: Number of stipple points to generate
            lloyd_iterations: Number of Lloyd relaxation iterations
            density_power: Power for density weighting (higher = more contrast)
        """
        self.source_dir = Path(source_dir)
        self.image_size = image_size
        self.num_points = num_points
        self.lloyd_iterations = lloyd_iterations
        self.density_power = density_power
        
        # Find all images
        self.image_paths = self._list_images(self.source_dir)
        if not self.image_paths:
            raise ValueError(f"No images found in {source_dir}")
        
        print(f"VoronoiStipplingDataset: {len(self.image_paths)} images")
    
    def __len__(self) -> int:
        return len(self.image_paths)
    
    def __getitem__(self, idx: int) -> dict:
        """Generate stippling for an image using Lloyd's algorithm."""
        img_path = self.image_paths[idx]
        
        # Load and preprocess image
        img = Image.open(img_path).convert('L').resize(
            (self.image_size, self.image_size), Image.BILINEAR
        )
        img_np = np.asarray(img, dtype=np.float32) / 255.0
        
        # Generate density map (inverted: dark areas = more points)
        density = 1.0 - img_np
        density = np.power(density, self.density_power)
        density = density / (density.sum() + 1e-6)  # Normalize
        
        # Generate stipple points using weighted Voronoi
        points = self._lloyd_relaxation(density, self.num_points, self.lloyd_iterations)
        
        # Normalize to [-1, 1]
        points_norm = (points / (self.image_size - 1)) * 2.0 - 1.0
        
        return {
            'image': torch.from_numpy(img_np).float().unsqueeze(0),  # (1, H, W)
            'points': torch.from_numpy(points_norm).float(),  # (N, 2)
        }
    
    def _lloyd_relaxation(
        self,
        density: np.ndarray,
        num_points: int,
        iterations: int,
    ) -> np.ndarray:
        """
        Lloyd's algorithm for weighted Voronoi stippling.
        
        Args:
            density: (H, W) density map (higher = more points)
            num_points: Target number of points
            iterations: Number of relaxation iterations
        
        Returns:
            (N, 2) point coordinates in pixel space [0, size-1]
        """
        H, W = density.shape
        
        # Initialize points: weighted random sampling
        flat_density = density.ravel()
        flat_density = flat_density / (flat_density.sum() + 1e-6)
        
        indices = np.random.choice(
            H * W,
            size=num_points,
            replace=False,
            p=flat_density,
        )
        
        ys, xs = np.unravel_index(indices, (H, W))
        points = np.stack([xs, ys], axis=-1).astype(np.float32)
        
        # Lloyd relaxation: iteratively move points to density-weighted centroids
        for _ in range(iterations):
            points = self._lloyd_step(points, density)
        
        return points
    
    def _lloyd_step(
        self,
        points: np.ndarray,
        density: np.ndarray,
    ) -> np.ndarray:
        """
        Single Lloyd relaxation step.
        
        Each point moves to the density-weighted centroid of its Voronoi cell.
        """
        H, W = density.shape
        
        # Build KD-tree equivalent: for each pixel, find nearest point
        ys, xs = np.meshgrid(np.arange(H), np.arange(W), indexing='ij')
        pixels = np.stack([xs.ravel(), ys.ravel()], axis=-1)  # (H*W, 2)
        
        # Compute distances to all points (vectorized)
        # points: (N, 2), pixels: (H*W, 2)
        dists = np.linalg.norm(
            pixels[:, None, :] - points[None, :, :], axis=2
        )  # (H*W, N)
        
        nearest = np.argmin(dists, axis=1)  # (H*W,)
        
        # For each point, compute weighted centroid of its cell
        new_points = np.zeros_like(points)
        weights = density.ravel()
        
        for i in range(len(points)):
            mask = (nearest == i)
            if mask.sum() == 0:
                # No pixels assigned to this point, keep it
                new_points[i] = points[i]
            else:
                cell_pixels = pixels[mask]
                cell_weights = weights[mask]
                # Weighted centroid
                new_points[i] = (cell_pixels * cell_weights[:, None]).sum(axis=0) / (cell_weights.sum() + 1e-6)
        
        # Clamp to image bounds
        new_points[:, 0] = np.clip(new_points[:, 0], 0, W - 1)
        new_points[:, 1] = np.clip(new_points[:, 1], 0, H - 1)
        
        return new_points.astype(np.float32)
    
    @staticmethod
    def _list_images(directory: Path):
        exts = {'.png', '.jpg', '.jpeg', '.bmp', '.tif', '.tiff'}
        return sorted([p for p in directory.iterdir() if p.suffix.lower() in exts])


class FastStipplingDataset(Dataset):
    """Fast approximation: sample from density map without Lloyd relaxation.
    
    This is faster but produces lower-quality stippling (more clumping).
    Use for initial testing; switch to VoronoiStipplingDataset for training.
    """
    
    def __init__(
        self,
        source_dir: str,
        image_size: int = 512,
        num_points: int = 2048,
        density_power: float = 2.0,
        jitter: float = 0.5,
    ):
        self.source_dir = Path(source_dir)
        self.image_size = image_size
        self.num_points = num_points
        self.density_power = density_power
        self.jitter = jitter
        
        self.image_paths = self._list_images(self.source_dir)
        if not self.image_paths:
            raise ValueError(f"No images found in {source_dir}")
        
        print(f"FastStipplingDataset: {len(self.image_paths)} images")
    
    def __len__(self) -> int:
        return len(self.image_paths)
    
    def __getitem__(self, idx: int) -> dict:
        img_path = self.image_paths[idx]
        
        # Load image
        img = Image.open(img_path).convert('L').resize(
            (self.image_size, self.image_size), Image.BILINEAR
        )
        img_np = np.asarray(img, dtype=np.float32) / 255.0
        
        # Density map
        density = 1.0 - img_np
        density = np.power(density, self.density_power)
        density = density / (density.sum() + 1e-6)
        
        # Weighted random sampling
        H, W = density.shape
        flat_density = density.ravel()
        indices = np.random.choice(H * W, size=self.num_points, replace=False, p=flat_density)
        
        ys, xs = np.unravel_index(indices, (H, W))
        
        # Add jitter for smoother distribution
        xs = xs + np.random.uniform(-self.jitter, self.jitter, size=len(xs))
        ys = ys + np.random.uniform(-self.jitter, self.jitter, size=len(ys))
        
        # Clamp and normalize
        xs = np.clip(xs, 0, W - 1)
        ys = np.clip(ys, 0, H - 1)
        points = np.stack([xs, ys], axis=-1).astype(np.float32)
        points_norm = (points / (self.image_size - 1)) * 2.0 - 1.0
        
        return {
            'image': torch.from_numpy(img_np).float().unsqueeze(0),
            'points': torch.from_numpy(points_norm).float(),
        }
    
    @staticmethod
    def _list_images(directory: Path):
        exts = {'.png', '.jpg', '.jpeg', '.bmp', '.tif', '.tiff'}
        return sorted([p for p in directory.iterdir() if p.suffix.lower() in exts])
