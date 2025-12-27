"""Dataset with Voronoi stippling / Lloyd's algorithm for blue noise ground truth.

This generates high-quality stipple distributions that the model learns to replicate.
"""

import torch
from torch.utils.data import Dataset, DataLoader
from pathlib import Path
from PIL import Image
import numpy as np
from typing import Literal, Union, Tuple
from scipy.spatial import Voronoi
import cv2


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


def create_dataloaders(
    source_dir: str,
    batch_size: int = 4,
    num_workers: int = 4,
    val_split: float = 0.1,
    dataset_type: Literal['voronoi', 'fast'] = 'voronoi',
    **dataset_kwargs,
) -> Tuple[DataLoader, DataLoader]:
    """Create train and validation dataloaders."""
    
    # Select dataset and filter kwargs
    if dataset_type == 'voronoi':
        dataset_cls = VoronoiStipplingDataset
        # VoronoiStipplingDataset accepts: image_size, num_points, lloyd_iterations, density_power
        valid_keys = {'image_size', 'num_points', 'lloyd_iterations', 'density_power'}
    else:
        dataset_cls = FastStipplingDataset
        # FastStipplingDataset accepts: image_size, num_points, density_power, jitter
        valid_keys = {'image_size', 'num_points', 'density_power', 'jitter'}
    
    # Filter kwargs to only include valid parameters for this dataset
    filtered_kwargs = {k: v for k, v in dataset_kwargs.items() if k in valid_keys}
    
    # Create full dataset
    full_dataset = dataset_cls(source_dir=source_dir, **filtered_kwargs)
    
    # Split train/val
    dataset_size = len(full_dataset)
    val_size = int(dataset_size * val_split)
    train_size = dataset_size - val_size
    
    train_dataset, val_dataset = torch.utils.data.random_split(
        full_dataset,
        [train_size, val_size],
        generator=torch.Generator().manual_seed(42),
    )
    
    train_loader = DataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=True,
        num_workers=num_workers,
        pin_memory=True,
    )
    
    val_loader = DataLoader(
        val_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=True,
    )
    
    return train_loader, val_loader


if __name__ == '__main__':
    # Test dataset
    import matplotlib.pyplot as plt
    
    # Create dummy images for testing
    import tempfile
    import os
    
    with tempfile.TemporaryDirectory() as tmpdir:
        # Create test images
        for i in range(3):
            img = np.random.rand(512, 512) * 255
            Image.fromarray(img.astype(np.uint8)).save(os.path.join(tmpdir, f'test_{i}.png'))
        
        # Test fast dataset
        print("Testing FastStipplingDataset...")
        dataset = FastStipplingDataset(tmpdir, num_points=1000)
        sample = dataset[0]
        print(f"Image shape: {sample['image'].shape}")
        print(f"Points shape: {sample['points'].shape}")
        
        # Test Voronoi dataset (small for speed)
        print("\nTesting VoronoiStipplingDataset...")
        dataset = VoronoiStipplingDataset(tmpdir, num_points=500, lloyd_iterations=5)
        sample = dataset[0]
        print(f"Image shape: {sample['image'].shape}")
        print(f"Points shape: {sample['points'].shape}")
        
        print("\n✓ Dataset tests passed")
