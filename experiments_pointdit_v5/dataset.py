"""Simple dataset for loading images and point clouds from paired source/target images.

For synthetic stippling datasets, see tools/synthetic_dataset/
"""

import torch
from torch.utils.data import Dataset, DataLoader
from pathlib import Path
from PIL import Image
import numpy as np
from typing import Tuple, Optional


class SimpleImageDataset(Dataset):
    """Load paired source images and target stippling masks.
    
    This dataset expects:
    - source_dir: Directory with grayscale input images
    - target_dir: Directory with binary stippling masks (black pixels = stipple points)
    
    If target_dir is not provided, generates random uniform points.
    """
    
    def __init__(
        self,
        source_dir: str,
        image_size: int = 512,
        num_points: int = 2048,
        target_dir: Optional[str] = None,
    ):
        """
        Args:
            source_dir: Directory with source images
            image_size: Resize images to this size
            num_points: Number of points to sample from target or generate
            target_dir: Optional directory with target stippling masks
        """
        self.source_dir = Path(source_dir)
        self.target_dir = Path(target_dir) if target_dir else None
        self.image_size = image_size
        self.num_points = num_points
        
        # Find all images
        self.image_paths = self._list_images(self.source_dir)
        if not self.image_paths:
            raise ValueError(f"No images found in {source_dir}")
        
        # Check if target directory exists and has matching files
        if self.target_dir:
            if not self.target_dir.exists():
                raise ValueError(f"Target directory not found: {target_dir}")
            print(f"SimpleImageDataset: {len(self.image_paths)} paired images from {source_dir}")
        else:
            print(f"SimpleImageDataset: {len(self.image_paths)} images from {source_dir} (random points)")
    
    def __len__(self) -> int:
        return len(self.image_paths)
    
    def __getitem__(self, idx: int) -> dict:
        """Load source image and target stippling points."""
        source_path = self.image_paths[idx]
        
        # Load source image (grayscale)
        img = Image.open(source_path).convert('L').resize(
            (self.image_size, self.image_size), Image.BILINEAR
        )
        img_np = np.asarray(img, dtype=np.float32) / 255.0
        
        # Load or generate points
        if self.target_dir:
            # Load target stippling mask
            target_path = self.target_dir / source_path.name
            if not target_path.exists():
                raise FileNotFoundError(f"Target image not found: {target_path}")
            
            target = Image.open(target_path).convert('1').resize(
                (self.image_size, self.image_size), Image.NEAREST
            )
            target_np = np.asarray(target, dtype=bool)
            
            # Extract stipple points (black pixels = False = stipple locations)
            ys, xs = np.where(~target_np)  # Get coordinates of False (black) pixels
            
            if len(xs) == 0:
                # No stipple points, generate random
                points = np.random.uniform(-1.0, 1.0, size=(self.num_points, 2)).astype(np.float32)
            elif len(xs) < self.num_points:
                # Not enough points, sample with replacement
                indices = np.random.choice(len(xs), size=self.num_points, replace=True)
                points_pixel = np.stack([xs[indices], ys[indices]], axis=1).astype(np.float32)
                # Normalize to [-1, 1]
                points = (points_pixel / (self.image_size - 1)) * 2.0 - 1.0
            else:
                # Enough points, sample without replacement
                indices = np.random.choice(len(xs), size=self.num_points, replace=False)
                points_pixel = np.stack([xs[indices], ys[indices]], axis=1).astype(np.float32)
                # Normalize to [-1, 1]
                points = (points_pixel / (self.image_size - 1)) * 2.0 - 1.0
        else:
            # Generate random points uniformly in [-1, 1]
            points = np.random.uniform(-1.0, 1.0, size=(self.num_points, 2)).astype(np.float32)
        
        return {
            'image': torch.from_numpy(img_np).float().unsqueeze(0),  # (1, H, W)
            'points': torch.from_numpy(points).float(),  # (N, 2)
        }
    
    @staticmethod
    def _list_images(directory: Path):
        """List all image files in directory."""
        exts = {'.png', '.jpg', '.jpeg', '.bmp', '.tif', '.tiff'}
        return sorted([p for p in directory.iterdir() if p.suffix.lower() in exts])


def create_dataloaders(
    source_dir: str,
    batch_size: int = 4,
    num_workers: int = 4,
    val_split: float = 0.1,
    image_size: int = 512,
    num_points: int = 2048,
    target_dir: Optional[str] = None,
) -> Tuple[DataLoader, DataLoader]:
    """Create train and validation dataloaders.
    
    Args:
        source_dir: Directory with images
        batch_size: Batch size
        num_workers: Number of data loading workers
        val_split: Fraction of data for validation
        image_size: Image size (images will be resized to this)
        num_points: Number of points in each point cloud
        target_dir: Optional directory with target stippling masks
        
    Returns:
        train_loader, val_loader
    """
    # Create dataset
    full_dataset = SimpleImageDataset(
        source_dir=source_dir,
        image_size=image_size,
        num_points=num_points,
        target_dir=target_dir,
    )
    
    # Split train/val
    dataset_size = len(full_dataset)
    val_size = int(dataset_size * val_split)
    train_size = dataset_size - val_size
    
    train_dataset, val_dataset = torch.utils.data.random_split(
        full_dataset,
        [train_size, val_size],
        generator=torch.Generator().manual_seed(42),
    )
    
    print(f"Train samples: {train_size}, Val samples: {val_size}")
    
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
    import tempfile
    import os
    
    with tempfile.TemporaryDirectory() as tmpdir:
        # Create test images
        for i in range(3):
            img = np.random.rand(512, 512) * 255
            Image.fromarray(img.astype(np.uint8)).save(os.path.join(tmpdir, f'test_{i}.png'))
        
        # Test dataset
        print("Testing SimpleImageDataset...")
        dataset = SimpleImageDataset(tmpdir, num_points=1000)
        sample = dataset[0]
        print(f"Image shape: {sample['image'].shape}")
        print(f"Points shape: {sample['points'].shape}")
        print(f"Points range: [{sample['points'].min():.2f}, {sample['points'].max():.2f}]")
        
        print("\n✓ Dataset test passed")
