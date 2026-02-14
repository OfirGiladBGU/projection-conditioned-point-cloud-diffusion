"""Dataset for Point-RT training.

Extends V5 dataset with Lloyd's ground truth support.
"""

import torch
from torch.utils.data import Dataset, DataLoader
from pathlib import Path
from PIL import Image
import numpy as np
from typing import Tuple, Optional


class PointRTDataset(Dataset):
    """Dataset for Point-RT training with Lloyd's ground truth support.
    
    Loads:
    - Source images (conditioning context)
    - Fast initialization points (from density-based sampling)
    - Lloyd's ground truth (offline preprocessed)
    """
    
    def __init__(
        self,
        source_dir: str,
        image_size: int = 512,
        num_points: int = 2048,
        target_dir: Optional[str] = None,
        lloyd_dir: Optional[str] = None,
    ):
        """Initialize dataset.
        
        Args:
            source_dir: Directory with source grayscale images
            image_size: Resize images to this size
            num_points: Number of points to use
            target_dir: Optional directory with density targets (for reference)
            lloyd_dir: Optional directory with Lloyd's ground truth point clouds
        """
        self.source_dir = Path(source_dir)
        self.target_dir = Path(target_dir) if target_dir else None
        self.lloyd_dir = Path(lloyd_dir) if lloyd_dir else None
        self.image_size = image_size
        self.num_points = num_points
        
        # Find all source images
        self.image_paths = self._list_images(self.source_dir)
        if not self.image_paths:
            raise ValueError(f"No images found in {source_dir}")
        
        print(f"PointRTDataset: {len(self.image_paths)} images")
        if self.lloyd_dir:
            print(f"  └─ With Lloyd's ground truth from {self.lloyd_dir}")
    
    def _list_images(self, directory: Path) -> list:
        """Find all image files in directory."""
        extensions = {'.png', '.jpg', '.jpeg', '.bmp', '.tiff'}
        paths = [p for p in directory.iterdir() 
                if p.suffix.lower() in extensions]
        return sorted(paths)
    
    def __len__(self) -> int:
        return len(self.image_paths)
    
    def __getitem__(self, idx: int) -> dict:
        """Load image and ground truth.
        
        Returns:
            dict with keys:
            - 'image': (1, H, W) normalized to [0, 1]
            - 'lloyd_points': (N, 2) Lloyd's ground truth, or None
            - 'image_path': str
        """
        source_path = self.image_paths[idx]
        
        # Load image
        img = Image.open(source_path).convert('L').resize(
            (self.image_size, self.image_size), Image.BILINEAR
        )
        img_np = np.asarray(img, dtype=np.float32) / 255.0
        img_tensor = torch.from_numpy(img_np).unsqueeze(0)  # (1, H, W)
        
        # Load Lloyd's ground truth if available
        lloyd_points = None
        if self.lloyd_dir:
            lloyd_path = self.lloyd_dir / source_path.stem / "points.pt"
            if lloyd_path.exists():
                lloyd_points = torch.load(lloyd_path)  # (N, 2)
            else:
                # Fallback: try alternative naming
                lloyd_path_alt = self.lloyd_dir / (source_path.stem + ".pt")
                if lloyd_path_alt.exists():
                    lloyd_points = torch.load(lloyd_path_alt)
        
        return {
            'image': img_tensor,
            'lloyd_points': lloyd_points,
            'image_path': str(source_path),
        }


def create_dataloaders(
    source_dir: str,
    batch_size: int = 16,
    num_workers: int = 4,
    image_size: int = 512,
    num_points: int = 2048,
    target_dir: Optional[str] = None,
    lloyd_dir: Optional[str] = None,
    val_split: float = 0.1,
) -> Tuple[DataLoader, DataLoader]:
    """Create train/val dataloaders for Point-RT.
    
    Args:
        source_dir: Directory with source images
        batch_size: Batch size for training
        num_workers: Number of data loading workers
        image_size: Image resolution
        num_points: Number of points
        target_dir: Optional target directory (unused currently)
        lloyd_dir: Directory with Lloyd's ground truth
        val_split: Fraction of data for validation
    
    Returns:
        (train_loader, val_loader)
    """
    dataset = PointRTDataset(
        source_dir=source_dir,
        image_size=image_size,
        num_points=num_points,
        target_dir=target_dir,
        lloyd_dir=lloyd_dir,
    )
    
    # Split into train/val
    n_val = int(len(dataset) * val_split)
    n_train = len(dataset) - n_val
    
    from torch.utils.data import random_split
    train_dataset, val_dataset = random_split(dataset, [n_train, n_val])
    
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


if __name__ == "__main__":
    print("Dataset test")
    # This would require actual data directory
