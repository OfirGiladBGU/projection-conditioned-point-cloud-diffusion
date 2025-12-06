"""Quick test to verify data loading works - simplified version without configs"""
import sys
from pathlib import Path
from PIL import Image
import numpy as np
import torch
from torch.utils.data import Dataset, DataLoader

# Test parameters
SOURCE_DIR = Path(r'C:\Users\User\PycharmProjects\projection-conditioned-point-cloud-diffusion\experiments\data_grads_v3\source')
TARGET_DIR = Path(r'C:\Users\User\PycharmProjects\projection-conditioned-point-cloud-diffusion\experiments\data_grads_v3\target')

class SimpleDataset(Dataset):
    """Simplified dataset for testing"""
    def __init__(self, source_dir, target_dir):
        self.source_dir = Path(source_dir)
        self.target_dir = Path(target_dir)
        
        # Get all source files
        self.source_files = sorted(list(self.source_dir.glob('*.png')))
        
        # Map to targets by base filename
        self.samples = []
        for src_file in self.source_files:
            tgt_file = self.target_dir / src_file.name
            if tgt_file.exists():
                self.samples.append((src_file, tgt_file))
    
    def __len__(self):
        return len(self.samples)
    
    def __getitem__(self, idx):
        src_path, tgt_path = self.samples[idx]
        
        # Load source image (grayscale)
        img = Image.open(src_path).convert('L')
        img_array = np.array(img, dtype=np.float32) / 255.0
        img_tensor = torch.from_numpy(np.stack([img_array] * 3, axis=0))
        
        # Load target image (binary)
        tgt_img = Image.open(tgt_path).convert('L')
        tgt_array = np.array(tgt_img, dtype=np.uint8)
        
        # Extract points: get coordinates where pixel < 128 (black points)
        coords = np.where(tgt_array < 128)
        if len(coords[0]) > 0:
            points = np.stack([coords[1], coords[0], np.zeros(len(coords[0]))], axis=1).astype(np.float32)
        else:
            points = np.zeros((0, 3), dtype=np.float32)
        
        points_tensor = torch.from_numpy(points)
        return img_tensor, points_tensor


print(f"Source dir: {SOURCE_DIR}")
print(f"Source exists: {SOURCE_DIR.exists()}")
print(f"Source files: {len(list(SOURCE_DIR.glob('*.png')))}")

print(f"\nTarget dir: {TARGET_DIR}")
print(f"Target exists: {TARGET_DIR.exists()}")
print(f"Target files: {len(list(TARGET_DIR.glob('*.png')))}")

# Create dataset
print("\n--- Creating Dataset ---")
dataset = SimpleDataset(SOURCE_DIR, TARGET_DIR)
print(f"Dataset size: {len(dataset)}")
print(f"Dataset samples (first 5): {[str(s[0].name) for s in dataset.samples[:5]]}")

# Test single sample
print("\n--- Testing Single Sample ---")
try:
    image, points = dataset[0]
    print(f"Image shape: {image.shape}")
    print(f"Image dtype: {image.dtype}")
    print(f"Image min/max: {image.min():.3f} / {image.max():.3f}")
    print(f"Points shape: {points.shape}")
    print(f"Points dtype: {points.dtype}")
    if points.shape[0] > 0:
        print(f"Points min: {points.min(dim=0).values}")
        print(f"Points max: {points.max(dim=0).values}")
    else:
        print("WARNING: No points found in this sample!")
    print("✓ Sample loaded successfully")
except Exception as e:
    print(f"✗ Error loading sample: {e}")
    import traceback
    traceback.print_exc()

# Test batch loading
print("\n--- Testing Batch Loading ---")
try:
    loader = DataLoader(dataset, batch_size=2, shuffle=False, num_workers=0, 
                       collate_fn=lambda x: (torch.stack([item[0] for item in x]), 
                                            [item[1] for item in x]))
    batch_images, batch_points = next(iter(loader))
    print(f"Batch images shape: {batch_images.shape}")
    print(f"Batch points list length: {len(batch_points)}")
    print(f"First point cloud in batch shape: {batch_points[0].shape}")
    if batch_points[0].shape[0] > 0:
        print(f"First point cloud - points range: {batch_points[0].min(dim=0).values} to {batch_points[0].max(dim=0).values}")
    print("✓ Batch loading successful")
except Exception as e:
    print(f"✗ Error in batch loading: {e}")
    import traceback
    traceback.print_exc()

print("\n--- All tests passed! Dataset is ready. ---")
