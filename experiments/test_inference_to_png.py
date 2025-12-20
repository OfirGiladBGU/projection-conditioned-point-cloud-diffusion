"""Generate PNG from checkpoint inference"""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent))

import torch
import numpy as np
from PIL import Image
from torch.utils.data import Dataset, DataLoader
import warnings
warnings.filterwarnings('ignore')

SOURCE_DIR = Path(r'/groups/asharf_group/ofirgila/ControlNet/training/data_grads_v3/source')
TARGET_DIR = Path(r'/groups/asharf_group/ofirgila/ControlNet/training/data_grads_v3/target')
CHECKPOINT_PATH = Path(r'/groups/asharf_group/ofirgila/projection-conditioned-point-cloud-diffusion/outputs/checkpoint_epoch_2.pth')
OUTPUT_PATH = Path(r'/groups/asharf_group/ofirgila/projection-conditioned-point-cloud-diffusion/outputs/generated_output.png')

class SimpleDataset(Dataset):
    def __init__(self, source_dir, target_dir, image_size=512):
        self.source_dir = Path(source_dir)
        self.target_dir = Path(target_dir)
        self.image_size = image_size
        
        self.source_files = sorted(list(self.source_dir.glob('*.png')))
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
        img = img.resize((self.image_size, self.image_size), Image.Resampling.BILINEAR)
        img_array = np.array(img, dtype=np.float32) / 255.0
        img_tensor = torch.from_numpy(np.stack([img_array] * 3, axis=0))
        
        return img_tensor, src_path.stem


device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(f"Device: {device}")

# Load dataset
print("\n" + "="*60)
print("LOADING DATASET")
print("="*60)
dataset = SimpleDataset(SOURCE_DIR, TARGET_DIR, image_size=512)
print(f"Dataset size: {len(dataset)}")

# Get first image
image_tensor, name = dataset[0]
print(f"First image: {name}")
print(f"Image shape: {image_tensor.shape}")

# Load model
print("\n" + "="*60)
print("LOADING MODEL FROM CHECKPOINT")
print("="*60)

from model.model import ConditionalPointCloudDiffusionModel
from pytorch3d.structures import Pointclouds
from pytorch3d.renderer.cameras import PerspectiveCameras
from pytorch3d.implicitron.dataset.data_loader_map_provider import FrameData

# Create model with same config as training
model = ConditionalPointCloudDiffusionModel(
    image_size=512,
    image_feature_model='vit_small_patch16_224_msn',
    use_local_colors=True,
    use_local_features=True,
    use_global_features=False,
    use_mask=True,
    use_distance_transform=True,
    scale_factor=1.0,
    colors_mean=0.5,
    colors_std=0.5,
    color_channels=3,
    predict_shape=True,
    predict_color=False,
    beta_start=1e-5,
    beta_end=8e-3,
    beta_schedule='linear',
    point_cloud_model='simple',
    point_cloud_model_embed_dim=64,
).to(device)

# Load checkpoint
print(f"Loading checkpoint from: {CHECKPOINT_PATH}")
checkpoint = torch.load(CHECKPOINT_PATH, map_location=device)

# Check what keys are in the checkpoint
print(f"Checkpoint keys: {list(checkpoint.keys())}")

# Try different possible keys for the model state dict
if 'model_state_dict' in checkpoint:
    model.load_state_dict(checkpoint['model_state_dict'])
elif 'state_dict' in checkpoint:
    model.load_state_dict(checkpoint['state_dict'])
elif 'model' in checkpoint:
    model.load_state_dict(checkpoint['model'])
else:
    # If checkpoint is the state dict itself
    model.load_state_dict(checkpoint)

model.eval()
print(f"Checkpoint loaded [OK] - Epoch {checkpoint.get('epoch', 'unknown') if isinstance(checkpoint, dict) else 'unknown'}")

# Run inference
print("\n" + "="*60)
print("RUNNING INFERENCE")
print("="*60)

with torch.no_grad():
    image_rgb = image_tensor.unsqueeze(0).to(device)
    
    # Create dummy camera
    R = torch.eye(3).unsqueeze(0).to(device)
    T = torch.tensor([[0.0, 0.0, 2.0]], dtype=torch.float32).to(device)
    camera = PerspectiveCameras(R=R, T=T, image_size=((512, 512),), device=device)
    
    # Create mask
    mask = torch.ones(1, 1, 512, 512).to(device)
    
    # Create dummy point cloud for batch (will be replaced during sampling)
    dummy_points = torch.zeros(1, 1000, 3).to(device)  # Batch of 1000 points
    dummy_pc = Pointclouds(points=dummy_points)
    
    # Create batch
    batch = FrameData(
        sequence_point_cloud=dummy_pc,
        camera=camera,
        image_rgb=image_rgb,
        fg_probability=mask,
    )
    
    print("Sampling point cloud from noise...")
    output, all_outputs = model(batch, mode='sample', num_inference_steps=50, return_sample_every_n_steps=10, num_points=1000)
    
    if isinstance(output, Pointclouds):
        pred_points = output.points_packed().cpu().numpy()
        print(f"Generated point cloud: {pred_points.shape[0]} points")
        print(f"  X range: [{pred_points[:, 0].min():.2f}, {pred_points[:, 0].max():.2f}]")
        print(f"  Y range: [{pred_points[:, 1].min():.2f}, {pred_points[:, 1].max():.2f}]")
        print(f"  Z range: [{pred_points[:, 2].min():.2f}, {pred_points[:, 2].max():.2f}]")
    else:
        print(f"Unexpected output type: {type(output)}")
        sys.exit(1)

# Convert points to binary PNG
print("\n" + "="*60)
print("CONVERTING TO BINARY PNG")
print("="*60)

# Create blank image
img_size = 512
output_image = np.ones((img_size, img_size), dtype=np.uint8) * 255  # White background

# Normalize points to image coordinates
# Assuming points are in range roughly [-1, 1] or similar, we need to map to [0, 512]
x_coords = pred_points[:, 0]
y_coords = pred_points[:, 1]

# Map to image coordinates (0 to 512)
x_min, x_max = x_coords.min(), x_coords.max()
y_min, y_max = y_coords.min(), y_coords.max()

# Handle edge case where all points have same coordinate
if x_max - x_min < 1e-6:
    x_normalized = np.ones_like(x_coords) * (img_size / 2)
else:
    x_normalized = ((x_coords - x_min) / (x_max - x_min) * (img_size - 1)).astype(int)

if y_max - y_min < 1e-6:
    y_normalized = np.ones_like(y_coords) * (img_size / 2)
else:
    y_normalized = ((y_coords - y_min) / (y_max - y_min) * (img_size - 1)).astype(int)

# Clip to valid range
x_normalized = np.clip(x_normalized, 0, img_size - 1)
y_normalized = np.clip(y_normalized, 0, img_size - 1)

# Set points to black
output_image[y_normalized, x_normalized] = 0

# Save image
OUTPUT_PATH.parent.mkdir(parents=True, exist_ok=True)
Image.fromarray(output_image).save(OUTPUT_PATH)

print(f"Output saved to: {OUTPUT_PATH}")
print(f"  Image size: {img_size}x{img_size}")
print(f"  Black pixels: {(output_image == 0).sum()}")
print(f"  White pixels: {(output_image == 255).sum()}")

print("\n" + "="*60)
print("[OK] INFERENCE COMPLETE")
print("="*60)
