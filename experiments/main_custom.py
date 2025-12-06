"""
Custom main.py for training and inference on custom datasets using the PC^2 architecture.
Uses the actual ConditionalPointCloudDiffusionModel from the project.

Expects:
  - source folder: RGB images (jpg, png)
  - target folder: 3D point clouds (ply, pth, npy)
"""

import datetime
import math
import os
import sys
from pathlib import Path

# Add experiments directory to path for relative imports
sys.path.insert(0, str(Path(__file__).parent))

import time
from contextlib import nullcontext
from typing import Any, Dict, Iterable, List, Optional, Tuple

import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
import numpy as np
from PIL import Image
from tqdm import tqdm

try:
    import wandb
    WANDB_AVAILABLE = True
except ImportError:
    WANDB_AVAILABLE = False

try:
    from pytorch3d.structures import Pointclouds
    from pytorch3d.io import IO
    from pytorch3d.renderer.cameras import PerspectiveCameras
    PYTORCH3D_AVAILABLE = True
except ImportError:
    PYTORCH3D_AVAILABLE = False

# Import project modules
try:
    from model import ConditionalPointCloudDiffusionModel
    from model.model_utils import get_num_points
    ARCHITECTURE_AVAILABLE = True
except ImportError:
    ARCHITECTURE_AVAILABLE = False
    print("Warning: Could not import project architecture. Using basic model.")


# ============================================================================
# CUSTOM DATASET
# ============================================================================

class CustomPointCloudDataset(Dataset):
    """
    Custom dataset that loads images and point clouds from separate folders.
    Formats data for compatibility with ConditionalPointCloudDiffusionModel.
    
    Args:
        source_dir: Directory containing RGB images
        target_dir: Directory containing 3D point clouds
        image_extensions: Tuple of valid image extensions
        pointcloud_extensions: Tuple of valid point cloud extensions
        image_size: Size to resize images to
    """
    
    def __init__(
        self,
        source_dir: str,
        target_dir: str,
        image_extensions: Tuple[str, ...] = ('.jpg', '.jpeg', '.png'),
        pointcloud_extensions: Tuple[str, ...] = ('.jpg', '.jpeg', '.png', '.ply', '.pth', '.npy'),
        image_size: int = 256,
    ):
        self.source_dir = Path(source_dir)
        self.target_dir = Path(target_dir)
        self.image_extensions = image_extensions
        self.pointcloud_extensions = pointcloud_extensions
        self.image_size = image_size
        
        # Get list of image files
        self.image_files = []
        for ext in image_extensions:
            self.image_files.extend(self.source_dir.glob(f'*{ext}'))
            self.image_files.extend(self.source_dir.glob(f'*{ext.upper()}'))
        
        self.image_files = sorted(list(set(self.image_files)))
        
        if len(self.image_files) == 0:
            raise ValueError(f"No images found in {source_dir}")
        
        print(f"Found {len(self.image_files)} images in {source_dir}")
    
    def __len__(self):
        return len(self.image_files)
    
    def _load_image(self, image_path: Path) -> torch.Tensor:
        """Load and preprocess image (grayscale)."""
        img = Image.open(image_path).convert('L')  # Load as grayscale
        # Resize to target size
        img = img.resize((self.image_size, self.image_size), Image.Resampling.BILINEAR)
        # Convert to tensor [0, 1]
        img_tensor = torch.from_numpy(np.array(img)).float() / 255.0
        # Expand to 3 channels for compatibility with model
        img_tensor = img_tensor.unsqueeze(0).repeat(3, 1, 1)  # [3, H, W]
        return img_tensor
    
    def _load_pointcloud(self, pointcloud_path: Path) -> torch.Tensor:
        """Load point cloud from file or convert binary image to 3D points."""
        if pointcloud_path.suffix in ['.jpg', '.jpeg', '.png', '.bmp']:
            # Load binary image and convert to 3D points
            img = Image.open(pointcloud_path).convert('L')  # Grayscale
            img_array = np.array(img)
            
            # Threshold: extract black points (value < 128)
            y_coords, x_coords = np.where(img_array < 128)
            
            # Create 3D points: [x, y, 0]
            points = np.stack([x_coords, y_coords, np.zeros_like(x_coords)], axis=1)
            points = torch.from_numpy(points).float()
            
            return points
        
        elif pointcloud_path.suffix == '.ply':
            # Using pytorch3d IO
            if PYTORCH3D_AVAILABLE:
                io = IO()
                pointcloud = io.load_pointcloud(str(pointcloud_path))
                points = pointcloud.points_packed()  # [N, 3]
                return points
            else:
                raise ImportError("PyTorch3D required for .ply files")
        
        elif pointcloud_path.suffix == '.pth':
            data = torch.load(pointcloud_path)
            if isinstance(data, dict):
                # Assume 'points' or 'vertices' key
                if 'points' in data:
                    return data['points']
                elif 'vertices' in data:
                    return data['vertices']
                else:
                    return list(data.values())[0]
            else:
                return data
        
        elif pointcloud_path.suffix == '.npy':
            points = np.load(pointcloud_path)
            return torch.from_numpy(points).float()
        
        else:
            raise ValueError(f"Unknown point cloud format: {pointcloud_path.suffix}")
    
    def __getitem__(self, idx: int) -> Dict[str, Any]:
        """Load image and corresponding point cloud."""
        image_path = self.image_files[idx]
        
        # Find corresponding point cloud
        base_name = image_path.stem
        pointcloud_path = None
        
        for ext in self.pointcloud_extensions:
            candidate = self.target_dir / f'{base_name}{ext}'
            if candidate.exists():
                pointcloud_path = candidate
                break
        
        if pointcloud_path is None:
            raise FileNotFoundError(
                f"No point cloud found for {base_name} in {self.target_dir}"
            )
        
        # Load data
        image = self._load_image(image_path)  # [3, H, W] in [0, 1]
        points = self._load_pointcloud(pointcloud_path)  # [N, 3]
        
        # Ensure points are [N, 3]
        if points.dim() == 1:
            points = points.unsqueeze(0)
        
        # Create dummy camera (looking at origin)
        # For a simple orthographic camera
        if PYTORCH3D_AVAILABLE:
            # Create a perspective camera at distance 2 from origin
            R = torch.eye(3).unsqueeze(0)
            T = torch.tensor([[[0.0, 0.0, 2.0]]])
            camera = PerspectiveCameras(R=R, T=T, image_size=((self.image_size, self.image_size),))
        else:
            camera = None
        
        # Create mask (all 1s, indicating all pixels are valid)
        mask = torch.ones(1, self.image_size, self.image_size)
        
        return {
            'image_rgb': image.unsqueeze(0),  # [1, 3, H, W] for batch compatibility
            'sequence_point_cloud': Pointclouds(points=points.unsqueeze(0)) if PYTORCH3D_AVAILABLE else points.unsqueeze(0),
            'camera': camera,
            'mask': mask.unsqueeze(0),  # [1, 1, H, W]
            'image_path': str(image_path),
            'points_path': str(pointcloud_path),
            'name': base_name,
        }


# ============================================================================
# UTILITIES
# ============================================================================

class AverageMeter:
    """Simple average meter for metrics."""
    
    def __init__(self):
        self.reset()
    
    def reset(self):
        self.val = 0
        self.avg = 0
        self.sum = 0
        self.count = 0
    
    def update(self, val, n=1):
        self.val = val
        self.sum += val * n
        self.count += n
        self.avg = self.sum / self.count


# ============================================================================
# MODEL SETUP
# ============================================================================

def create_model(device: torch.device):
    """Create the diffusion model using project architecture."""
    
    if not ARCHITECTURE_AVAILABLE:
        raise RuntimeError(
            "Cannot import ConditionalPointCloudDiffusionModel. "
            "Make sure you're running this from the project directory."
        )
    
    # Model configuration (matching project defaults)
    model = ConditionalPointCloudDiffusionModel(
        # Feature extraction
        image_size=512,
        image_feature_model='vit_small_patch16_224_msn',
        use_local_colors=True,
        use_local_features=True,
        use_global_features=False,
        use_mask=True,
        use_distance_transform=True,
        
        # Point cloud
        scale_factor=1.0,
        colors_mean=0.5,
        colors_std=0.5,
        color_channels=3,
        predict_shape=True,
        predict_color=False,
        
        # Diffusion
        beta_start=1e-5,
        beta_end=8e-3,
        beta_schedule='linear',
        
        # Point cloud model
        point_cloud_model='pvcnn',
        point_cloud_model_embed_dim=64,
    ).to(device)
    
    return model


# ============================================================================
# TRAINING FUNCTIONS
# ============================================================================

def train_step(
    model: nn.Module,
    batch: Dict[str, torch.Tensor],
    optimizer: torch.optim.Optimizer,
    device: torch.device,
) -> float:
    """
    Single training step using the PC^2 diffusion model.
    
    Args:
        model: ConditionalPointCloudDiffusionModel
        batch: Batch of data
        optimizer: Optimizer
        device: Device to run on
    
    Returns:
        loss value
    """
    image_rgb = batch['image_rgb'].to(device)  # [B, 3, H, W]
    pc = batch['sequence_point_cloud']  # Pointclouds object or tensor
    camera = batch['camera']
    mask = batch['mask'].to(device)  # [B, 1, H, W]
    
    # Convert tensor to Pointclouds if needed
    if isinstance(pc, torch.Tensor):
        pc = Pointclouds(points=pc.squeeze(0))
    
    # Forward pass through diffusion model
    # The model handles: noise injection, conditioning, and loss computation
    loss = model(
        pc=pc,
        camera=camera,
        image_rgb=image_rgb,
        mask=mask,
    )
    
    # Backward pass
    optimizer.zero_grad()
    loss.backward()
    torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
    optimizer.step()
    
    return loss.item()


def validate(
    model: nn.Module,
    dataloader: DataLoader,
    device: torch.device,
) -> float:
    """
    Validation pass - compute loss on validation set.
    
    Args:
        model: The diffusion model
        dataloader: Validation dataloader
        device: Device to run on
    
    Returns:
        average loss
    """
    model.eval()
    losses = AverageMeter()
    
    with torch.no_grad():
        for batch in tqdm(dataloader, desc='Validating'):
            image_rgb = batch['image_rgb'].to(device)
            pc = batch['sequence_point_cloud']
            camera = batch['camera']
            mask = batch['mask'].to(device)
            
            # Convert tensor to Pointclouds if needed
            if isinstance(pc, torch.Tensor):
                pc = Pointclouds(points=pc.squeeze(0))
            
            loss = model(
                pc=pc,
                camera=camera,
                image_rgb=image_rgb,
                mask=mask,
            )
            
            losses.update(loss.item())
    
    model.train()
    return losses.avg


def predict(
    model: nn.Module,
    dataloader: DataLoader,
    device: torch.device,
    output_dir: str = 'predictions',
    num_inference_steps: int = 50,
) -> None:
    """
    Run inference (sampling) using the diffusion model.
    
    Args:
        model: The diffusion model
        dataloader: Test dataloader
        device: Device to run on
        output_dir: Where to save predictions
        num_inference_steps: Number of diffusion steps for inference
    """
    model.eval()
    output_dir = Path(output_dir)
    output_dir.mkdir(exist_ok=True, parents=True)
    
    with torch.no_grad():
        for batch_idx, batch in enumerate(tqdm(dataloader, desc='Predicting')):
            image_rgb = batch['image_rgb'].to(device)
            camera = batch['camera']
            mask = batch['mask'].to(device)
            names = batch['name']
            
            # Sample from diffusion model
            # Returns Pointclouds objects
            try:
                output, all_outputs = model(
                    batch=None,  # Will create from components
                    pc=None,  # Start from noise
                    camera=camera,
                    image_rgb=image_rgb,
                    mask=mask,
                    mode='sample',
                    num_inference_steps=num_inference_steps,
                    return_sample_every_n_steps=10,
                )
            except Exception as e:
                print(f"Sampling failed: {e}")
                print("Falling back to unconditional sampling...")
                output = None
            
            # Save predictions
            if output is not None:
                for i, name in enumerate(names):
                    if isinstance(output, Pointclouds):
                        points = output[i].points_packed().cpu().numpy()
                    else:
                        points = output[i].cpu().numpy()
                    
                    pred_file = output_dir / f'{name}_pred.npy'
                    np.save(pred_file, points)
                    print(f'Saved: {pred_file}')


# ============================================================================
# MAIN TRAINING SCRIPT
# ============================================================================

def main(
    source_dir: str,
    target_dir: str,
    output_dir: str = 'outputs',
    batch_size: int = 4,
    num_epochs: int = 10,
    learning_rate: float = 1e-3,
    num_workers: int = 4,
    device: str = 'cuda' if torch.cuda.is_available() else 'cpu',
    mode: str = 'train',  # 'train' or 'predict'
    checkpoint: Optional[str] = None,
    use_wandb: bool = False,
):
    """
    Main training/inference loop.
    
    Args:
        source_dir: Path to images folder
        target_dir: Path to point clouds folder
        output_dir: Where to save outputs
        batch_size: Batch size for training
        num_epochs: Number of training epochs
        learning_rate: Learning rate
        num_workers: Number of data loading workers
        device: Device to use ('cuda' or 'cpu')
        mode: 'train' or 'predict'
        checkpoint: Path to checkpoint to resume from
        use_wandb: Whether to use Weights & Biases for logging
    """
    
    device = torch.device(device)
    output_dir = Path(output_dir)
    output_dir.mkdir(exist_ok=True, parents=True)
    
    print(f"Device: {device}")
    print(f"Source dir: {source_dir}")
    print(f"Target dir: {target_dir}")
    print(f"Mode: {mode}")
    
    # ========================================================================
    # DATASET & DATALOADER
    # ========================================================================
    print("\nLoading dataset...")
    dataset = CustomPointCloudDataset(
        source_dir=source_dir,
        target_dir=target_dir,
        image_size=512,
    )
    
    # Split into train/val
    train_size = int(0.8 * len(dataset))
    val_size = len(dataset) - train_size
    train_dataset, val_dataset = torch.utils.data.random_split(
        dataset, [train_size, val_size]
    )
    
    dataloader_train = DataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=True,
        num_workers=num_workers,
        pin_memory=True if device.type == 'cuda' else False,
    )
    
    dataloader_val = DataLoader(
        val_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=True if device.type == 'cuda' else False,
    )
    
    print(f"Train set: {train_size} samples")
    print(f"Val set: {val_size} samples")
    
    # ========================================================================
    # MODEL
    # ========================================================================
    print("\nInitializing model (PC^2 Architecture)...")
    model = create_model(device)
    
    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"Total parameters: {total_params:,}")
    print(f"Trainable parameters: {trainable_params:,}")
    
    # ========================================================================
    # OPTIMIZER & SCHEDULER
    # ========================================================================
    optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
        optimizer, T_max=num_epochs * len(dataloader_train)
    )
    
    # ========================================================================
    # WEIGHTS & BIASES
    # ========================================================================
    if use_wandb and WANDB_AVAILABLE:
        wandb.init(
            project='point-cloud-diffusion-custom',
            name=f'{mode}_{datetime.datetime.now().strftime("%Y%m%d_%H%M%S")}',
            config={
                'batch_size': batch_size,
                'num_epochs': num_epochs,
                'learning_rate': learning_rate,
                'device': str(device),
            },
        )
    
    # ========================================================================
    # PREDICT MODE
    # ========================================================================
    if mode == 'predict':
        print("\n" + "="*60)
        print("PREDICTION MODE")
        print("="*60)
        
        if checkpoint is not None:
            print(f"Loading checkpoint: {checkpoint}")
            model.load_state_dict(torch.load(checkpoint, map_location=device))
        
        predict(
            model=model,
            dataloader=dataloader_val,
            device=device,
            output_dir=output_dir / 'predictions',
        )
        print(f"\nPredictions saved to: {output_dir / 'predictions'}")
        return
    
    # ========================================================================
    # TRAINING MODE
    # ========================================================================
    print("\n" + "="*60)
    print("TRAINING MODE")
    print("="*60)
    print(f"Epochs: {num_epochs}")
    print(f"Learning rate: {learning_rate}")
    print(f"Batch size: {batch_size}")
    
    # Load checkpoint if provided
    start_epoch = 0
    if checkpoint is not None:
        print(f"Loading checkpoint: {checkpoint}")
        checkpoint_data = torch.load(checkpoint, map_location=device)
        model.load_state_dict(checkpoint_data['model'])
        optimizer.load_state_dict(checkpoint_data['optimizer'])
        scheduler.load_state_dict(checkpoint_data['scheduler'])
        start_epoch = checkpoint_data.get('epoch', 0)
    
    model.train()
    best_val_loss = float('inf')
    
    # Training loop
    for epoch in range(start_epoch, num_epochs):
        print(f"\n--- Epoch {epoch + 1}/{num_epochs} ---")
        
        # Train
        train_loss = AverageMeter()
        pbar = tqdm(dataloader_train, desc='Training')
        
        for batch_idx, batch in enumerate(pbar):
            loss = train_step(
                model=model,
                batch=batch,
                optimizer=optimizer,
                device=device,
            )
            
            train_loss.update(loss)
            scheduler.step()
            
            pbar.set_postfix({'loss': f'{train_loss.avg:.4f}'})
            
            # Log to wandb
            if use_wandb and WANDB_AVAILABLE and batch_idx % 10 == 0:
                wandb.log({
                    'train_loss': loss,
                    'learning_rate': optimizer.param_groups[0]['lr'],
                })
        
        print(f"Train loss: {train_loss.avg:.4f}")
        
        # Validate
        val_loss = validate(model, dataloader_val, device)
        print(f"Val loss: {val_loss:.4f}")
        
        if use_wandb and WANDB_AVAILABLE:
            wandb.log({'val_loss': val_loss, 'epoch': epoch + 1})
        
        # Save checkpoint
        checkpoint_path = output_dir / f'checkpoint_epoch_{epoch + 1}.pth'
        torch.save({
            'model': model.state_dict(),
            'optimizer': optimizer.state_dict(),
            'scheduler': scheduler.state_dict(),
            'epoch': epoch + 1,
            'train_loss': train_loss.avg,
            'val_loss': val_loss,
        }, checkpoint_path)
        
        # Save best checkpoint
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            best_checkpoint_path = output_dir / 'checkpoint_best.pth'
            torch.save({
                'model': model.state_dict(),
                'optimizer': optimizer.state_dict(),
                'scheduler': scheduler.state_dict(),
                'epoch': epoch + 1,
                'train_loss': train_loss.avg,
                'val_loss': val_loss,
            }, best_checkpoint_path)
            print(f"Saved best checkpoint: {best_checkpoint_path}")
    
    print("\n" + "="*60)
    print("Training completed!")
    print("="*60)
    
    if use_wandb and WANDB_AVAILABLE:
        wandb.finish()


if __name__ == '__main__':
    # ========================================================================
    # HARDCODED PARAMETERS - MODIFY HERE
    # ========================================================================
    
    # Data paths
    SOURCE_DIR = r'C:\Users\User\PycharmProjects\projection-conditioned-point-cloud-diffusion\experiments\data_grads_v3\source'
    TARGET_DIR = r'C:\Users\User\PycharmProjects\projection-conditioned-point-cloud-diffusion\experiments\data_grads_v3\target'
    
    # ========================================================================
    # HARDCODED PARAMETERS - CUSTOMIZE HERE
    # ========================================================================
    
    # Data parameters (for 512×512 images with ~5000 points per cloud)
    IMAGE_SIZE = 512
    NUM_POINTS = 5000  # Approximate points per cloud
    
    # Training parameters
    BATCH_SIZE = 2
    NUM_EPOCHS = 10
    LEARNING_RATE = 1e-3
    NUM_WORKERS = 0  # Set to 0 for debugging, increase for faster data loading
    
    # Model parameters
    DEVICE = 'cuda' if torch.cuda.is_available() else 'cpu'
    
    # Mode: 'train' or 'predict'
    MODE = 'train'
    
    # Output and checkpoint
    OUTPUT_DIR = 'outputs'
    CHECKPOINT = None  # Set to path like 'outputs/checkpoint_best.pth' to resume
    
    # Logging
    USE_WANDB = False
    
    # ========================================================================
    # Run main function with hardcoded parameters
    # ========================================================================
    
    main(
        source_dir=SOURCE_DIR,
        target_dir=TARGET_DIR,
        output_dir=OUTPUT_DIR,
        batch_size=BATCH_SIZE,
        num_epochs=NUM_EPOCHS,
        learning_rate=LEARNING_RATE,
        num_workers=NUM_WORKERS,
        device=DEVICE,
        mode=MODE,
        checkpoint=CHECKPOINT,
        use_wandb=USE_WANDB,
    )
