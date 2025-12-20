"""Test actual model prediction with 512×512 images"""
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

class SimpleDataset(Dataset):
    """Simplified dataset for testing"""
    def __init__(self, source_dir, target_dir, image_size=512):
        self.source_dir = Path(source_dir)
        self.target_dir = Path(target_dir)
        self.image_size = image_size
        
        self.source_files = sorted(list(self.source_dir.glob('*.png')))[:5]  # Just 5 samples for speed
        
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
        return img_tensor, points_tensor, src_path.stem


device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(f"Device: {device}")

# Create dataset
print("\n" + "="*60)
print("LOADING DATASET")
print("="*60)
dataset = SimpleDataset(SOURCE_DIR, TARGET_DIR, image_size=512)
print(f"Dataset size: {len(dataset)}")

loader = DataLoader(
    dataset, 
    batch_size=1, 
    shuffle=False, 
    num_workers=0,
    collate_fn=lambda x: (
        torch.stack([item[0] for item in x]),
        [item[1] for item in x],
        [item[2] for item in x]
    )
)

batch_images, batch_points, names = next(iter(loader))
print(f"Batch images shape: {batch_images.shape}")
print(f"First point cloud shape: {batch_points[0].shape}")

# Now load and use the actual model
print("\n" + "="*60)
print("LOADING PC^2 MODEL")
print("="*60)

try:
    # Import model using absolute imports
    from model.model import ConditionalPointCloudDiffusionModel
    from pytorch3d.structures import Pointclouds
    from pytorch3d.renderer.cameras import PerspectiveCameras
    
    print("Model imports successful [OK]")
    
    print("\nCreating model with image_size=512...")
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
        point_cloud_model='simple',  # Use 'simple' instead of 'pvcnn' to avoid gcc dependency
        point_cloud_model_embed_dim=64,
    ).to(device)
    
    print("Model created successfully [OK]")
    
    # Count parameters
    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"Total parameters: {total_params:,}")
    print(f"Trainable parameters: {trainable_params:,}")
    
    # Test with actual batch
    print("\n" + "="*60)
    print("TESTING MODEL PREDICTION")
    print("="*60)
    
    model.eval()
    
    # Use CPU device for inference (PyTorch3D rasterizer has no GPU support on Windows)
    test_device = 'cuda'  # 'cpu'
    print(f"Using device for test: {test_device} (PyTorch3D rasterizer limitation)")
    model = model.to(test_device)
    
    with torch.no_grad():
        from pytorch3d.implicitron.dataset.data_loader_map_provider import FrameData
        
        image_rgb = batch_images.to(test_device)
        points = batch_points[0].to(test_device).unsqueeze(0)
        pc = Pointclouds(points=points)
        
        R = torch.eye(3).unsqueeze(0).to(test_device)
        T = torch.tensor([[0.0, 0.0, 2.0]], dtype=torch.float32).to(test_device)
        camera = PerspectiveCameras(R=R, T=T, image_size=((512, 512),), device=test_device)
        
        mask = torch.ones(1, 1, 512, 512).to(test_device)
        
        print(f"Input batch:")
        print(f"  image_rgb: {image_rgb.shape}")
        print(f"  points: {points.shape}")
        print(f"  mask: {mask.shape}")
        
        # Create a FrameData batch
        batch = FrameData(
            # frame_number=0,
            # sequence_name="test_sequence",
            # sequence_category="test",
            sequence_point_cloud=pc,
            camera=camera,
            image_rgb=image_rgb,
            fg_probability=mask,
        )
        
        # Try training forward pass (with ground truth points)
        print("\nRunning training forward pass (with ground truth)...")
        try:
            loss = model(batch, mode='train')
            print(f"[OK] Training loss: {loss.item():.4f}")
        except Exception as e:
            print(f"Training pass failed: {e}")
        
        # Try sampling (inference mode)
        print("\nRunning inference (sampling from noise)...")
        try:
            output, all_outputs = model(batch, mode='sample', num_inference_steps=10, return_sample_every_n_steps=5)
            
            if isinstance(output, Pointclouds):
                pred_points = output.points_packed()
                print(f"[OK] Generated point cloud shape: {pred_points.shape}")
                print(f"  Points: {pred_points.shape[0]}")
                print(f"  X range: [{pred_points[:, 0].min():.2f}, {pred_points[:, 0].max():.2f}]")
                print(f"  Y range: [{pred_points[:, 1].min():.2f}, {pred_points[:, 1].max():.2f}]")
                print(f"  Z range: [{pred_points[:, 2].min():.2f}, {pred_points[:, 2].max():.2f}]")
            else:
                print(f"Output type: {type(output)}, shape: {output.shape if hasattr(output, 'shape') else 'N/A'}")
                
        except Exception as e:
            print(f"Inference failed: {e}")
            import traceback
            traceback.print_exc()
    
    print("\n" + "="*60)
    print("[OK] MODEL TEST COMPLETE")
    print("="*60)
    
except ImportError as e:
    print(f"[ERROR] Import error: {e}")
    print("\nTrying alternative import path...")
    try:
        from model import ConditionalPointCloudDiffusionModel
        from pytorch3d.structures import Pointclouds
        from pytorch3d.renderer.cameras import PerspectiveCameras
        
        print("Model imports successful [OK]")
        
        print("\nCreating model with image_size=512...")
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
            point_cloud_model='pvcnn',
            point_cloud_model_embed_dim=64,
        ).to(device)
        
        print("Model created successfully [OK]")
        
        total_params = sum(p.numel() for p in model.parameters())
        trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
        print(f"Total parameters: {total_params:,}")
        print(f"Trainable parameters: {trainable_params:,}")
        
        model.eval()
        
        print("\n" + "="*60)
        print("TESTING MODEL WITH 512x512 INPUT")
        print("="*60)
        
        with torch.no_grad():
            image_rgb = batch_images.to(device)
            points = batch_points[0].to(device).unsqueeze(0)
            pc = Pointclouds(points=points)
            
            R = torch.eye(3).unsqueeze(0).to(device)
            T = torch.tensor([[[0.0, 0.0, 2.0]]]).to(device)
            camera = PerspectiveCameras(R=R, T=T, image_size=((512, 512),), device=device)
            
            mask = torch.ones(1, 1, 512, 512).to(device)
            
            print(f"Input batch:")
            print(f"  image_rgb: {image_rgb.shape}")
            print(f"  points: {points.shape}")
            print(f"  mask: {mask.shape}")
            
            print("\nRunning forward pass...")
            try:
                loss = model(
                    pc=pc,
                    camera=camera,
                    image_rgb=image_rgb,
                    mask=mask,
                )
                print(f"[OK] Loss computed: {loss.item():.4f}")
            except Exception as e:
                print(f"Error: {e}")
        
        print("\n[OK] Model handles 512x512 images correctly!")
        
    except Exception as e2:
        print(f"[ERROR] Alternative import also failed: {e2}")
        import traceback
        traceback.print_exc()

except Exception as e:
    print(f"[ERROR] Error: {e}")
    import traceback
    traceback.print_exc()
