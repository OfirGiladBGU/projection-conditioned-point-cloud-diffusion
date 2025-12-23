"""Generate PNG from checkpoint inference"""
import sys
import random
import shutil
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent))

import torch
import numpy as np
from PIL import Image
from torch.utils.data import Dataset, DataLoader
import warnings
warnings.filterwarnings('ignore')


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
        
        # Keep as grayscale if using CNN, expand to 3 channels for ViT
        if hasattr(self, 'use_cnn_extractor') and self.use_cnn_extractor:
            img_tensor = torch.from_numpy(img_array[np.newaxis, :, :])  # [1, H, W]
        else:
            img_tensor = torch.from_numpy(np.stack([img_array] * 3, axis=0))  # [3, H, W]
        
        return img_tensor, src_path.stem


def main(use_cnn_extractor=False):
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Device: {device}")
    print(f"Using CNN feature extractor: {use_cnn_extractor}")

    # Load dataset
    print("\n" + "="*60)
    print("LOADING DATASET")
    print("="*60)
    dataset = SimpleDataset(SOURCE_DIR, TARGET_DIR, image_size=512)
    dataset.use_cnn_extractor = use_cnn_extractor  # Pass flag to dataset
    print(f"Dataset size: {len(dataset)}")
    num_images = min(NUM_IMAGES_TO_PROCESS, len(dataset))
    indices = random.sample(range(len(dataset)), k=num_images)
    print(f"Sampling {num_images} image(s): {indices}")

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
        use_grayscale_normalization=True,  # Use simple 0.5/0.5 normalization for grayscale
        use_cnn_extractor=use_cnn_extractor,  # NEW: flag for CNN feature extractor
        image_color_channels=1 if use_cnn_extractor else 3,  # NEW: 1 for grayscale CNN, 3 for RGB ViT
        scale_factor=1.0,
        colors_mean=0.5,
        colors_std=0.5,
        color_channels=3,
        predict_shape=True,
        predict_color=False,
        beta_start=1e-5,
        beta_end=8e-3,
        beta_schedule='linear',
        loss_xy_only=True,
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
    if CLEAN_PREDICT_DIR and PREDICT_DIR.exists():
        print(f"Cleaning predict dir: {PREDICT_DIR}")
        shutil.rmtree(PREDICT_DIR)
    PREDICT_DIR.mkdir(parents=True, exist_ok=True)

    with torch.no_grad():
        for start in range(0, len(indices), BATCH_SIZE):
            batch_indices = indices[start:start + BATCH_SIZE]
            batch_images = []
            batch_names = []
            for idx in batch_indices:
                image_tensor, name = dataset[idx]
                batch_images.append(image_tensor)
                batch_names.append((name, idx))

            print(f"\nProcessing batch indices {batch_indices}")
            image_rgb = torch.stack(batch_images, dim=0).to(device)
            
            # DIAGNOSTIC: Log input statistics per image
            print("\n[DIAGNOSTIC] Input image statistics:")
            for i, (name, _) in enumerate(batch_names):
                img_mean = image_rgb[i].mean().item()
                img_std = image_rgb[i].std().item()
                img_min = image_rgb[i].min().item()
                img_max = image_rgb[i].max().item()
                print(f"  '{name}': mean={img_mean:.4f}, std={img_std:.4f}, min={img_min:.4f}, max={img_max:.4f}")

            # Create dummy camera per sample
            R = torch.eye(3).unsqueeze(0).repeat(len(batch_indices), 1, 1).to(device)
            T = torch.tensor([[0.0, 0.0, 2.0]], dtype=torch.float32).repeat(len(batch_indices), 1).to(device)
            camera = PerspectiveCameras(R=R, T=T, image_size=((512, 512),), device=device)

            # Create mask
            mask = torch.ones(len(batch_indices), 1, 512, 512).to(device)

            # Dummy point clouds (one per sample)
            dummy_points = [torch.zeros(NUM_POINTS, 3, device=device) for _ in batch_indices]
            dummy_pc = Pointclouds(points=dummy_points)

            batch = FrameData(
                sequence_point_cloud=dummy_pc,
                camera=camera,
                image_rgb=image_rgb,
                fg_probability=mask,
            )
            
            # DIAGNOSTIC: Check conditioning inputs
            print("\n[DIAGNOSTIC] Conditioning check:")
            print(f"  image_rgb shape: {image_rgb.shape}, device: {image_rgb.device}")
            print(f"  mask shape: {mask.shape}, unique values: {mask.unique().tolist()}")
            print(f"  camera R shape: {camera.R.shape if camera else 'None'}")
            print(f"  Model in eval mode: {not model.training}")

            print("\nSampling point clouds from noise...")
            try:
                output, all_outputs = model(
                    batch,
                    mode='sample',
                    num_inference_steps=50,
                    return_sample_every_n_steps=10,
                    num_points=NUM_POINTS,
                )
            except Exception as exc:
                print(f"Sampling failed for batch {batch_indices}: {exc}")
                continue

            if not isinstance(output, Pointclouds):
                print(f"Unexpected output type for batch {batch_indices}: {type(output)}")
                continue

            pred_lists = output.points_list()
            
            # DIAGNOSTIC: Log output uniqueness
            print("\n[DIAGNOSTIC] Output point cloud hashes:")
            for i, ((name, _), pred_pts) in enumerate(zip(batch_names, pred_lists)):
                pts_hash = hash(pred_pts.cpu().numpy().tobytes())
                pts_mean = pred_pts.mean().item()
                pts_std = pred_pts.std().item()
                print(f"  '{name}': hash={pts_hash}, mean={pts_mean:.4f}, std={pts_std:.4f}")

            for (name, idx_local), pred_points_tensor, image_tensor in zip(batch_names, pred_lists, batch_images):
                pred_points = pred_points_tensor.cpu().numpy()
                print(f"\nGenerated point cloud for '{name}': {pred_points.shape[0]} points")
                print(f"  X range: [{pred_points[:, 0].min():.2f}, {pred_points[:, 0].max():.2f}]")
                print(f"  Y range: [{pred_points[:, 1].min():.2f}, {pred_points[:, 1].max():.2f}]")
                print(f"  Z range: [{pred_points[:, 2].min():.2f}, {pred_points[:, 2].max():.2f}]")

                print("\n" + "="*60)
                print("CONVERTING TO BINARY PNG AND SAVING OUTPUTS")
                print("="*60)

                image_name = name
                img_size = 512
                output_image = np.ones((img_size, img_size), dtype=np.uint8) * 255

                x_coords = pred_points[:, 0]
                y_coords = pred_points[:, 1]

                x_min, x_max = x_coords.min(), x_coords.max()
                y_min, y_max = y_coords.min(), y_coords.max()

                if x_max - x_min < 1e-6:
                    x_normalized = np.ones_like(x_coords) * (img_size / 2)
                else:
                    x_normalized = ((x_coords - x_min) / (x_max - x_min) * (img_size - 1)).astype(int)

                if y_max - y_min < 1e-6:
                    y_normalized = np.ones_like(y_coords) * (img_size / 2)
                else:
                    y_normalized = ((y_coords - y_min) / (y_max - y_min) * (img_size - 1)).astype(int)

                x_normalized = np.clip(x_normalized, 0, img_size - 1)
                y_normalized = np.clip(y_normalized, 0, img_size - 1)

                output_image[y_normalized, x_normalized] = 0

                predict_output_path = PREDICT_DIR / f"{image_name}_predict.png"
                Image.fromarray(output_image).save(predict_output_path)
                print(f"Predicted image saved to: {predict_output_path}")
                print(f"  Image size: {img_size}x{img_size}")
                print(f"  Black pixels: {(output_image == 0).sum()}")
                print(f"  White pixels: {(output_image == 255).sum()}")

                source_output_path = PREDICT_DIR / f"{image_name}_source.png"
                if image_tensor.shape[0] == 1:
                    # Grayscale image
                    source_img_array = (image_tensor.cpu().numpy()[0] * 255).astype(np.uint8)
                    Image.fromarray(source_img_array, mode='L').save(source_output_path)
                else:
                    # RGB image
                    source_img_array = (image_tensor.cpu().numpy().transpose(1, 2, 0) * 255).astype(np.uint8)
                    Image.fromarray(source_img_array).save(source_output_path)
                print(f"Source image saved to: {source_output_path}")

                src_path, tgt_path = dataset.samples[idx_local]
                target_img = Image.open(tgt_path).convert('RGB').resize((img_size, img_size), Image.Resampling.BILINEAR)
                target_output_path = PREDICT_DIR / f"{image_name}_target.png"
                target_img.save(target_output_path)
                print(f"Target (ground truth) image saved to: {target_output_path}")

    print("\n" + "="*60)
    print("[OK] INFERENCE COMPLETE")
    print("="*60)

if __name__ == "__main__":
    SOURCE_DIR = Path(r'/groups/asharf_group/ofirgila/ControlNet/training/data_grads_v3/source')
    TARGET_DIR = Path(r'/groups/asharf_group/ofirgila/ControlNet/training/data_grads_v3/target')
    CHECKPOINT_PATH = Path(r'/groups/asharf_group/ofirgila/projection-conditioned-point-cloud-diffusion/outputs/checkpoint_epoch_2.pth')
    PREDICT_DIR = Path(r'/groups/asharf_group/ofirgila/projection-conditioned-point-cloud-diffusion/outputs/predict')
    NUM_POINTS = 5000  # Number of points to generate
    NUM_IMAGES_TO_PROCESS = 10  # For testing, process only first N image
    BATCH_SIZE = 4  # Batch size for batched prediction
    CLEAN_PREDICT_DIR = True  # Whether to clean output directory before saving
    
    # ====================
    # Model Architecture
    # ====================
    # Set to True to use SimpleCNNFeatureExtractor (trained from scratch on grayscale)
    # Set to False to use ImageNet-pretrained ViT (old default)
    USE_CNN_EXTRACTOR = True
    
    main(use_cnn_extractor=USE_CNN_EXTRACTOR)
