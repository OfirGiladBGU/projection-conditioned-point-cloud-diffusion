"""Evaluation script for Point-DiT with ground truth comparison."""

import torch
import numpy as np
import argparse
from pathlib import Path
from PIL import Image
import matplotlib.pyplot as plt
import os
from typing import Union

from config import Config
from model import PointDiT
from diffusion import DDPMScheduler, sample


def load_model(checkpoint_path: str, device: str = 'cuda'):
    """Load trained model from checkpoint."""
    checkpoint = torch.load(checkpoint_path, map_location=device, weights_only=False)
    config = checkpoint.get('config', Config.default())
    
    model = PointDiT(
        n_points=config.model.n_points,
        dim=config.model.dim,
        n_layers=config.model.n_layers,
        n_heads=config.model.n_heads,
        image_size=config.model.image_size,
        dropout=0.0,  # No dropout for inference
    ).to(device)
    
    model.load_state_dict(checkpoint['model_state_dict'])
    model.eval()
    
    print(f"Loaded model from epoch {checkpoint['epoch']}")
    print(f"Val loss: {checkpoint.get('val_loss', 'N/A')}")
    
    return model, config


def load_image(image_path: str, size: int = 512) -> tuple[torch.Tensor, np.ndarray]:
    """Load image for the model (grayscale) and keep the color original for display."""
    original_color = Image.open(image_path).convert('RGB').resize((size, size), Image.BILINEAR)
    img_gray = original_color.convert('L')
    img_np = np.asarray(img_gray, dtype=np.float32) / 255.0
    img_tensor = torch.from_numpy(img_np).float().unsqueeze(0).unsqueeze(0)
    display_np = np.asarray(original_color)
    return img_tensor, display_np


def load_target_points(target_path: str) -> np.ndarray:
    """Load ground truth points from .npy file."""
    points = np.load(target_path)
    # Ensure shape is (n_points, 2)
    if points.ndim == 3:
        points = points[0]  # Remove batch dimension if present
    return points


def extract_points_from_image(image_path: str, size: int = 512, num_points: int = None) -> np.ndarray:
    """Extract stipple points from target image (black pixels = stipple locations)."""
    target = Image.open(image_path).convert('1').resize((size, size), Image.NEAREST)
    target_np = np.asarray(target, dtype=bool)
    
    # Extract stipple points (black pixels = False = stipple locations)
    ys, xs = np.where(~target_np)  # Get coordinates of False (black) pixels
    
    if len(xs) == 0:
        # No stipple points found
        return np.array([]).reshape(0, 2)
    
    if num_points is not None and len(xs) > num_points:
        # Sample a subset of points
        indices = np.random.choice(len(xs), size=num_points, replace=False)
        xs, ys = xs[indices], ys[indices]
    
    points_pixel = np.stack([xs, ys], axis=1).astype(np.float32)
    # Normalize to [-1, 1]
    points = (points_pixel / (size - 1)) * 2.0 - 1.0
    
    return points


def visualize_comparison(
    source_image: np.ndarray,
    gt_points: np.ndarray,
    pred_points: np.ndarray,
    save_path: str = None,
    title: str = "Point-DiT Evaluation",
):
    """Visualize comparison between ground truth and predicted stippling."""
    H, W = source_image.shape[:2]
    
    # Convert from [-1, 1] to pixel coordinates
    gt_pix = (gt_points + 1.0) * np.array([W, H]) / 2.0
    pred_pix = (pred_points + 1.0) * np.array([W, H]) / 2.0
    
    fig, axes = plt.subplots(1, 3, figsize=(18, 6))
    
    # Input image
    if source_image.ndim == 3:
        axes[0].imshow(source_image)
    else:
        axes[0].imshow(source_image, cmap='gray')
    axes[0].set_title('Input Image')
    axes[0].axis('off')
    
    # Ground truth points
    axes[1].scatter(gt_pix[:, 0], gt_pix[:, 1], c='black', s=1, alpha=0.8)
    axes[1].set_xlim(0, W)
    axes[1].set_ylim(H, 0)
    axes[1].set_aspect('equal')
    axes[1].set_facecolor('white')
    axes[1].set_title('Ground Truth')
    axes[1].axis('off')
    
    # Predicted points
    axes[2].scatter(pred_pix[:, 0], pred_pix[:, 1], c='black', s=1, alpha=0.8)
    axes[2].set_xlim(0, W)
    axes[2].set_ylim(H, 0)
    axes[2].set_aspect('equal')
    axes[2].set_facecolor('white')
    axes[2].set_title('Predicted')
    axes[2].axis('off')
    
    plt.suptitle(title)
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
        print(f"Saved visualization to {save_path}")
    else:
        plt.show()
    
    plt.close()


def main():
    parser = argparse.ArgumentParser(description="Evaluate Point-DiT model with GT comparison")
    parser.add_argument('--checkpoint', type=str, required=True, help='Path to checkpoint')
    parser.add_argument('--source', type=str, required=True, help='Path to input source image')
    parser.add_argument('--target', type=str, required=True, help='Path to ground truth points (.npy)')
    parser.add_argument('--output', type=str, default=None, help='Output path for visualization')
    parser.add_argument('--num-inference-steps', type=int, default=50, help='Number of sampling steps')
    parser.add_argument('--seed', type=int, default=42, help='Random seed')
    args = parser.parse_args()
    
    # Set seed
    torch.manual_seed(args.seed)
    np.random.seed(args.seed)
    
    # Device
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")
    
    # Load model
    print(f"Loading model from {args.checkpoint}")
    model, config = load_model(args.checkpoint, device)
    
    # Create scheduler
    scheduler = DDPMScheduler(
        num_train_timesteps=config.diffusion.num_train_timesteps,
        beta_start=config.diffusion.beta_start,
        beta_end=config.diffusion.beta_end,
        beta_schedule=config.diffusion.beta_schedule,
    )
    
    # Load source image
    print(f"Loading source image from {args.source}")
    image_tensor, display_image = load_image(args.source, config.model.image_size)
    image = image_tensor.to(device)
    
    # Load ground truth points
    print(f"Loading ground truth from {args.target}")
    if args.target.endswith('.npy'):
        gt_points = load_target_points(args.target)
    else:
        # Extract points from target image
        print("Extracting points from target image...")
        gt_points = extract_points_from_image(args.target, config.model.image_size)
        print(f"Extracted {len(gt_points)} ground truth points")
    
    # Decide how many points to generate: match GT count if available
    num_pred_points = (
        len(gt_points) if gt_points is not None and gt_points.size > 0 else config.model.n_points
    )
    print(f"Generating stippling with {args.num_inference_steps} steps (N={num_pred_points})...")
    with torch.no_grad():
        points = sample(
            model,
            scheduler,
            image,
            num_pred_points,
            args.num_inference_steps,
            device,
            show_progress=True,
        )
    
    pred_points = points[0].cpu().numpy()
    
    # Visualize comparison
    output_path = args.output or f"eval_output_{Path(args.source).stem}.png"
    
    if gt_points is not None:
        visualize_comparison(
            display_image,
            gt_points,
            pred_points,
            output_path,
            f"Point-DiT Evaluation ({num_pred_points} points)",
        )
    else:
        print("Could not load ground truth points for comparison")
    
    # Save predicted points as NPY
    npy_path = output_path.replace('.png', '_pred_points.npy')
    np.save(npy_path, pred_points)
    print(f"Saved predicted points to {npy_path}")
    
    print("\n✓ Evaluation complete!")


if __name__ == '__main__':
    main()
