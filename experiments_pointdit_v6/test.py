"""Test/inference script for Point-DiT."""

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


def visualize_stippling(
    image: torch.Tensor,
    points: torch.Tensor,
    original_image: Union[np.ndarray, None] = None,
    save_path: str = None,
    title: str = "Generated Stippling",
):
    """Visualize stippling result."""
    img = image[0, 0].cpu().numpy()
    display_img = original_image if original_image is not None else img
    pts = points[0].cpu().numpy()
    
    H, W = img.shape
    # Convert from [-1, 1] to pixel coordinates
    pts_pix = (pts + 1.0) * np.array([W, H]) / 2.0
    
    fig, axes = plt.subplots(1, 3, figsize=(18, 6))
    
    # Original image
    if display_img.ndim == 3:
        axes[0].imshow(display_img)
    else:
        axes[0].imshow(display_img, cmap='gray')
    axes[0].set_title('Input Image')
    axes[0].axis('off')
    
    # Stippling on white background
    axes[1].scatter(pts_pix[:, 0], pts_pix[:, 1], c='black', s=1, alpha=0.8)
    axes[1].set_xlim(0, W)
    axes[1].set_ylim(H, 0)
    axes[1].set_aspect('equal')
    axes[1].set_facecolor('white')
    axes[1].set_title('Generated Stippling')
    axes[1].axis('off')
    
    # Overlay
    if display_img.ndim == 3:
        axes[2].imshow(display_img, alpha=0.35)
    else:
        axes[2].imshow(display_img, cmap='gray', alpha=0.35)
    axes[2].scatter(pts_pix[:, 0], pts_pix[:, 1], c='red', s=1, alpha=0.7)
    axes[2].set_title('Overlay')
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
    parser = argparse.ArgumentParser(description="Test Point-DiT model")
    # parser.add_argument('--checkpoint', type=str, required=True, help='Path to checkpoint')
    # parser.add_argument('--image', type=str, required=True, help='Path to input image')
    # parser.add_argument('--output', type=str, default=None, help='Output path for visualization')
    # parser.add_argument('--num-inference-steps', type=int, default=50, help='Number of sampling steps')
    # parser.add_argument('--seed', type=int, default=42, help='Random seed')
    args = parser.parse_args()
    
    args.checkpoint = r"/groups/asharf_group/ofirgila/projection-conditioned-point-cloud-diffusion/outputs_pointdit_v6/checkpoint_best.pth"
    args.image = r"/groups/asharf_group/ofirgila/ControlNet/training/fill50k-gs/source/55.png"
    args.output =r"/groups/asharf_group/ofirgila/projection-conditioned-point-cloud-diffusion/experiments_pointdit_v6/outputs_pointdit_v6/sample_55/comparison"
    args.num_inference_steps = 50
    args.seed = 42
    
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
    
    # Load image
    print(f"Loading image from {args.image}")
    image_tensor, display_image = load_image(args.image, config.model.image_size)
    image = image_tensor.to(device)
    
    # Sample
    # Use current default model config points (now 5000) rather than checkpoint's saved value.
    # This ensures predict supports updated output size independent of past configs.
    num_pred_points = Config.default().model.n_points
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
    
    # Visualize
    output_path = args.output or f"stippling_output_{Path(args.image).stem}.png"
    visualize_stippling(
        image,
        points,
        display_image,
        output_path,
        f"Point-DiT Stippling ({num_pred_points} points)",
    )
    
    # Save points as NPY
    npy_path = output_path.replace('.png', '_points.npy')
    np.save(npy_path, points.cpu().numpy())
    print(f"Saved points to {npy_path}")
    
    print("\n✓ Inference complete!")


if __name__ == '__main__':
    main()
