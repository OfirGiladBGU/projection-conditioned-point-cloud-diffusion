"""
Deep Analysis of Ground Truth vs Generated Points

This script investigates why our methods produce points with smaller
nearest-neighbor distances than ground truth.
"""

import torch
import torch.nn.functional as F
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).parent))

from diffusion import ChamferLoss, density_guided_init
from sinkhorn_lloyd_losses import DifferentiableLloydStep, DifferentiableCCVT
from dataset import create_dataloaders
from config import Config


def analyze_point_distribution(points: torch.Tensor, name: str):
    """Detailed analysis of point distribution."""
    B, N, _ = points.shape
    
    # NN distances
    dists = torch.cdist(points, points)
    mask = torch.eye(N, device=points.device).bool().unsqueeze(0).expand(B, -1, -1)
    dists = dists.masked_fill(mask, float('inf'))
    nn_dists, _ = dists.min(dim=2)
    
    # Spatial coverage (how much of [-1,1]^2 is covered)
    pts = points[0].cpu().numpy()
    x_range = pts[:, 0].max() - pts[:, 0].min()
    y_range = pts[:, 1].max() - pts[:, 1].min()
    
    print(f"\n{name}:")
    print(f"  Points: {N}")
    print(f"  NN dist: min={nn_dists.min().item():.4f}, mean={nn_dists.mean().item():.4f}, "
          f"max={nn_dists.max().item():.4f}, std={nn_dists.std().item():.4f}")
    print(f"  Spatial range: x=[{pts[:,0].min():.3f}, {pts[:,0].max():.3f}], "
          f"y=[{pts[:,1].min():.3f}, {pts[:,1].max():.3f}]")
    print(f"  Coverage: {x_range:.3f} x {y_range:.3f}")
    
    return nn_dists[0].cpu().numpy()


def visualize_nn_histograms(
    nn_dists_dict: dict,
    save_path: str,
):
    """Compare NN distance distributions."""
    fig, ax = plt.subplots(1, 1, figsize=(10, 6))
    
    colors = ['red', 'blue', 'green', 'orange', 'purple']
    for i, (name, nn_dists) in enumerate(nn_dists_dict.items()):
        ax.hist(nn_dists, bins=50, alpha=0.5, label=name, color=colors[i % len(colors)])
    
    ax.set_xlabel('Nearest Neighbor Distance')
    ax.set_ylabel('Count')
    ax.set_title('Distribution of NN Distances')
    ax.legend()
    
    plt.tight_layout()
    plt.savefig(save_path, dpi=150)
    plt.close()
    print(f"Saved: {save_path}")


def visualize_detailed_comparison(
    image: torch.Tensor,
    points_dict: dict,
    save_path: str,
):
    """Create detailed side-by-side comparison with zoom."""
    n_cols = len(points_dict)
    fig, axes = plt.subplots(3, n_cols, figsize=(4 * n_cols, 12))
    
    img = image[0, 0].cpu().numpy()
    
    for col, (name, points) in enumerate(points_dict.items()):
        pts = (points[0].cpu().numpy() + 1) / 2  # [-1,1] -> [0,1]
        
        # Row 1: Full image with points
        axes[0, col].imshow(img, cmap='gray', origin='lower', alpha=0.3, extent=[0, 1, 0, 1])
        axes[0, col].scatter(pts[:, 0], pts[:, 1], s=1, c='blue', alpha=0.8)
        axes[0, col].set_title(f'{name}\n({len(pts)} pts)')
        axes[0, col].set_xlim(0, 1)
        axes[0, col].set_ylim(0, 1)
        axes[0, col].axis('off')
        
        # Row 2: Zoom to center
        axes[1, col].imshow(img, cmap='gray', origin='lower', alpha=0.3, extent=[0, 1, 0, 1])
        axes[1, col].scatter(pts[:, 0], pts[:, 1], s=5, c='blue', alpha=0.9)
        axes[1, col].set_xlim(0.35, 0.65)
        axes[1, col].set_ylim(0.35, 0.65)
        axes[1, col].set_title('Center Zoom')
        axes[1, col].axis('off')
        
        # Row 3: NN distance histogram
        B, N, _ = points.shape
        dists = torch.cdist(points, points)
        mask = torch.eye(N, device=points.device).bool().unsqueeze(0)
        dists = dists.masked_fill(mask, float('inf'))
        nn_dists = dists.min(dim=2).values[0].cpu().numpy()
        
        axes[2, col].hist(nn_dists, bins=30, color='blue', alpha=0.7)
        axes[2, col].set_xlabel('NN Distance')
        axes[2, col].set_ylabel('Count')
        mean_nn = nn_dists.mean()
        axes[2, col].axvline(mean_nn, color='red', linestyle='--', label=f'Mean={mean_nn:.4f}')
        axes[2, col].legend()
        axes[2, col].set_title('NN Distribution')
    
    plt.tight_layout()
    plt.savefig(save_path, dpi=200, bbox_inches='tight')
    plt.close()
    print(f"Saved: {save_path}")


def main():
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    print(f"Device: {device}")
    
    config = Config.default()
    _, val_loader = create_dataloaders(
        source_dir=config.data.source_dir,
        target_dir=config.data.target_dir,
        batch_size=1,
        image_size=config.data.image_size,
        num_points=config.data.num_points,
        val_split=config.data.val_split,
    )
    
    output_dir = Path(__file__).parent / 'outputs_analysis'
    output_dir.mkdir(exist_ok=True)
    
    # Refinement methods
    lloyd = DifferentiableLloydStep(grid_size=128, tau=0.001, num_steps=30).to(device)
    ccvt = DifferentiableCCVT(grid_size=128, tau=0.001, num_steps=50, weight_lr=0.3).to(device)
    
    print("\n" + "="*70)
    print("Ground Truth Analysis")
    print("="*70)
    
    batch = next(iter(val_loader))
    image = batch['image'].to(device)
    gt_points = batch['points'].to(device)
    
    # Different initializations
    density_points = density_guided_init(image, config.data.num_points, device, std=0.02)
    
    with torch.no_grad():
        lloyd_points = lloyd(density_points, image)
        ccvt_points = ccvt(density_points, image)
    
    # Analyze each
    nn_dict = {}
    nn_dict['GT'] = analyze_point_distribution(gt_points, "Ground Truth")
    nn_dict['Density Init'] = analyze_point_distribution(density_points, "Density Init")
    nn_dict['Lloyd'] = analyze_point_distribution(lloyd_points, "Lloyd (30 steps)")
    nn_dict['CCVT'] = analyze_point_distribution(ccvt_points, "CCVT (50 steps)")
    
    # Visualize NN histograms
    visualize_nn_histograms(nn_dict, str(output_dir / 'nn_histograms.png'))
    
    # Detailed comparison
    visualize_detailed_comparison(
        image,
        {
            'Density Init': density_points,
            'Lloyd': lloyd_points,
            'CCVT': ccvt_points,
            'Ground Truth': gt_points,
        },
        str(output_dir / 'detailed_comparison.png')
    )
    
    # Key insight check: Does GT have more points than we're generating?
    print(f"\n" + "="*70)
    print("KEY FINDING:")
    print("="*70)
    print(f"GT points: {gt_points.shape[1]}")
    print(f"Our points: {config.data.num_points}")
    print(f"\nGT Mean NN: {nn_dict['GT'].mean():.4f}")
    print(f"Lloyd Mean NN: {nn_dict['Lloyd'].mean():.4f}")
    print(f"Ratio: {nn_dict['GT'].mean() / nn_dict['Lloyd'].mean():.2f}x")
    
    # If GT has larger NN distances, it might mean GT is sparser or more spread out
    # This could indicate the GT was generated differently
    
    print(f"\nConclusion:")
    if nn_dict['GT'].mean() > nn_dict['Lloyd'].mean():
        print("GT has LARGER NN distances = GT points are MORE spread out")
        print("This suggests GT uses different blue noise parameters or algorithm")
    else:
        print("GT has SMALLER NN distances = GT points are more clustered")
    
    print(f"\nVisualizations saved to: {output_dir}")


if __name__ == '__main__':
    main()
