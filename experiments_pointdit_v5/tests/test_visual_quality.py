"""
Visual Quality Test for Sinkhorn + Lloyd Pipeline

This test creates clear visualizations showing:
1. How density-guided initialization works
2. How Lloyd refinement improves point distribution
3. Comparison with ground truth
"""

import torch
import torch.nn.functional as F
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).parent))

from diffusion import ChamferLoss, density_guided_init
from sinkhorn_lloyd_losses import DifferentiableLloydStep, DifferentiableCCVT, DifferentiableLloydWithRepulsion, HAS_GEOMLOSS
from dataset import create_dataloaders
from config import Config


def compute_metrics(points: torch.Tensor, gt_points: torch.Tensor, image: torch.Tensor) -> dict:
    """Compute quality metrics."""
    B, N, _ = points.shape
    
    # Chamfer distance
    chamfer = ChamferLoss()
    chamfer_val = chamfer(points, gt_points).item()
    
    # Nearest neighbor statistics
    dists = torch.cdist(points, points)
    mask = torch.eye(N, device=points.device).bool().unsqueeze(0).expand(B, -1, -1)
    dists = dists.masked_fill(mask, float('inf'))
    nn_dists, _ = dists.min(dim=2)
    
    return {
        'chamfer': chamfer_val,
        'mean_nn': nn_dists.mean().item(),
        'std_nn': nn_dists.std().item(),
        'min_nn': nn_dists.min().item(),
    }


def visualize_stippling_quality(
    image: torch.Tensor,
    points_dict: dict,  # {'name': points_tensor}
    save_path: str,
    title: str = "",
):
    """Create a detailed visualization of stippling quality."""
    n_cols = len(points_dict)
    fig, axes = plt.subplots(2, n_cols, figsize=(5 * n_cols, 10))
    
    img = image[0, 0].cpu().numpy()
    H, W = img.shape
    
    for col, (name, points) in enumerate(points_dict.items()):
        pts = points[0].cpu().numpy()
        pts_pixel = (pts + 1) / 2  # [-1,1] -> [0,1]
        
        # Top row: Points on image
        ax = axes[0, col]
        ax.imshow(img, cmap='gray', origin='lower', alpha=0.3, extent=[0, 1, 0, 1])
        ax.scatter(pts_pixel[:, 0], pts_pixel[:, 1], s=2, c='blue', alpha=0.9)
        ax.set_title(f'{name}\n({len(pts)} points)', fontsize=12)
        ax.set_xlim(0, 1)
        ax.set_ylim(0, 1)
        ax.set_aspect('equal')
        ax.axis('off')
        
        # Bottom row: Zoomed region (center crop)
        ax = axes[1, col]
        ax.imshow(img, cmap='gray', origin='lower', alpha=0.3, extent=[0, 1, 0, 1])
        ax.scatter(pts_pixel[:, 0], pts_pixel[:, 1], s=8, c='blue', alpha=0.9)
        ax.set_xlim(0.3, 0.7)  # Zoom to center
        ax.set_ylim(0.3, 0.7)
        ax.set_title('Zoomed (center)', fontsize=10)
        ax.set_aspect('equal')
    
    if title:
        fig.suptitle(title, fontsize=14, fontweight='bold')
    
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
    
    output_dir = Path(__file__).parent / 'outputs_visual_test'
    output_dir.mkdir(exist_ok=True)
    
    # Lloyd with optimized parameters (from our analysis)
    lloyd = DifferentiableLloydStep(
        grid_size=128,  # Higher resolution
        tau=0.001,      # Sharp Voronoi
        num_steps=20    # More iterations
    ).to(device)
    
    # CCVT - Capacity Constrained (true blue noise)
    ccvt = DifferentiableCCVT(
        grid_size=128,
        tau=0.001,
        num_steps=30,
        weight_lr=0.2
    ).to(device)
    
    # Lloyd with Repulsion - prevents point collapse
    lloyd_repulsion = DifferentiableLloydWithRepulsion(
        grid_size=128,
        tau=0.001,
        num_steps=30,
        repulsion_strength=1.0,  # Strong repulsion
        repulsion_radius=0.02    # Similar to GT min NN
    ).to(device)
    
    print("\n" + "="*60)
    print("Visual Quality Test: Density Init -> Lloyd/CCVT/Lloyd+Rep -> GT")
    print("="*60)
    
    for sample_idx, batch in enumerate(val_loader):
        if sample_idx >= 5:  # Test 5 samples
            break
        
        image = batch['image'].to(device)
        gt_points = batch['points'].to(device)
        
        # 1. Random initialization
        random_points = (torch.rand_like(gt_points) * 2 - 1)
        
        # 2. Density-guided initialization
        density_points = density_guided_init(image, config.data.num_points, device, std=0.02)
        
        # 3. Lloyd refinement on density points
        with torch.no_grad():
            lloyd_points = lloyd(density_points, image)
        
        # 4. CCVT refinement (capacity constrained)
        with torch.no_grad():
            ccvt_points = ccvt(density_points, image)
        
        # 5. Lloyd with Repulsion
        with torch.no_grad():
            lloyd_rep_points = lloyd_repulsion(density_points, image)
        
        # Compute metrics
        metrics_random = compute_metrics(random_points, gt_points, image)
        metrics_density = compute_metrics(density_points, gt_points, image)
        metrics_lloyd = compute_metrics(lloyd_points, gt_points, image)
        metrics_ccvt = compute_metrics(ccvt_points, gt_points, image)
        metrics_lloyd_rep = compute_metrics(lloyd_rep_points, gt_points, image)
        metrics_gt = compute_metrics(gt_points, gt_points, image)
        
        print(f"\nSample {sample_idx}:")
        print(f"  Density:   Chamfer={metrics_density['chamfer']:.6f}, Mean NN={metrics_density['mean_nn']:.4f}")
        print(f"  Lloyd:     Chamfer={metrics_lloyd['chamfer']:.6f}, Mean NN={metrics_lloyd['mean_nn']:.4f}")
        print(f"  CCVT:      Chamfer={metrics_ccvt['chamfer']:.6f}, Mean NN={metrics_ccvt['mean_nn']:.4f}")
        print(f"  Lloyd+Rep: Chamfer={metrics_lloyd_rep['chamfer']:.6f}, Mean NN={metrics_lloyd_rep['mean_nn']:.4f}")
        print(f"  GT:        Chamfer={metrics_gt['chamfer']:.6f}, Mean NN={metrics_gt['mean_nn']:.4f}")
        
        # Visualize
        visualize_stippling_quality(
            image,
            {
                'Density Init': density_points,
                'Lloyd+Repulsion': lloyd_rep_points,
                'Ground Truth': gt_points,
            },
            str(output_dir / f'sample_{sample_idx}_quality.png'),
            title=f'Sample {sample_idx}: Lloyd+Rep Mean NN={metrics_lloyd_rep["mean_nn"]:.4f} vs GT={metrics_gt["mean_nn"]:.4f}'
        )
    
    print("\n" + "="*60)
    print(f"Visualizations saved to: {output_dir}")
    print("="*60)


if __name__ == '__main__':
    main()
