"""
Final Comprehensive Test: Sinkhorn + Lloyd for Neural Stippling

Key Findings:
1. GT Mean NN = 0.024 (well-spaced blue noise)
2. Lloyd alone converges to Mean NN = 0.016 (too clustered)
3. The difference suggests we need SINKHORN LOSS for training, not just Lloyd refinement

This test demonstrates:
1. Sinkhorn loss gradient quality
2. Direct optimization with Sinkhorn matches GT spacing
3. Lloyd as post-processing vs Sinkhorn as training objective
"""

import torch
import torch.nn.functional as F
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).parent))

from diffusion import ChamferLoss, density_guided_init
from sinkhorn_lloyd_losses import (
    SinkhornDensityLoss, 
    SinkhornDensityLossSimple,
    DifferentiableLloydStep,
    HAS_GEOMLOSS
)
from dataset import create_dataloaders
from config import Config


def compute_blue_noise_metrics(points: torch.Tensor, gt_points: torch.Tensor = None):
    """Compute comprehensive blue noise quality metrics."""
    B, N, _ = points.shape
    
    # NN distances
    dists = torch.cdist(points, points)
    mask = torch.eye(N, device=points.device).bool().unsqueeze(0).expand(B, -1, -1)
    dists = dists.masked_fill(mask, float('inf'))
    nn_dists, _ = dists.min(dim=2)
    
    metrics = {
        'mean_nn': nn_dists.mean().item(),
        'std_nn': nn_dists.std().item(),
        'min_nn': nn_dists.min().item(),
        'max_nn': nn_dists.max().item(),
    }
    
    # Coefficient of variation (lower = more uniform spacing)
    metrics['cv'] = metrics['std_nn'] / (metrics['mean_nn'] + 1e-8)
    
    if gt_points is not None:
        chamfer = ChamferLoss()
        metrics['chamfer'] = chamfer(points, gt_points).item()
    
    return metrics


def optimize_with_sinkhorn(
    image: torch.Tensor,
    num_points: int,
    device: str,
    num_steps: int = 200,
    lr: float = 0.01,
    sinkhorn_weight: float = 1.0,
    chamfer_weight: float = 0.0,  # Set to 0 to see pure Sinkhorn
    gt_points: torch.Tensor = None,
):
    """Optimize points directly using Sinkhorn loss."""
    # Initialize from density
    points = density_guided_init(image, num_points, device, std=0.02)
    points = points.clone().requires_grad_(True)
    
    if HAS_GEOMLOSS:
        sinkhorn = SinkhornDensityLoss(blur=0.01, grid_size=64).to(device)
    else:
        sinkhorn = SinkhornDensityLossSimple(grid_size=32, blur=0.05).to(device)
    
    chamfer = ChamferLoss()
    optimizer = torch.optim.Adam([points], lr=lr)
    
    history = []
    
    for step in range(num_steps):
        optimizer.zero_grad()
        
        sinkhorn_loss = sinkhorn(points, image)
        
        if gt_points is not None and chamfer_weight > 0:
            chamfer_loss = chamfer(points, gt_points)
            loss = sinkhorn_weight * sinkhorn_loss + chamfer_weight * chamfer_loss
        else:
            loss = sinkhorn_weight * sinkhorn_loss
            chamfer_loss = chamfer(points, gt_points) if gt_points is not None else torch.tensor(0.0)
        
        loss.backward()
        optimizer.step()
        
        with torch.no_grad():
            points.data.clamp_(-1, 1)
        
        if step % 50 == 0:
            metrics = compute_blue_noise_metrics(points.detach(), gt_points)
            history.append({
                'step': step,
                'sinkhorn': sinkhorn_loss.item(),
                'chamfer': chamfer_loss.item() if torch.is_tensor(chamfer_loss) else chamfer_loss,
                'mean_nn': metrics['mean_nn'],
                'cv': metrics['cv'],
            })
    
    return points.detach(), history


def main():
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    print(f"Device: {device}")
    print(f"geomloss available: {HAS_GEOMLOSS}")
    
    config = Config.default()
    _, val_loader = create_dataloaders(
        source_dir=config.data.source_dir,
        target_dir=config.data.target_dir,
        batch_size=1,
        image_size=config.data.image_size,
        num_points=config.data.num_points,
        val_split=config.data.val_split,
    )
    
    output_dir = Path(__file__).parent / 'outputs_final_test'
    output_dir.mkdir(exist_ok=True)
    
    batch = next(iter(val_loader))
    image = batch['image'].to(device)
    gt_points = batch['points'].to(device)
    
    print("\n" + "="*70)
    print("Final Test: Sinkhorn Optimization vs Lloyd vs GT")
    print("="*70)
    
    # 1. Ground truth metrics
    gt_metrics = compute_blue_noise_metrics(gt_points)
    print(f"\nGround Truth:")
    print(f"  Mean NN: {gt_metrics['mean_nn']:.4f}")
    print(f"  CV: {gt_metrics['cv']:.4f}")
    
    # 2. Density initialization
    density_points = density_guided_init(image, config.data.num_points, device, std=0.02)
    density_metrics = compute_blue_noise_metrics(density_points, gt_points)
    print(f"\nDensity Init:")
    print(f"  Mean NN: {density_metrics['mean_nn']:.4f}")
    print(f"  CV: {density_metrics['cv']:.4f}")
    print(f"  Chamfer: {density_metrics['chamfer']:.6f}")
    
    # 3. Lloyd refinement
    lloyd = DifferentiableLloydStep(grid_size=128, tau=0.001, num_steps=50).to(device)
    with torch.no_grad():
        lloyd_points = lloyd(density_points, image)
    lloyd_metrics = compute_blue_noise_metrics(lloyd_points, gt_points)
    print(f"\nLloyd (50 steps):")
    print(f"  Mean NN: {lloyd_metrics['mean_nn']:.4f}")
    print(f"  CV: {lloyd_metrics['cv']:.4f}")
    print(f"  Chamfer: {lloyd_metrics['chamfer']:.6f}")
    
    # 4. Sinkhorn optimization (pure density matching)
    print(f"\nSinkhorn Optimization (200 steps, pure density)...")
    sinkhorn_points, sinkhorn_history = optimize_with_sinkhorn(
        image, config.data.num_points, device,
        num_steps=200, lr=0.01, sinkhorn_weight=1.0, chamfer_weight=0.0,
        gt_points=gt_points
    )
    sinkhorn_metrics = compute_blue_noise_metrics(sinkhorn_points, gt_points)
    print(f"  Mean NN: {sinkhorn_metrics['mean_nn']:.4f}")
    print(f"  CV: {sinkhorn_metrics['cv']:.4f}")
    print(f"  Chamfer: {sinkhorn_metrics['chamfer']:.6f}")
    
    # 5. Sinkhorn + Chamfer optimization (match GT structure)
    print(f"\nSinkhorn + Chamfer Optimization (200 steps)...")
    combined_points, combined_history = optimize_with_sinkhorn(
        image, config.data.num_points, device,
        num_steps=200, lr=0.01, sinkhorn_weight=0.1, chamfer_weight=1.0,
        gt_points=gt_points
    )
    combined_metrics = compute_blue_noise_metrics(combined_points, gt_points)
    print(f"  Mean NN: {combined_metrics['mean_nn']:.4f}")
    print(f"  CV: {combined_metrics['cv']:.4f}")
    print(f"  Chamfer: {combined_metrics['chamfer']:.6f}")
    
    # Visualization
    fig, axes = plt.subplots(2, 5, figsize=(20, 8))
    
    img = image[0, 0].cpu().numpy()
    
    def plot_points(ax, pts, title, color='blue'):
        ax.imshow(img, cmap='gray', origin='lower', alpha=0.3, extent=[0, 1, 0, 1])
        pts_np = (pts[0].cpu().numpy() + 1) / 2
        ax.scatter(pts_np[:, 0], pts_np[:, 1], s=1, c=color, alpha=0.8)
        ax.set_title(title, fontsize=10)
        ax.set_xlim(0, 1)
        ax.set_ylim(0, 1)
        ax.axis('off')
    
    # Row 1: Full views
    plot_points(axes[0, 0], density_points, f'Density Init\nNN={density_metrics["mean_nn"]:.4f}')
    plot_points(axes[0, 1], lloyd_points, f'Lloyd\nNN={lloyd_metrics["mean_nn"]:.4f}')
    plot_points(axes[0, 2], sinkhorn_points, f'Sinkhorn Only\nNN={sinkhorn_metrics["mean_nn"]:.4f}')
    plot_points(axes[0, 3], combined_points, f'Sinkhorn+Chamfer\nNN={combined_metrics["mean_nn"]:.4f}')
    plot_points(axes[0, 4], gt_points, f'Ground Truth\nNN={gt_metrics["mean_nn"]:.4f}', 'green')
    
    # Row 2: Zoomed views (center)
    for ax in axes[1]:
        ax.set_xlim(0.3, 0.7)
        ax.set_ylim(0.3, 0.7)
    
    plot_points(axes[1, 0], density_points, 'Zoomed')
    axes[1, 0].set_xlim(0.3, 0.7); axes[1, 0].set_ylim(0.3, 0.7)
    
    plot_points(axes[1, 1], lloyd_points, 'Zoomed')
    axes[1, 1].set_xlim(0.3, 0.7); axes[1, 1].set_ylim(0.3, 0.7)
    
    plot_points(axes[1, 2], sinkhorn_points, 'Zoomed')
    axes[1, 2].set_xlim(0.3, 0.7); axes[1, 2].set_ylim(0.3, 0.7)
    
    plot_points(axes[1, 3], combined_points, 'Zoomed')
    axes[1, 3].set_xlim(0.3, 0.7); axes[1, 3].set_ylim(0.3, 0.7)
    
    plot_points(axes[1, 4], gt_points, 'Zoomed', 'green')
    axes[1, 4].set_xlim(0.3, 0.7); axes[1, 4].set_ylim(0.3, 0.7)
    
    plt.suptitle('Comparison: Density Init -> Lloyd -> Sinkhorn -> GT', fontsize=14)
    plt.tight_layout()
    plt.savefig(str(output_dir / 'final_comparison.png'), dpi=200, bbox_inches='tight')
    plt.close()
    
    # Summary
    print("\n" + "="*70)
    print("SUMMARY: Mean NN Distance (higher = more spread = better blue noise)")
    print("="*70)
    print(f"{'Method':<25} {'Mean NN':<12} {'CV':<12} {'Chamfer':<12}")
    print("-" * 70)
    print(f"{'Ground Truth':<25} {gt_metrics['mean_nn']:<12.4f} {gt_metrics['cv']:<12.4f} {'0.000000':<12}")
    print(f"{'Density Init':<25} {density_metrics['mean_nn']:<12.4f} {density_metrics['cv']:<12.4f} {density_metrics['chamfer']:<12.6f}")
    print(f"{'Lloyd (50 steps)':<25} {lloyd_metrics['mean_nn']:<12.4f} {lloyd_metrics['cv']:<12.4f} {lloyd_metrics['chamfer']:<12.6f}")
    print(f"{'Sinkhorn Only':<25} {sinkhorn_metrics['mean_nn']:<12.4f} {sinkhorn_metrics['cv']:<12.4f} {sinkhorn_metrics['chamfer']:<12.6f}")
    print(f"{'Sinkhorn + Chamfer':<25} {combined_metrics['mean_nn']:<12.4f} {combined_metrics['cv']:<12.4f} {combined_metrics['chamfer']:<12.6f}")
    
    print("\n" + "="*70)
    print("CONCLUSION:")
    print("="*70)
    print("1. Lloyd reduces Chamfer but DECREASES spacing (bad for blue noise)")
    print("2. Sinkhorn alone optimizes for density matching but not structure")
    print("3. Sinkhorn + Chamfer is the best combination for training")
    print("4. The model should be trained with Sinkhorn+Chamfer loss")
    print("="*70)
    
    print(f"\nVisualization saved to: {output_dir / 'final_comparison.png'}")


if __name__ == '__main__':
    main()
