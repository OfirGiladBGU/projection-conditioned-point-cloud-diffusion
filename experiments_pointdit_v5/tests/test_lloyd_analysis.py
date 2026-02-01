"""
Improved Test for Sinkhorn + Lloyd Pipeline

This test focuses on evaluating Lloyd refinement quality more thoroughly:
1. Tests Lloyd with different iteration counts (1, 5, 10, 20)
2. Tests different grid resolutions (32, 64, 128)
3. Tests different tau (temperature) values
4. Compares against ground truth properly

Key insight from TODO: Lloyd should move points to weighted centroids.
The goal is to find optimal parameters for the refinement layer.
"""

import torch
import torch.nn.functional as F
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path
import sys
import time

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


def compute_blue_noise_quality(points: torch.Tensor, image: torch.Tensor) -> dict:
    """
    Compute blue noise quality metrics for a point set.
    
    Blue noise characteristics:
    1. No clumping (minimum distance between points)
    2. Even spacing (variance of nearest neighbor distances)
    3. Density matching (correlation with image)
    """
    B, N, _ = points.shape
    
    # 1. Nearest neighbor distances
    dists = torch.cdist(points, points)  # (B, N, N)
    # Mask diagonal
    mask = torch.eye(N, device=points.device).bool().unsqueeze(0).expand(B, -1, -1)
    dists = dists.masked_fill(mask, float('inf'))
    nn_dists, _ = dists.min(dim=2)  # (B, N)
    
    # 2. Min distance (should be non-zero for blue noise)
    min_dist = nn_dists.min(dim=1).values.mean().item()
    
    # 3. Mean NN distance
    mean_nn_dist = nn_dists.mean().item()
    
    # 4. Variance of NN distances (lower = more uniform)
    std_nn_dist = nn_dists.std().item()
    
    # 5. Coefficient of variation (CV) - lower is better for blue noise
    cv = std_nn_dist / (mean_nn_dist + 1e-8)
    
    return {
        'min_dist': min_dist,
        'mean_nn_dist': mean_nn_dist,
        'std_nn_dist': std_nn_dist,
        'cv': cv,  # Coefficient of variation - key blue noise metric
    }


def visualize_lloyd_progression(
    image: torch.Tensor,
    initial_points: torch.Tensor,
    gt_points: torch.Tensor,
    lloyd_steps_list: list,
    save_path: str,
):
    """Visualize how points evolve through Lloyd iterations."""
    n_cols = len(lloyd_steps_list) + 3  # Initial + steps + GT
    fig, axes = plt.subplots(1, n_cols, figsize=(4 * n_cols, 4))
    
    img = image[0, 0].cpu().numpy()
    
    def plot_points(ax, pts, title, color='blue'):
        ax.imshow(img, cmap='gray', origin='lower', alpha=0.3, extent=[0, 1, 0, 1])
        pts_np = (pts[0].cpu().numpy() + 1) / 2  # [-1,1] -> [0,1]
        ax.scatter(pts_np[:, 0], pts_np[:, 1], s=1, c=color, alpha=0.8)
        ax.set_title(title)
        ax.set_xlim(0, 1)
        ax.set_ylim(0, 1)
        ax.set_aspect('equal')
        ax.axis('off')
    
    # Initial points
    plot_points(axes[0], initial_points, 'Initial (random)', 'red')
    
    # Lloyd progressions
    for i, (steps, pts) in enumerate(lloyd_steps_list):
        plot_points(axes[i + 1], pts, f'Lloyd {steps} steps', 'blue')
    
    # Ground truth
    plot_points(axes[-1], gt_points, 'Ground Truth', 'green')
    
    plt.tight_layout()
    plt.savefig(save_path, dpi=150, bbox_inches='tight')
    plt.close()
    print(f"Saved: {save_path}")


def test_lloyd_iterations(device: str = 'cuda'):
    """Test Lloyd refinement with different iteration counts."""
    print("\n" + "="*70)
    print("TEST: Lloyd Iterations Analysis")
    print("="*70)
    
    config = Config.default()
    train_loader, val_loader = create_dataloaders(
        source_dir=config.data.source_dir,
        target_dir=config.data.target_dir,
        batch_size=1,
        image_size=config.data.image_size,
        num_points=config.data.num_points,
        val_split=config.data.val_split,
    )
    
    output_dir = Path(__file__).parent / 'outputs_lloyd_analysis'
    output_dir.mkdir(exist_ok=True)
    
    chamfer = ChamferLoss()
    
    # Test parameters
    iteration_counts = [1, 2, 5, 10, 20, 50]
    grid_sizes = [32, 64, 128]
    tau_values = [0.001, 0.01, 0.05, 0.1]
    
    # Get a few samples
    samples = []
    for i, batch in enumerate(val_loader):
        if i >= 3:
            break
        samples.append(batch)
    
    print(f"\nTesting on {len(samples)} samples")
    print(f"Iteration counts: {iteration_counts}")
    print(f"Grid sizes: {grid_sizes}")
    print(f"Tau values: {tau_values}")
    
    # Best config tracking
    best_configs = []
    
    for sample_idx, batch in enumerate(samples):
        print(f"\n--- Sample {sample_idx} ---")
        
        image = batch['image'].to(device)
        gt_points = batch['points'].to(device)
        
        # Initialize points from density (realistic starting point)
        initial_points = density_guided_init(image, config.data.num_points, device, std=0.02)
        
        initial_chamfer = chamfer(initial_points, gt_points).item()
        initial_quality = compute_blue_noise_quality(initial_points, image)
        print(f"Initial: Chamfer={initial_chamfer:.6f}, CV={initial_quality['cv']:.4f}")
        
        results = []
        
        # Test different configurations
        for grid_size in grid_sizes:
            for tau in tau_values:
                for n_steps in iteration_counts:
                    lloyd = DifferentiableLloydStep(
                        grid_size=grid_size, 
                        tau=tau, 
                        num_steps=n_steps
                    ).to(device)
                    
                    start_time = time.time()
                    with torch.no_grad():
                        refined = lloyd(initial_points, image)
                    elapsed = time.time() - start_time
                    
                    ref_chamfer = chamfer(refined, gt_points).item()
                    ref_quality = compute_blue_noise_quality(refined, image)
                    
                    results.append({
                        'grid': grid_size,
                        'tau': tau,
                        'steps': n_steps,
                        'chamfer': ref_chamfer,
                        'cv': ref_quality['cv'],
                        'mean_nn': ref_quality['mean_nn_dist'],
                        'time': elapsed,
                    })
        
        # Find best config
        best = min(results, key=lambda x: x['chamfer'])
        best_configs.append(best)
        
        print(f"Best config: grid={best['grid']}, tau={best['tau']}, steps={best['steps']}")
        print(f"  Chamfer: {initial_chamfer:.6f} -> {best['chamfer']:.6f} "
              f"({100*(best['chamfer']-initial_chamfer)/initial_chamfer:+.1f}%)")
        print(f"  CV: {initial_quality['cv']:.4f} -> {best['cv']:.4f}")
        
        # Visualize progression with best tau and grid
        lloyd_progression = []
        for n_steps in [1, 5, 10, 20, 50]:
            lloyd = DifferentiableLloydStep(
                grid_size=best['grid'], 
                tau=best['tau'], 
                num_steps=n_steps
            ).to(device)
            with torch.no_grad():
                refined = lloyd(initial_points, image)
            lloyd_progression.append((n_steps, refined))
        
        visualize_lloyd_progression(
            image, initial_points, gt_points, lloyd_progression,
            str(output_dir / f'sample_{sample_idx}_lloyd_progression.png')
        )
    
    # Summary
    print("\n" + "="*70)
    print("SUMMARY: Best Configurations")
    print("="*70)
    
    avg_grid = np.mean([c['grid'] for c in best_configs])
    avg_tau = np.mean([c['tau'] for c in best_configs])
    avg_steps = np.mean([c['steps'] for c in best_configs])
    
    print(f"Average best: grid={avg_grid:.0f}, tau={avg_tau:.3f}, steps={avg_steps:.0f}")
    
    # Find overall best
    for c in best_configs:
        print(f"  grid={c['grid']}, tau={c['tau']:.3f}, steps={c['steps']}: "
              f"Chamfer={c['chamfer']:.6f}, CV={c['cv']:.4f}")
    
    return best_configs


def test_sinkhorn_vs_chamfer(device: str = 'cuda'):
    """Compare Sinkhorn loss gradient signal vs Chamfer."""
    print("\n" + "="*70)
    print("TEST: Sinkhorn vs Chamfer Gradient Analysis")
    print("="*70)
    
    config = Config.default()
    train_loader, _ = create_dataloaders(
        source_dir=config.data.source_dir,
        target_dir=config.data.target_dir,
        batch_size=4,
        image_size=config.data.image_size,
        num_points=config.data.num_points,
        val_split=config.data.val_split,
    )
    
    chamfer = ChamferLoss()
    if HAS_GEOMLOSS:
        sinkhorn = SinkhornDensityLoss(blur=0.01, grid_size=64).to(device)
    else:
        sinkhorn = SinkhornDensityLossSimple(grid_size=32, blur=0.05).to(device)
    
    batch = next(iter(train_loader))
    image = batch['image'].to(device)
    gt_points = batch['points'].to(device)
    
    # Test gradient quality at different noise levels
    noise_levels = [0.01, 0.05, 0.1, 0.2, 0.5]
    
    print("\nGradient analysis at different noise levels:")
    print("-" * 70)
    print(f"{'Noise':>8} | {'Chamfer Loss':>12} | {'Chamfer Grad':>12} | {'Sinkhorn Loss':>13} | {'Sinkhorn Grad':>13}")
    print("-" * 70)
    
    for noise in noise_levels:
        # Add noise to GT points
        noisy_points = gt_points + torch.randn_like(gt_points) * noise
        noisy_points = noisy_points.clamp(-1, 1)
        
        # Chamfer gradient
        noisy_points.requires_grad_(True)
        chamfer_loss = chamfer(noisy_points, gt_points)
        chamfer_loss.backward()
        chamfer_grad_norm = noisy_points.grad.norm().item()
        noisy_points.grad.zero_()
        
        # Sinkhorn gradient
        sinkhorn_loss = sinkhorn(noisy_points, image)
        sinkhorn_loss.backward()
        sinkhorn_grad_norm = noisy_points.grad.norm().item()
        noisy_points.requires_grad_(False)
        
        print(f"{noise:>8.2f} | {chamfer_loss.item():>12.6f} | {chamfer_grad_norm:>12.6f} | "
              f"{sinkhorn_loss.item():>13.6f} | {sinkhorn_grad_norm:>13.6f}")
    
    print("-" * 70)
    print("\nInterpretation:")
    print("- Chamfer gradient should increase with noise (drives points back to GT)")
    print("- Sinkhorn gradient provides density-matching signal (independent of GT)")


def test_combined_loss_optimization(device: str = 'cuda'):
    """Test optimizing points directly with combined loss (no neural network)."""
    print("\n" + "="*70)
    print("TEST: Direct Point Optimization with Combined Loss")
    print("="*70)
    
    config = Config.default()
    _, val_loader = create_dataloaders(
        source_dir=config.data.source_dir,
        target_dir=config.data.target_dir,
        batch_size=1,
        image_size=config.data.image_size,
        num_points=config.data.num_points,
        val_split=config.data.val_split,
    )
    
    output_dir = Path(__file__).parent / 'outputs_lloyd_analysis'
    output_dir.mkdir(exist_ok=True)
    
    batch = next(iter(val_loader))
    image = batch['image'].to(device)
    gt_points = batch['points'].to(device)
    
    # Initialize from density
    points = density_guided_init(image, config.data.num_points, device, std=0.05)
    points = points.clone().requires_grad_(True)
    
    chamfer = ChamferLoss()
    if HAS_GEOMLOSS:
        sinkhorn = SinkhornDensityLoss(blur=0.01, grid_size=64).to(device)
    else:
        sinkhorn = SinkhornDensityLossSimple(grid_size=32, blur=0.05).to(device)
    
    optimizer = torch.optim.Adam([points], lr=0.01)
    
    print("\nOptimizing points directly (no network):")
    print("Loss = Chamfer + 0.1 * Sinkhorn")
    print("-" * 50)
    
    history = []
    snapshots = []
    
    for step in range(200):
        optimizer.zero_grad()
        
        chamfer_loss = chamfer(points, gt_points)
        sinkhorn_loss = sinkhorn(points, image)
        loss = chamfer_loss + 0.1 * sinkhorn_loss
        
        loss.backward()
        optimizer.step()
        
        # Clamp to valid range
        with torch.no_grad():
            points.data.clamp_(-1, 1)
        
        if step % 20 == 0:
            quality = compute_blue_noise_quality(points.detach(), image)
            history.append({
                'step': step,
                'chamfer': chamfer_loss.item(),
                'sinkhorn': sinkhorn_loss.item(),
                'cv': quality['cv'],
            })
            print(f"Step {step:3d}: Chamfer={chamfer_loss.item():.6f}, "
                  f"Sinkhorn={sinkhorn_loss.item():.6f}, CV={quality['cv']:.4f}")
            
            if step in [0, 50, 100, 199]:
                snapshots.append((step, points.detach().clone()))
    
    # Visualize optimization progression
    fig, axes = plt.subplots(1, len(snapshots) + 1, figsize=(4 * (len(snapshots) + 1), 4))
    img = image[0, 0].cpu().numpy()
    
    for i, (step, pts) in enumerate(snapshots):
        axes[i].imshow(img, cmap='gray', origin='lower', alpha=0.3, extent=[0, 1, 0, 1])
        pts_np = (pts[0].cpu().numpy() + 1) / 2
        axes[i].scatter(pts_np[:, 0], pts_np[:, 1], s=1, c='blue', alpha=0.8)
        axes[i].set_title(f'Step {step}')
        axes[i].set_xlim(0, 1)
        axes[i].set_ylim(0, 1)
        axes[i].axis('off')
    
    # GT
    gt_np = (gt_points[0].cpu().numpy() + 1) / 2
    axes[-1].imshow(img, cmap='gray', origin='lower', alpha=0.3, extent=[0, 1, 0, 1])
    axes[-1].scatter(gt_np[:, 0], gt_np[:, 1], s=1, c='green', alpha=0.8)
    axes[-1].set_title('Ground Truth')
    axes[-1].set_xlim(0, 1)
    axes[-1].set_ylim(0, 1)
    axes[-1].axis('off')
    
    plt.tight_layout()
    plt.savefig(str(output_dir / 'direct_optimization.png'), dpi=150, bbox_inches='tight')
    plt.close()
    print(f"\nSaved: {output_dir / 'direct_optimization.png'}")
    
    # Final comparison
    print("\n--- Final Results ---")
    initial_chamfer = history[0]['chamfer']
    final_chamfer = history[-1]['chamfer']
    print(f"Chamfer: {initial_chamfer:.6f} -> {final_chamfer:.6f} "
          f"({100*(final_chamfer-initial_chamfer)/initial_chamfer:+.1f}%)")
    print(f"CV: {history[0]['cv']:.4f} -> {history[-1]['cv']:.4f}")


def main():
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    print(f"Device: {device}")
    print(f"geomloss available: {HAS_GEOMLOSS}")
    
    # Test 1: Lloyd iterations analysis
    best_configs = test_lloyd_iterations(device)
    
    # Test 2: Sinkhorn vs Chamfer gradients
    test_sinkhorn_vs_chamfer(device)
    
    # Test 3: Direct optimization
    test_combined_loss_optimization(device)
    
    print("\n" + "="*70)
    print("ALL TESTS COMPLETED!")
    print("="*70)
    print("\nVisualizations saved to: outputs_lloyd_analysis/")


if __name__ == '__main__':
    main()
