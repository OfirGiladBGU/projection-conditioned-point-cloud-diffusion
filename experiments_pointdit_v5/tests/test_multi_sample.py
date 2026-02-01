"""
Multi-Sample Comparison Test: Verify Sinkhorn+Chamfer results across many samples

The user is skeptical about "perfect match" results - let's test on multiple samples!
"""

import torch
import torch.nn.functional as F
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).parent.parent))

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


def optimize_with_loss(
    image: torch.Tensor,
    gt_points: torch.Tensor,
    num_points: int,
    device: str,
    method: str = 'sinkhorn_only',
    num_steps: int = 300,
    lr: float = 0.01,
):
    """Optimize points with different loss combinations."""
    # Initialize from density
    points = density_guided_init(image, num_points, device, std=0.02)
    points = points.clone().requires_grad_(True)
    
    if HAS_GEOMLOSS:
        sinkhorn = SinkhornDensityLoss(blur=0.01, grid_size=64).to(device)
    else:
        sinkhorn = SinkhornDensityLossSimple(grid_size=32, blur=0.05).to(device)
    
    chamfer_fn = ChamferLoss()
    optimizer = torch.optim.Adam([points], lr=lr)
    
    for step in range(num_steps):
        optimizer.zero_grad()
        
        if method == 'sinkhorn_only':
            loss = sinkhorn(points, image)
        elif method == 'chamfer_only':
            loss = chamfer_fn(points, gt_points)
        elif method == 'sinkhorn_chamfer':
            # Balanced combo
            loss = sinkhorn(points, image) + 10.0 * chamfer_fn(points, gt_points)
        else:
            raise ValueError(f"Unknown method: {method}")
        
        loss.backward()
        optimizer.step()
        
        with torch.no_grad():
            points.data.clamp_(-1, 1)
    
    return points.detach()


def lloyd_refinement(points: torch.Tensor, image: torch.Tensor, num_steps: int = 50):
    """Apply Lloyd's relaxation."""
    lloyd = DifferentiableLloydStep(tau=0.001, grid_size=64)
    
    for _ in range(num_steps):
        points = lloyd(points, image)
    
    return points


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
    
    output_dir = Path(__file__).parent / 'outputs_multi_sample'
    output_dir.mkdir(exist_ok=True)
    
    # Test on multiple samples
    NUM_SAMPLES = 8
    
    all_results = []
    
    print("\n" + "="*80)
    print(f"Testing on {NUM_SAMPLES} samples - comparing methods")
    print("="*80)
    
    val_iter = iter(val_loader)
    
    for sample_idx in range(NUM_SAMPLES):
        try:
            batch = next(val_iter)
        except StopIteration:
            val_iter = iter(val_loader)
            batch = next(val_iter)
        
        image = batch['image'].to(device)
        gt_points = batch['points'].to(device)
        num_points = gt_points.shape[1]
        
        print(f"\n--- Sample {sample_idx + 1}/{NUM_SAMPLES} ---")
        
        # Ground truth metrics
        gt_metrics = compute_blue_noise_metrics(gt_points)
        print(f"GT:              Mean NN={gt_metrics['mean_nn']:.4f}, CV={gt_metrics['cv']:.3f}")
        
        # 1. Density init only
        init_points = density_guided_init(image, num_points, device, std=0.02)
        init_metrics = compute_blue_noise_metrics(init_points, gt_points)
        print(f"Density Init:    Mean NN={init_metrics['mean_nn']:.4f}, CV={init_metrics['cv']:.3f}, Chamfer={init_metrics['chamfer']:.6f}")
        
        # 2. Lloyd only (from density init)
        lloyd_points = lloyd_refinement(init_points.clone(), image, num_steps=50)
        lloyd_metrics = compute_blue_noise_metrics(lloyd_points, gt_points)
        print(f"Lloyd (50):      Mean NN={lloyd_metrics['mean_nn']:.4f}, CV={lloyd_metrics['cv']:.3f}, Chamfer={lloyd_metrics['chamfer']:.6f}")
        
        # 3. Sinkhorn only
        sinkhorn_points = optimize_with_loss(image, gt_points, num_points, device, 'sinkhorn_only', num_steps=300)
        sinkhorn_metrics = compute_blue_noise_metrics(sinkhorn_points, gt_points)
        print(f"Sinkhorn Only:   Mean NN={sinkhorn_metrics['mean_nn']:.4f}, CV={sinkhorn_metrics['cv']:.3f}, Chamfer={sinkhorn_metrics['chamfer']:.6f}")
        
        # 4. Chamfer only
        chamfer_points = optimize_with_loss(image, gt_points, num_points, device, 'chamfer_only', num_steps=300)
        chamfer_metrics = compute_blue_noise_metrics(chamfer_points, gt_points)
        print(f"Chamfer Only:    Mean NN={chamfer_metrics['mean_nn']:.4f}, CV={chamfer_metrics['cv']:.3f}, Chamfer={chamfer_metrics['chamfer']:.6f}")
        
        # 5. Sinkhorn + Chamfer
        combined_points = optimize_with_loss(image, gt_points, num_points, device, 'sinkhorn_chamfer', num_steps=300)
        combined_metrics = compute_blue_noise_metrics(combined_points, gt_points)
        print(f"Sinkhorn+Chamfer: Mean NN={combined_metrics['mean_nn']:.4f}, CV={combined_metrics['cv']:.3f}, Chamfer={combined_metrics['chamfer']:.6f}")
        
        all_results.append({
            'sample_idx': sample_idx,
            'gt': gt_metrics,
            'init': init_metrics,
            'lloyd': lloyd_metrics,
            'sinkhorn': sinkhorn_metrics,
            'chamfer': chamfer_metrics,
            'combined': combined_metrics,
            'image': image.cpu(),
            'gt_points': gt_points.cpu(),
            'init_points': init_points.cpu(),
            'lloyd_points': lloyd_points.cpu(),
            'sinkhorn_points': sinkhorn_points.cpu(),
            'chamfer_points': chamfer_points.cpu(),
            'combined_points': combined_points.cpu(),
        })
    
    # Print summary statistics
    print("\n" + "="*80)
    print("SUMMARY STATISTICS ACROSS ALL SAMPLES")
    print("="*80)
    
    methods = ['gt', 'init', 'lloyd', 'sinkhorn', 'chamfer', 'combined']
    method_names = ['Ground Truth', 'Density Init', 'Lloyd (50)', 'Sinkhorn Only', 'Chamfer Only', 'Sinkhorn+Chamfer']
    
    print(f"\n{'Method':<20} {'Mean NN (avg)':<15} {'CV (avg)':<12} {'Chamfer (avg)':<15}")
    print("-"*65)
    
    for method, name in zip(methods, method_names):
        mean_nns = [r[method]['mean_nn'] for r in all_results]
        cvs = [r[method]['cv'] for r in all_results]
        if method != 'gt':
            chamfers = [r[method]['chamfer'] for r in all_results]
            print(f"{name:<20} {np.mean(mean_nns):.4f} ± {np.std(mean_nns):.4f}   {np.mean(cvs):.3f} ± {np.std(cvs):.3f}  {np.mean(chamfers):.6f}")
        else:
            print(f"{name:<20} {np.mean(mean_nns):.4f} ± {np.std(mean_nns):.4f}   {np.mean(cvs):.3f} ± {np.std(cvs):.3f}  N/A")
    
    # Create visualization grid
    print("\nGenerating visualization...")
    
    fig, axes = plt.subplots(NUM_SAMPLES, 6, figsize=(24, NUM_SAMPLES * 4))
    
    for i, result in enumerate(all_results):
        # Get grayscale image
        img = result['image'][0, 0].numpy()
        
        # Column 0: Input image + GT overlay
        axes[i, 0].imshow(img, cmap='gray', extent=[-1, 1, -1, 1], origin='lower')
        gt_pts = result['gt_points'][0].numpy()
        axes[i, 0].scatter(gt_pts[:, 0], gt_pts[:, 1], c='red', s=1, alpha=0.8)
        axes[i, 0].set_title(f"GT\nNN={result['gt']['mean_nn']:.4f}\nCV={result['gt']['cv']:.3f}")
        axes[i, 0].set_xlim(-1, 1)
        axes[i, 0].set_ylim(-1, 1)
        axes[i, 0].set_aspect('equal')
        
        # Column 1: Density Init
        axes[i, 1].imshow(img, cmap='gray', extent=[-1, 1, -1, 1], origin='lower')
        pts = result['init_points'][0].numpy()
        axes[i, 1].scatter(pts[:, 0], pts[:, 1], c='blue', s=1, alpha=0.8)
        axes[i, 1].set_title(f"Density Init\nNN={result['init']['mean_nn']:.4f}\nChamfer={result['init']['chamfer']:.6f}")
        axes[i, 1].set_xlim(-1, 1)
        axes[i, 1].set_ylim(-1, 1)
        axes[i, 1].set_aspect('equal')
        
        # Column 2: Lloyd
        axes[i, 2].imshow(img, cmap='gray', extent=[-1, 1, -1, 1], origin='lower')
        pts = result['lloyd_points'][0].numpy()
        axes[i, 2].scatter(pts[:, 0], pts[:, 1], c='green', s=1, alpha=0.8)
        axes[i, 2].set_title(f"Lloyd\nNN={result['lloyd']['mean_nn']:.4f}\nChamfer={result['lloyd']['chamfer']:.6f}")
        axes[i, 2].set_xlim(-1, 1)
        axes[i, 2].set_ylim(-1, 1)
        axes[i, 2].set_aspect('equal')
        
        # Column 3: Sinkhorn Only
        axes[i, 3].imshow(img, cmap='gray', extent=[-1, 1, -1, 1], origin='lower')
        pts = result['sinkhorn_points'][0].numpy()
        axes[i, 3].scatter(pts[:, 0], pts[:, 1], c='purple', s=1, alpha=0.8)
        axes[i, 3].set_title(f"Sinkhorn Only\nNN={result['sinkhorn']['mean_nn']:.4f}\nChamfer={result['sinkhorn']['chamfer']:.6f}")
        axes[i, 3].set_xlim(-1, 1)
        axes[i, 3].set_ylim(-1, 1)
        axes[i, 3].set_aspect('equal')
        
        # Column 4: Chamfer Only
        axes[i, 4].imshow(img, cmap='gray', extent=[-1, 1, -1, 1], origin='lower')
        pts = result['chamfer_points'][0].numpy()
        axes[i, 4].scatter(pts[:, 0], pts[:, 1], c='orange', s=1, alpha=0.8)
        axes[i, 4].set_title(f"Chamfer Only\nNN={result['chamfer']['mean_nn']:.4f}\nChamfer={result['chamfer']['chamfer']:.6f}")
        axes[i, 4].set_xlim(-1, 1)
        axes[i, 4].set_ylim(-1, 1)
        axes[i, 4].set_aspect('equal')
        
        # Column 5: Sinkhorn + Chamfer
        axes[i, 5].imshow(img, cmap='gray', extent=[-1, 1, -1, 1], origin='lower')
        pts = result['combined_points'][0].numpy()
        axes[i, 5].scatter(pts[:, 0], pts[:, 1], c='cyan', s=1, alpha=0.8)
        axes[i, 5].set_title(f"Sinkhorn+Chamfer\nNN={result['combined']['mean_nn']:.4f}\nChamfer={result['combined']['chamfer']:.6f}")
        axes[i, 5].set_xlim(-1, 1)
        axes[i, 5].set_ylim(-1, 1)
        axes[i, 5].set_aspect('equal')
        
        # Remove axis ticks for cleaner look
        for j in range(6):
            axes[i, j].set_xticks([])
            axes[i, j].set_yticks([])
    
    plt.tight_layout()
    save_path = output_dir / 'multi_sample_comparison.png'
    plt.savefig(save_path, dpi=150, bbox_inches='tight')
    plt.close()
    
    print(f"\nVisualization saved to: {save_path}")
    
    # Also create a summary bar chart
    fig, axes = plt.subplots(1, 3, figsize=(15, 5))
    
    x = np.arange(len(method_names) - 1)  # Exclude GT for some metrics
    width = 0.35
    
    # Mean NN comparison
    gt_avg_nn = np.mean([r['gt']['mean_nn'] for r in all_results])
    mean_nns = [np.mean([r[m]['mean_nn'] for r in all_results]) for m in methods[1:]]
    std_nns = [np.std([r[m]['mean_nn'] for r in all_results]) for m in methods[1:]]
    
    axes[0].bar(x, mean_nns, width, yerr=std_nns, capsize=3)
    axes[0].axhline(y=gt_avg_nn, color='r', linestyle='--', label=f'GT avg: {gt_avg_nn:.4f}')
    axes[0].set_ylabel('Mean NN Distance')
    axes[0].set_title('Mean NN Distance (higher = better spaced)')
    axes[0].set_xticks(x)
    axes[0].set_xticklabels(method_names[1:], rotation=45, ha='right')
    axes[0].legend()
    
    # CV comparison
    gt_avg_cv = np.mean([r['gt']['cv'] for r in all_results])
    cvs = [np.mean([r[m]['cv'] for r in all_results]) for m in methods[1:]]
    std_cvs = [np.std([r[m]['cv'] for r in all_results]) for m in methods[1:]]
    
    axes[1].bar(x, cvs, width, yerr=std_cvs, capsize=3)
    axes[1].axhline(y=gt_avg_cv, color='r', linestyle='--', label=f'GT avg: {gt_avg_cv:.3f}')
    axes[1].set_ylabel('Coefficient of Variation')
    axes[1].set_title('CV of NN Distance (lower = more uniform)')
    axes[1].set_xticks(x)
    axes[1].set_xticklabels(method_names[1:], rotation=45, ha='right')
    axes[1].legend()
    
    # Chamfer comparison
    chamfers = [np.mean([r[m]['chamfer'] for r in all_results]) for m in methods[1:]]
    std_chamfers = [np.std([r[m]['chamfer'] for r in all_results]) for m in methods[1:]]
    
    axes[2].bar(x, chamfers, width, yerr=std_chamfers, capsize=3)
    axes[2].set_ylabel('Chamfer Distance')
    axes[2].set_title('Chamfer Distance (lower = closer to GT)')
    axes[2].set_xticks(x)
    axes[2].set_xticklabels(method_names[1:], rotation=45, ha='right')
    
    plt.tight_layout()
    save_path = output_dir / 'summary_charts.png'
    plt.savefig(save_path, dpi=150, bbox_inches='tight')
    plt.close()
    
    print(f"Summary charts saved to: {save_path}")
    
    # Key findings
    print("\n" + "="*80)
    print("KEY FINDINGS")
    print("="*80)
    
    # Compare methods to GT
    gt_avg_nn = np.mean([r['gt']['mean_nn'] for r in all_results])
    
    for method, name in zip(methods[1:], method_names[1:]):
        avg_nn = np.mean([r[method]['mean_nn'] for r in all_results])
        avg_chamfer = np.mean([r[method]['chamfer'] for r in all_results])
        nn_ratio = avg_nn / gt_avg_nn
        print(f"\n{name}:")
        print(f"  NN ratio to GT: {nn_ratio:.3f} (1.0 = matches GT spacing)")
        print(f"  Avg Chamfer: {avg_chamfer:.6f}")


if __name__ == "__main__":
    main()
