"""Comprehensive evaluation with proper similarity metrics.

Tests different loss combinations and measures:
- Nearest Neighbor (NN) mean and std
- Coverage metrics
- Histogram comparison (Earth Mover's Distance / Wasserstein)
- Density correlation
- Chamfer distance variants
"""

import argparse
import os
import sys
import numpy as np
import torch
import torch.nn.functional as F
from scipy.stats import wasserstein_distance
from scipy.spatial.distance import cdist

sys.stdout.reconfigure(line_buffering=True) if hasattr(sys.stdout, 'reconfigure') else None

try:
    import matplotlib.pyplot as plt
    MATPLOTLIB_AVAILABLE = True
except Exception:
    plt = None
    MATPLOTLIB_AVAILABLE = False

from config import Config
from dataset import SimpleImageDataset
from diffusion import (
    DDPMScheduler, sample,
    ChamferLoss, RepulsionLoss, GridDensityLoss, AdaptiveRepulsionLoss
)
from model import PointDiT


def compute_nn_metrics(pred_points: np.ndarray, gt_points: np.ndarray) -> dict:
    """Compute Nearest Neighbor based metrics.
    
    Args:
        pred_points: (N, 2) predicted points
        gt_points: (M, 2) ground truth points
    
    Returns:
        Dict with NN metrics
    """
    # Compute pairwise distances
    dist_pred_to_gt = cdist(pred_points, gt_points, 'euclidean')  # (N, M)
    dist_gt_to_pred = cdist(gt_points, pred_points, 'euclidean')  # (M, N)
    
    # For each predicted point, find distance to nearest GT point
    nn_pred_to_gt = np.min(dist_pred_to_gt, axis=1)  # (N,)
    
    # For each GT point, find distance to nearest predicted point
    nn_gt_to_pred = np.min(dist_gt_to_pred, axis=1)  # (M,)
    
    return {
        # Pred -> GT (how well predictions cover GT)
        'nn_pred_to_gt_mean': float(np.mean(nn_pred_to_gt)),
        'nn_pred_to_gt_std': float(np.std(nn_pred_to_gt)),
        'nn_pred_to_gt_median': float(np.median(nn_pred_to_gt)),
        'nn_pred_to_gt_max': float(np.max(nn_pred_to_gt)),
        
        # GT -> Pred (how well GT is covered by predictions)
        'nn_gt_to_pred_mean': float(np.mean(nn_gt_to_pred)),
        'nn_gt_to_pred_std': float(np.std(nn_gt_to_pred)),
        'nn_gt_to_pred_median': float(np.median(nn_gt_to_pred)),
        'nn_gt_to_pred_max': float(np.max(nn_gt_to_pred)),
        
        # Symmetric metrics
        'nn_symmetric_mean': float((np.mean(nn_pred_to_gt) + np.mean(nn_gt_to_pred)) / 2),
        'chamfer_distance': float(np.mean(nn_pred_to_gt**2) + np.mean(nn_gt_to_pred**2)),
    }


def compute_coverage_metrics(pred_points: np.ndarray, gt_points: np.ndarray, 
                             threshold: float = 0.02) -> dict:
    """Compute coverage metrics.
    
    Args:
        pred_points: (N, 2) predicted points in [-1, 1]
        gt_points: (M, 2) ground truth points in [-1, 1]
        threshold: Distance threshold for "coverage"
    
    Returns:
        Dict with coverage metrics
    """
    dist_gt_to_pred = cdist(gt_points, pred_points, 'euclidean')
    nn_gt_to_pred = np.min(dist_gt_to_pred, axis=1)
    
    dist_pred_to_gt = cdist(pred_points, gt_points, 'euclidean')
    nn_pred_to_gt = np.min(dist_pred_to_gt, axis=1)
    
    # Coverage: fraction of GT points that have a prediction within threshold
    coverage = float(np.mean(nn_gt_to_pred < threshold))
    
    # Precision: fraction of predicted points that are near a GT point
    precision = float(np.mean(nn_pred_to_gt < threshold))
    
    # F1 score
    if coverage + precision > 0:
        f1 = 2 * coverage * precision / (coverage + precision)
    else:
        f1 = 0.0
    
    return {
        'coverage': coverage,
        'precision': precision,
        'f1_score': f1,
    }


def compute_histogram_metrics(pred_points: np.ndarray, gt_points: np.ndarray,
                              bins: int = 32) -> dict:
    """Compute histogram-based metrics (distribution comparison).
    
    Args:
        pred_points: (N, 2) predicted points in [-1, 1]
        gt_points: (M, 2) ground truth points in [-1, 1]
        bins: Number of bins for histogram
    
    Returns:
        Dict with histogram metrics
    """
    # Create 2D histograms
    range_bounds = [[-1, 1], [-1, 1]]
    
    pred_hist, _, _ = np.histogram2d(pred_points[:, 0], pred_points[:, 1], 
                                      bins=bins, range=range_bounds, density=True)
    gt_hist, _, _ = np.histogram2d(gt_points[:, 0], gt_points[:, 1], 
                                    bins=bins, range=range_bounds, density=True)
    
    # Normalize histograms
    pred_hist = pred_hist / (pred_hist.sum() + 1e-6)
    gt_hist = gt_hist / (gt_hist.sum() + 1e-6)
    
    # Flatten for comparison
    pred_flat = pred_hist.flatten()
    gt_flat = gt_hist.flatten()
    
    # Earth Mover's Distance (1D approximation using flattened histograms)
    emd = wasserstein_distance(pred_flat, gt_flat)
    
    # Histogram intersection (higher = better match)
    hist_intersection = float(np.minimum(pred_flat, gt_flat).sum())
    
    # Chi-square distance
    chi_sq = float(np.sum((pred_flat - gt_flat)**2 / (pred_flat + gt_flat + 1e-6)) / 2)
    
    # Correlation
    pred_centered = pred_flat - pred_flat.mean()
    gt_centered = gt_flat - gt_flat.mean()
    correlation = float(np.sum(pred_centered * gt_centered) / 
                       (np.sqrt(np.sum(pred_centered**2)) * np.sqrt(np.sum(gt_centered**2)) + 1e-6))
    
    # MSE between histograms
    hist_mse = float(np.mean((pred_flat - gt_flat)**2))
    
    return {
        'hist_emd': emd,
        'hist_intersection': hist_intersection,
        'hist_chi_square': chi_sq,
        'hist_correlation': correlation,
        'hist_mse': hist_mse,
    }


def compute_spacing_metrics(points: np.ndarray) -> dict:
    """Compute point spacing metrics (blue noise quality).
    
    Args:
        points: (N, 2) points in [-1, 1]
    
    Returns:
        Dict with spacing metrics
    """
    # Compute all pairwise distances
    dist_matrix = cdist(points, points, 'euclidean')
    np.fill_diagonal(dist_matrix, np.inf)
    
    # Nearest neighbor distances
    nn_distances = np.min(dist_matrix, axis=1)
    
    return {
        'spacing_mean': float(np.mean(nn_distances)),
        'spacing_std': float(np.std(nn_distances)),
        'spacing_min': float(np.min(nn_distances)),
        'spacing_max': float(np.max(nn_distances)),
        'spacing_cv': float(np.std(nn_distances) / (np.mean(nn_distances) + 1e-6)),  # Coefficient of variation
    }


def compute_all_metrics(pred_points: torch.Tensor, gt_points: torch.Tensor) -> dict:
    """Compute all similarity metrics between predicted and GT points."""
    pred_np = pred_points[0].cpu().numpy()  # Remove batch dim
    gt_np = gt_points[0].cpu().numpy()
    
    metrics = {}
    
    # NN metrics
    nn_metrics = compute_nn_metrics(pred_np, gt_np)
    metrics.update(nn_metrics)
    
    # Coverage metrics
    coverage_metrics = compute_coverage_metrics(pred_np, gt_np, threshold=0.02)
    metrics.update(coverage_metrics)
    
    # Histogram metrics
    hist_metrics = compute_histogram_metrics(pred_np, gt_np, bins=32)
    metrics.update(hist_metrics)
    
    # Spacing metrics for both
    pred_spacing = compute_spacing_metrics(pred_np)
    gt_spacing = compute_spacing_metrics(gt_np)
    
    metrics['pred_spacing_mean'] = pred_spacing['spacing_mean']
    metrics['pred_spacing_std'] = pred_spacing['spacing_std']
    metrics['gt_spacing_mean'] = gt_spacing['spacing_mean']
    metrics['gt_spacing_std'] = gt_spacing['spacing_std']
    
    # Spacing similarity (how similar are the spacing statistics)
    metrics['spacing_mean_diff'] = abs(pred_spacing['spacing_mean'] - gt_spacing['spacing_mean'])
    metrics['spacing_std_diff'] = abs(pred_spacing['spacing_std'] - gt_spacing['spacing_std'])
    
    return metrics


def train_with_config(model, scheduler, image, points, device, 
                      steps, lr, loss_config: dict) -> tuple:
    """Train model with specific loss configuration."""
    # Re-initialize model weights
    for module in model.modules():
        if isinstance(module, torch.nn.Linear):
            torch.nn.init.xavier_uniform_(module.weight, gain=0.5)
            if module.bias is not None:
                torch.nn.init.zeros_(module.bias)
        elif isinstance(module, torch.nn.Conv2d):
            torch.nn.init.kaiming_normal_(module.weight, mode="fan_out", nonlinearity="relu")
    
    optimizer = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=1e-5)
    
    # Loss functions
    chamfer_loss_fn = ChamferLoss()
    grid_density_loss_fn = GridDensityLoss(grid_sizes=(32, 64))
    repulsion_loss_fn = RepulsionLoss(repulsion_radius=0.02)
    adaptive_repulsion_loss_fn = AdaptiveRepulsionLoss(base_radius=0.02)
    
    print(f"    Training for {steps} steps...")
    
    for step in range(steps):
        model.train()
        
        noise = torch.randn_like(points)
        timesteps = torch.randint(0, scheduler.num_train_timesteps, (1,), device=device, dtype=torch.long)
        x_t = scheduler.add_noise(points, noise, timesteps)
        pred_x0 = model(x_t, timesteps, image)
        
        # Compute losses based on config
        total_loss = 0.0
        
        if loss_config.get('use_chamfer', True):
            chamfer = chamfer_loss_fn(pred_x0, points)
            total_loss = total_loss + loss_config.get('chamfer_weight', 1.0) * chamfer
        
        if loss_config.get('use_grid_density', False):
            grid_density = grid_density_loss_fn(pred_x0, image)
            total_loss = total_loss + loss_config.get('grid_density_weight', 0.1) * grid_density
        
        if loss_config.get('use_repulsion', False):
            if loss_config.get('adaptive_repulsion', False):
                repulsion = adaptive_repulsion_loss_fn(pred_x0, image)
            else:
                repulsion = repulsion_loss_fn(pred_x0)
            total_loss = total_loss + loss_config.get('repulsion_weight', 0.3) * repulsion
        
        optimizer.zero_grad()
        total_loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        optimizer.step()
        
        if step % 100 == 0:
            print(f"      Step {step}: loss={total_loss.item():.4f}")
    
    # Sample
    model.eval()
    with torch.no_grad():
        sampled = sample(
            model, scheduler, image, points.shape[1],
            num_inference_steps=100, device=device,
            show_progress=False, init_from_density=True, init_std=0.02
        )
    
    return sampled


def visualize_results(image, gt_points, results: dict, save_path: str):
    """Create comprehensive visualization."""
    if not MATPLOTLIB_AVAILABLE:
        print("Matplotlib not available")
        return
    
    img = image[0, 0].cpu().numpy()
    H, W = img.shape
    gt = gt_points[0].cpu().numpy()
    gt_pix = np.stack([(gt[:, 0] + 1.0) * (W / 2.0), (gt[:, 1] + 1.0) * (H / 2.0)], axis=1)
    
    n_configs = len(results)
    fig, axes = plt.subplots(3, n_configs + 1, figsize=(4 * (n_configs + 1), 12))
    
    # Row 0: Input and GT
    axes[0, 0].imshow(img, cmap='gray')
    axes[0, 0].set_title('Input Image', fontsize=10)
    axes[0, 0].axis('off')
    
    axes[1, 0].scatter(gt_pix[:, 0], gt_pix[:, 1], c='black', s=0.3, alpha=0.7)
    axes[1, 0].set_xlim(0, W)
    axes[1, 0].set_ylim(H, 0)
    axes[1, 0].set_aspect('equal')
    axes[1, 0].set_facecolor('white')
    axes[1, 0].set_title('Ground Truth', fontsize=10)
    axes[1, 0].axis('off')
    
    # GT histogram
    gt_hist, _, _ = np.histogram2d(gt[:, 0], gt[:, 1], bins=32, range=[[-1, 1], [-1, 1]])
    axes[2, 0].imshow(gt_hist.T, cmap='hot', origin='lower', extent=[-1, 1, -1, 1])
    axes[2, 0].set_title('GT Density', fontsize=10)
    axes[2, 0].axis('off')
    
    # Results for each config
    for i, (config_name, data) in enumerate(results.items()):
        pred = data['points'][0].cpu().numpy()
        pred_pix = np.stack([(pred[:, 0] + 1.0) * (W / 2.0), (pred[:, 1] + 1.0) * (H / 2.0)], axis=1)
        m = data['metrics']
        
        # Key metrics text
        metrics_text = (
            f"NN Mean: {m['nn_symmetric_mean']:.4f}\n"
            f"Hist Corr: {m['hist_correlation']:.3f}\n"
            f"F1: {m['f1_score']:.3f}\n"
            f"Chamfer: {m['chamfer_distance']:.4f}"
        )
        
        axes[0, i + 1].text(0.5, 0.5, metrics_text, ha='center', va='center',
                           fontsize=9, transform=axes[0, i + 1].transAxes,
                           family='monospace')
        axes[0, i + 1].set_title(config_name, fontsize=8)
        axes[0, i + 1].axis('off')
        
        # Points
        axes[1, i + 1].scatter(pred_pix[:, 0], pred_pix[:, 1], c='black', s=0.3, alpha=0.7)
        axes[1, i + 1].set_xlim(0, W)
        axes[1, i + 1].set_ylim(H, 0)
        axes[1, i + 1].set_aspect('equal')
        axes[1, i + 1].set_facecolor('white')
        axes[1, i + 1].axis('off')
        
        # Histogram
        pred_hist, _, _ = np.histogram2d(pred[:, 0], pred[:, 1], bins=32, range=[[-1, 1], [-1, 1]])
        axes[2, i + 1].imshow(pred_hist.T, cmap='hot', origin='lower', extent=[-1, 1, -1, 1])
        axes[2, i + 1].axis('off')
    
    plt.tight_layout()
    plt.savefig(save_path, dpi=150, bbox_inches='tight')
    print(f"Saved visualization to {save_path}")
    plt.close()


def main():
    parser = argparse.ArgumentParser(description="Comprehensive loss evaluation")
    parser.add_argument("--steps", type=int, default=500, help="Training steps per config")
    parser.add_argument("--sample-index", type=int, default=0, help="Dataset sample index")
    parser.add_argument("--lr", type=float, default=2e-4, help="Learning rate")
    parser.add_argument("--seed", type=int, default=42, help="Random seed")
    args = parser.parse_args()
    
    print("=" * 80)
    print("COMPREHENSIVE LOSS EVALUATION")
    print("=" * 80)
    
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    torch.cuda.manual_seed_all(args.seed)
    
    config = Config.default()
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Device: {device}")
    
    # Load data
    dataset = SimpleImageDataset(
        source_dir=config.data.source_dir,
        target_dir=config.data.target_dir,
        image_size=config.data.image_size,
        num_points=config.model.n_points,
    )
    
    sample_data = dataset[args.sample_index]
    image = sample_data["image"].unsqueeze(0).to(device)
    points = sample_data["points"].unsqueeze(0).to(device)
    
    print(f"Sample {args.sample_index}: image {image.shape}, points {points.shape}")
    
    # Compute GT self-metrics for reference
    gt_np = points[0].cpu().numpy()
    gt_spacing = compute_spacing_metrics(gt_np)
    print(f"\nGT Spacing Stats: mean={gt_spacing['spacing_mean']:.4f}, std={gt_spacing['spacing_std']:.4f}")
    
    scheduler = DDPMScheduler(
        num_train_timesteps=config.diffusion.num_train_timesteps,
        beta_start=config.diffusion.beta_start,
        beta_end=config.diffusion.beta_end,
        beta_schedule=config.diffusion.beta_schedule,
    )
    
    # Define loss configurations to test
    loss_configs = {
        '1_Chamfer': {
            'use_chamfer': True, 'chamfer_weight': 1.0,
            'use_grid_density': False,
            'use_repulsion': False,
        },
        '2_Chamfer+Rep0.3': {
            'use_chamfer': True, 'chamfer_weight': 1.0,
            'use_grid_density': False,
            'use_repulsion': True, 'adaptive_repulsion': False, 'repulsion_weight': 0.3,
        },
        '3_Chamfer+Rep0.5': {
            'use_chamfer': True, 'chamfer_weight': 1.0,
            'use_grid_density': False,
            'use_repulsion': True, 'adaptive_repulsion': False, 'repulsion_weight': 0.5,
        },
        '4_Chamfer+Rep1.0': {
            'use_chamfer': True, 'chamfer_weight': 1.0,
            'use_grid_density': False,
            'use_repulsion': True, 'adaptive_repulsion': False, 'repulsion_weight': 1.0,
        },
    }
    
    results = {}
    
    for config_name, loss_config in loss_configs.items():
        print(f"\n{'='*80}")
        print(f"Testing: {config_name}")
        print(f"{'='*80}")
        
        # Create fresh model
        model = PointDiT(
            n_points=config.model.n_points,
            dim=config.model.dim,
            n_layers=config.model.n_layers,
            n_heads=config.model.n_heads,
            image_size=config.model.image_size,
            dropout=0.0,
        ).to(device)
        
        sampled_points = train_with_config(
            model, scheduler, image, points, device,
            args.steps, args.lr, loss_config
        )
        
        # Compute all metrics
        metrics = compute_all_metrics(sampled_points, points)
        
        results[config_name] = {
            'points': sampled_points,
            'metrics': metrics,
        }
        
        print(f"\n    Key Metrics:")
        print(f"      NN Symmetric Mean: {metrics['nn_symmetric_mean']:.4f}")
        print(f"      Chamfer Distance:  {metrics['chamfer_distance']:.4f}")
        print(f"      Histogram Corr:    {metrics['hist_correlation']:.3f}")
        print(f"      F1 Score:          {metrics['f1_score']:.3f}")
        print(f"      Coverage:          {metrics['coverage']:.3f}")
    
    # Summary comparison - sort by best metrics
    print("\n" + "=" * 80)
    print("SUMMARY COMPARISON (sorted by NN Mean - lower is better)")
    print("=" * 80)
    
    # Sort by nn_symmetric_mean (lower is better)
    sorted_results = sorted(results.items(), key=lambda x: x[1]['metrics']['nn_symmetric_mean'])
    
    print(f"\n{'Config':<25} {'NN Mean':>10} {'Chamfer':>10} {'HistCorr':>10} {'F1':>8} {'Coverage':>10}")
    print("-" * 80)
    for config_name, data in sorted_results:
        m = data['metrics']
        print(f"{config_name:<25} {m['nn_symmetric_mean']:>10.4f} {m['chamfer_distance']:>10.4f} "
              f"{m['hist_correlation']:>10.3f} {m['f1_score']:>8.3f} {m['coverage']:>10.3f}")
    
    # Find best config
    best_config = sorted_results[0][0]
    print(f"\n🏆 BEST CONFIG: {best_config}")
    
    # Save results
    output_dir = os.path.join(os.path.dirname(__file__), "outputs_pointdit_v5", f"comprehensive_eval_sample_{args.sample_index}")
    os.makedirs(output_dir, exist_ok=True)
    
    visualize_results(image, points, results, os.path.join(output_dir, "comparison.png"))
    
    # Save detailed metrics to text file
    with open(os.path.join(output_dir, "metrics.txt"), 'w') as f:
        f.write("COMPREHENSIVE EVALUATION RESULTS\n")
        f.write("=" * 80 + "\n\n")
        
        for config_name, data in sorted_results:
            f.write(f"\n{config_name}\n")
            f.write("-" * 40 + "\n")
            for k, v in sorted(data['metrics'].items()):
                f.write(f"  {k}: {v:.6f}\n")
    
    # Save points
    for config_name, data in results.items():
        pred = data['points'][0].cpu().numpy()
        np.save(os.path.join(output_dir, f"{config_name}_points.npy"), pred)
    
    print(f"\n✓ Results saved to: {output_dir}/")


if __name__ == "__main__":
    main()
