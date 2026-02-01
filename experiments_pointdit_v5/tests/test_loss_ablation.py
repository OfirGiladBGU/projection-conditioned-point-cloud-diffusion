"""Ablation study for loss functions - tests different loss combinations.

Runs overfit tests with different loss configurations to diagnose what helps/hurts.
Also computes quantitative similarity metrics between predicted and GT.
"""

import argparse
import os
import sys
import numpy as np
import torch
import torch.nn.functional as F

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


def compute_metrics(pred_points: torch.Tensor, gt_points: torch.Tensor, 
                    image: torch.Tensor, grid_size: int = 64) -> dict:
    """Compute similarity metrics between predicted and GT point distributions.
    
    Args:
        pred_points: (B, N, 2) predicted points in [-1, 1]
        gt_points: (B, N, 2) ground truth points in [-1, 1]
        image: (B, 1, H, W) conditioning image
        grid_size: Resolution for density comparison
    
    Returns:
        Dict with various metrics
    """
    B, N, _ = pred_points.shape
    device = pred_points.device
    
    # 1. Chamfer Distance
    chamfer_loss = ChamferLoss()
    chamfer = chamfer_loss(pred_points, gt_points).item()
    
    # 2. Rasterize both point sets to density grids
    def points_to_density(points, grid_size):
        """Convert points to density grid using bilinear splatting."""
        pts_grid = (points + 1) / 2 * (grid_size - 1)
        x = pts_grid[..., 0]
        y = pts_grid[..., 1]
        
        x0 = torch.floor(x).long().clamp(0, grid_size - 1)
        y0 = torch.floor(y).long().clamp(0, grid_size - 1)
        x1 = (x0 + 1).clamp(0, grid_size - 1)
        y1 = (y0 + 1).clamp(0, grid_size - 1)
        
        wa = (x1.float() - x) * (y1.float() - y)
        wb = (x1.float() - x) * (y - y0.float())
        wc = (x - x0.float()) * (y1.float() - y)
        wd = (x - x0.float()) * (y - y0.float())
        
        batch_indices = torch.arange(B, device=device).unsqueeze(1).expand(-1, N)
        
        def scatter_to_grid(idx_x, idx_y, weights):
            flat_idx = batch_indices * (grid_size ** 2) + idx_y * grid_size + idx_x
            result = torch.zeros(B * grid_size ** 2, device=device, dtype=points.dtype)
            result = result.scatter_add_(0, flat_idx.flatten(), weights.flatten())
            return result.view(B, grid_size, grid_size)
        
        density = (
            scatter_to_grid(x0, y0, wa) + 
            scatter_to_grid(x0, y1, wb) + 
            scatter_to_grid(x1, y0, wc) + 
            scatter_to_grid(x1, y1, wd)
        )
        return density
    
    pred_density = points_to_density(pred_points, grid_size)
    gt_density = points_to_density(gt_points, grid_size)
    
    # 3. Density MSE (how well do densities match)
    density_mse = F.mse_loss(pred_density, gt_density).item()
    
    # 4. Density Correlation (Pearson correlation between density maps)
    pred_flat = pred_density.flatten()
    gt_flat = gt_density.flatten()
    pred_centered = pred_flat - pred_flat.mean()
    gt_centered = gt_flat - gt_flat.mean()
    correlation = (pred_centered * gt_centered).sum() / (
        torch.sqrt((pred_centered ** 2).sum()) * torch.sqrt((gt_centered ** 2).sum()) + 1e-6
    )
    density_correlation = correlation.item()
    
    # 5. Target image density comparison
    target_density = F.interpolate(image, size=(grid_size, grid_size), mode='bilinear', align_corners=False)
    target_density = target_density.squeeze(1)
    # IMPORTANT: Invert! Dark pixels (low intensity) = high density
    target_density = 1.0 - target_density
    # Normalize to same scale as point density
    target_sum = target_density.sum() + 1e-6
    target_density_scaled = (target_density / target_sum) * N
    
    # MSE between predicted density and target image
    pred_vs_image_mse = F.mse_loss(pred_density, target_density_scaled).item()
    gt_vs_image_mse = F.mse_loss(gt_density, target_density_scaled).item()
    
    # 6. Coverage: what fraction of high-density areas have points?
    # Threshold the image to find "should have points" regions (dark areas after inversion = high density)
    high_density_mask = target_density > target_density.mean()
    pred_in_high = (pred_density * high_density_mask.float()).sum() / (pred_density.sum() + 1e-6)
    gt_in_high = (gt_density * high_density_mask.float()).sum() / (gt_density.sum() + 1e-6)
    
    return {
        'chamfer': chamfer,
        'density_mse': density_mse,
        'density_correlation': density_correlation,
        'pred_vs_image_mse': pred_vs_image_mse,
        'gt_vs_image_mse': gt_vs_image_mse,
        'coverage_pred': pred_in_high.item(),
        'coverage_gt': gt_in_high.item(),
    }


def train_with_config(model, scheduler, image, points, device, 
                      steps, lr, loss_config: dict) -> tuple:
    """Train model with specific loss configuration.
    
    Args:
        loss_config: Dict with keys 'use_chamfer', 'use_grid_density', 
                    'use_repulsion', 'adaptive_repulsion', and weights
    
    Returns:
        (final_loss_dict, sampled_points)
    """
    # Reset model weights
    model.apply(lambda m: m.reset_parameters() if hasattr(m, 'reset_parameters') else None)
    
    # Re-initialize (simple approach - recreate)
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
    
    losses_history = []
    
    for step in range(steps):
        model.train()
        
        noise = torch.randn_like(points)
        timesteps = torch.randint(0, scheduler.num_train_timesteps, (1,), device=device, dtype=torch.long)
        x_t = scheduler.add_noise(points, noise, timesteps)
        pred_x0 = model(x_t, timesteps, image)
        
        # Compute losses based on config
        total_loss = 0.0
        loss_dict = {}
        
        if loss_config.get('use_chamfer', True):
            chamfer = chamfer_loss_fn(pred_x0, points)
            total_loss = total_loss + loss_config.get('chamfer_weight', 1.0) * chamfer
            loss_dict['chamfer'] = chamfer.item()
        
        if loss_config.get('use_grid_density', False):
            grid_density = grid_density_loss_fn(pred_x0, image)
            total_loss = total_loss + loss_config.get('grid_density_weight', 1.0) * grid_density
            loss_dict['grid_density'] = grid_density.item()
        
        if loss_config.get('use_repulsion', False):
            if loss_config.get('adaptive_repulsion', False):
                repulsion = adaptive_repulsion_loss_fn(pred_x0, image)
            else:
                repulsion = repulsion_loss_fn(pred_x0)
            total_loss = total_loss + loss_config.get('repulsion_weight', 0.3) * repulsion
            loss_dict['repulsion'] = repulsion.item()
        
        loss_dict['total'] = total_loss.item()
        losses_history.append(loss_dict)
        
        optimizer.zero_grad()
        total_loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        optimizer.step()
        
        if step % 100 == 0 or step == steps - 1:
            print(f"  Step {step}: {loss_dict}")
    
    # Sample
    model.eval()
    with torch.no_grad():
        sampled = sample(
            model, scheduler, image, points.shape[1],
            num_inference_steps=100, device=device,
            show_progress=False, init_from_density=True, init_std=0.02
        )
    
    return losses_history, sampled


def visualize_ablation(image, gt_points, results: dict, save_path: str):
    """Create comparison visualization for all configurations."""
    if not MATPLOTLIB_AVAILABLE:
        print("Matplotlib not available, skipping visualization")
        return
    
    img = image[0, 0].cpu().numpy()
    H, W = img.shape
    gt = gt_points[0].cpu().numpy()
    gt_pix = np.stack([(gt[:, 0] + 1.0) * (W / 2.0), (gt[:, 1] + 1.0) * (H / 2.0)], axis=1)
    
    n_configs = len(results)
    fig, axes = plt.subplots(2, n_configs + 1, figsize=(4 * (n_configs + 1), 8))
    
    # Top row: input image + GT
    axes[0, 0].imshow(img, cmap='gray')
    axes[0, 0].set_title('Input Image')
    axes[0, 0].axis('off')
    
    axes[1, 0].scatter(gt_pix[:, 0], gt_pix[:, 1], c='black', s=0.5, alpha=0.7)
    axes[1, 0].set_xlim(0, W)
    axes[1, 0].set_ylim(H, 0)
    axes[1, 0].set_aspect('equal')
    axes[1, 0].set_facecolor('white')
    axes[1, 0].set_title('Ground Truth')
    axes[1, 0].axis('off')
    
    # Results for each config
    for i, (config_name, data) in enumerate(results.items()):
        pred = data['points'][0].cpu().numpy()
        pred_pix = np.stack([(pred[:, 0] + 1.0) * (W / 2.0), (pred[:, 1] + 1.0) * (H / 2.0)], axis=1)
        metrics = data['metrics']
        
        # Metrics text
        metrics_text = (f"Chamfer: {metrics['chamfer']:.4f}\n"
                       f"Density Corr: {metrics['density_correlation']:.3f}\n"
                       f"Density MSE: {metrics['density_mse']:.2f}")
        
        axes[0, i + 1].text(0.5, 0.5, metrics_text, ha='center', va='center',
                           fontsize=10, transform=axes[0, i + 1].transAxes)
        axes[0, i + 1].set_title(config_name)
        axes[0, i + 1].axis('off')
        
        axes[1, i + 1].scatter(pred_pix[:, 0], pred_pix[:, 1], c='black', s=0.5, alpha=0.7)
        axes[1, i + 1].set_xlim(0, W)
        axes[1, i + 1].set_ylim(H, 0)
        axes[1, i + 1].set_aspect('equal')
        axes[1, i + 1].set_facecolor('white')
        axes[1, i + 1].axis('off')
    
    plt.tight_layout()
    plt.savefig(save_path, dpi=150, bbox_inches='tight')
    print(f"Saved ablation visualization to {save_path}")
    plt.close()


def main():
    parser = argparse.ArgumentParser(description="Loss ablation study")
    parser.add_argument("--steps", type=int, default=400, help="Training steps per config")
    parser.add_argument("--sample-index", type=int, default=0, help="Dataset sample index")
    parser.add_argument("--lr", type=float, default=2e-4, help="Learning rate")
    parser.add_argument("--seed", type=int, default=42, help="Random seed")
    args = parser.parse_args()
    
    print("=" * 70)
    print("LOSS ABLATION STUDY")
    print("=" * 70)
    
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
    
    # Create scheduler
    scheduler = DDPMScheduler(
        num_train_timesteps=config.diffusion.num_train_timesteps,
        beta_start=config.diffusion.beta_start,
        beta_end=config.diffusion.beta_end,
        beta_schedule=config.diffusion.beta_schedule,
    )
    
    # Define loss configurations to test
    loss_configs = {
        '1_Chamfer_Only': {
            'use_chamfer': True,
            'use_grid_density': False,
            'use_repulsion': False,
        },
        '2_Chamfer+FixedRepulsion': {
            'use_chamfer': True,
            'use_grid_density': False,
            'use_repulsion': True,
            'adaptive_repulsion': False,
            'repulsion_weight': 0.5,
        },
        '3_Chamfer+GridDensity': {
            'use_chamfer': True,
            'use_grid_density': True,
            'use_repulsion': False,
            'grid_density_weight': 1.0,
        },
        '4_Full_Adaptive': {
            'use_chamfer': True,
            'use_grid_density': True,
            'use_repulsion': True,
            'adaptive_repulsion': True,
            'grid_density_weight': 1.0,
            'repulsion_weight': 0.3,
        },
    }
    
    results = {}
    
    for config_name, loss_config in loss_configs.items():
        print(f"\n{'='*70}")
        print(f"Testing: {config_name}")
        print(f"Config: {loss_config}")
        print(f"{'='*70}")
        
        # Create fresh model for each test
        model = PointDiT(
            n_points=config.model.n_points,
            dim=config.model.dim,
            n_layers=config.model.n_layers,
            n_heads=config.model.n_heads,
            image_size=config.model.image_size,
            dropout=0.0,
        ).to(device)
        
        losses_history, sampled_points = train_with_config(
            model, scheduler, image, points, device,
            args.steps, args.lr, loss_config
        )
        
        # Compute metrics
        metrics = compute_metrics(sampled_points, points, image)
        
        print(f"\nMetrics for {config_name}:")
        for k, v in metrics.items():
            print(f"  {k}: {v:.4f}")
        
        results[config_name] = {
            'points': sampled_points,
            'metrics': metrics,
            'losses': losses_history,
        }
    
    # Summary comparison
    print("\n" + "=" * 70)
    print("SUMMARY COMPARISON")
    print("=" * 70)
    print(f"{'Config':<30} {'Chamfer':>10} {'DensityCorr':>12} {'DensityMSE':>12}")
    print("-" * 70)
    for config_name, data in results.items():
        m = data['metrics']
        print(f"{config_name:<30} {m['chamfer']:>10.4f} {m['density_correlation']:>12.3f} {m['density_mse']:>12.2f}")
    
    # Save visualization
    output_dir = os.path.join(os.path.dirname(__file__), "outputs_pointdit_v5", f"ablation_sample_{args.sample_index}")
    os.makedirs(output_dir, exist_ok=True)
    
    visualize_ablation(image, points, results, os.path.join(output_dir, "ablation_comparison.png"))
    
    # Save individual results
    for config_name, data in results.items():
        pred = data['points'][0].cpu().numpy()
        np.save(os.path.join(output_dir, f"{config_name}_points.npy"), pred)
    
    print(f"\n✓ Results saved to: {output_dir}/")


if __name__ == "__main__":
    main()
