"""Debug script to understand why GridDensityLoss hurts performance.

Key questions:
1. Are the images already inverted (dark = low intensity)?
2. What are the actual loss scales?
3. Is the gradient helpful or harmful?
"""

import sys
import numpy as np
import torch
import torch.nn.functional as F

sys.stdout.reconfigure(line_buffering=True) if hasattr(sys.stdout, 'reconfigure') else None

try:
    import matplotlib.pyplot as plt
    MATPLOTLIB_AVAILABLE = True
except:
    plt = None
    MATPLOTLIB_AVAILABLE = False

from config import Config
from dataset import SimpleImageDataset


def analyze_image_data():
    """Analyze the input images to understand intensity encoding."""
    config = Config.default()
    dataset = SimpleImageDataset(
        source_dir=config.data.source_dir,
        target_dir=config.data.target_dir,
        image_size=config.data.image_size,
        num_points=config.model.n_points,
    )
    
    sample = dataset[0]
    image = sample["image"].numpy()  # (1, H, W)
    points = sample["points"].numpy()  # (N, 2)
    
    print("=" * 60)
    print("IMAGE ANALYSIS")
    print("=" * 60)
    print(f"Image shape: {image.shape}")
    print(f"Image min: {image.min():.4f}, max: {image.max():.4f}")
    print(f"Image mean: {image.mean():.4f}")
    print(f"Points shape: {points.shape}")
    print(f"Points min: {points.min():.4f}, max: {points.max():.4f}")
    
    # Check where points are concentrated
    # If stippling convention is correct: points should be in DARK areas
    
    # Sample image values at point locations
    img = image[0]  # (H, W)
    H, W = img.shape
    
    # Convert points from [-1,1] to pixel coords
    px = ((points[:, 0] + 1) / 2 * (W - 1)).astype(int).clip(0, W-1)
    py = ((points[:, 1] + 1) / 2 * (H - 1)).astype(int).clip(0, H-1)
    
    # Get image intensities at point locations
    intensities_at_points = img[py, px]
    
    print(f"\nIntensities at point locations:")
    print(f"  Mean: {intensities_at_points.mean():.4f}")
    print(f"  Std:  {intensities_at_points.std():.4f}")
    print(f"  Min:  {intensities_at_points.min():.4f}")
    print(f"  Max:  {intensities_at_points.max():.4f}")
    
    # Compare with overall image
    print(f"\nOverall image intensities:")
    print(f"  Mean: {img.mean():.4f}")
    
    # If points are in dark areas, mean intensity at points should be LOWER than overall mean
    if intensities_at_points.mean() < img.mean():
        print("\n✓ GOOD: Points are concentrated in DARKER areas (lower intensity)")
        print("  This means: dark pixels -> more points (standard stippling)")
        print("  Our GridDensityLoss should use: target = 1 - image (INVERT)")
    else:
        print("\n⚠ WARNING: Points are concentrated in LIGHTER areas (higher intensity)")
        print("  This means: bright pixels -> more points (INVERTED stippling)")
        print("  Our GridDensityLoss should use: target = image (NO INVERT)")
    
    # Histogram analysis
    print("\n" + "=" * 60)
    print("HISTOGRAM ANALYSIS")
    print("=" * 60)
    
    # Bin the image into intensity buckets
    bins = np.linspace(0, 1, 11)  # 10 bins
    img_hist, _ = np.histogram(img.flatten(), bins=bins, density=True)
    points_hist, _ = np.histogram(intensities_at_points, bins=bins, density=True)
    
    print(f"\n{'Bin Range':<15} {'Image %':<12} {'Points %':<12} {'Ratio (P/I)':<12}")
    print("-" * 52)
    for i in range(len(bins) - 1):
        img_pct = img_hist[i] / (img_hist.sum() + 1e-6) * 100
        pts_pct = points_hist[i] / (points_hist.sum() + 1e-6) * 100
        ratio = pts_pct / (img_pct + 1e-6)
        print(f"[{bins[i]:.1f}, {bins[i+1]:.1f}]     {img_pct:>8.1f}%    {pts_pct:>8.1f}%    {ratio:>8.2f}x")
    
    # If ratio > 1 for dark bins and < 1 for bright bins -> standard stippling
    # If ratio < 1 for dark bins and > 1 for bright bins -> inverted stippling
    
    dark_bin_ratio = points_hist[0] / (img_hist[0] + 1e-6)
    bright_bin_ratio = points_hist[-1] / (img_hist[-1] + 1e-6)
    
    print(f"\nDark bin (0.0-0.1) ratio: {dark_bin_ratio:.2f}x")
    print(f"Bright bin (0.9-1.0) ratio: {bright_bin_ratio:.2f}x")
    
    if dark_bin_ratio > bright_bin_ratio:
        print("\n✓ CONFIRMS: Dark areas have more points -> standard stippling")
    else:
        print("\n⚠ CONFIRMS: Bright areas have more points -> INVERTED stippling")
    
    # Save visualization if matplotlib available
    if MATPLOTLIB_AVAILABLE:
        import os
        output_dir = os.path.join(os.path.dirname(__file__), "outputs_pointdit_v5", "debug_data")
        os.makedirs(output_dir, exist_ok=True)
        
        fig, axes = plt.subplots(1, 3, figsize=(15, 5))
        
        # Image
        axes[0].imshow(img, cmap='gray')
        axes[0].set_title(f'Input Image\nMean intensity: {img.mean():.3f}')
        axes[0].axis('off')
        
        # Points overlaid
        axes[1].imshow(img, cmap='gray')
        axes[1].scatter(px, py, c='red', s=0.3, alpha=0.3)
        axes[1].set_title(f'Points on Image\nMean intensity at points: {intensities_at_points.mean():.3f}')
        axes[1].axis('off')
        
        # Histograms
        bin_centers = (bins[:-1] + bins[1:]) / 2
        width = 0.035
        axes[2].bar(bin_centers - width/2, img_hist / img_hist.sum(), width=width, label='Image', alpha=0.7)
        axes[2].bar(bin_centers + width/2, points_hist / points_hist.sum(), width=width, label='Points', alpha=0.7)
        axes[2].set_xlabel('Intensity')
        axes[2].set_ylabel('Density')
        axes[2].set_title('Intensity Distribution')
        axes[2].legend()
        
        plt.tight_layout()
        plt.savefig(os.path.join(output_dir, "data_analysis.png"), dpi=150)
        print(f"\nSaved visualization to {output_dir}/data_analysis.png")
        plt.close()
    
    return image, points


def analyze_loss_scales():
    """Check the actual loss magnitudes."""
    print("\n" + "=" * 60)
    print("LOSS SCALE ANALYSIS")
    print("=" * 60)
    
    from diffusion import ChamferLoss, RepulsionLoss, GridDensityLoss
    
    config = Config.default()
    device = "cuda" if torch.cuda.is_available() else "cpu"
    
    dataset = SimpleImageDataset(
        source_dir=config.data.source_dir,
        target_dir=config.data.target_dir,
        image_size=config.data.image_size,
        num_points=config.model.n_points,
    )
    
    sample = dataset[0]
    image = sample["image"].unsqueeze(0).to(device)  # (1, 1, H, W)
    points = sample["points"].unsqueeze(0).to(device)  # (1, N, 2)
    
    # Create random noise points (bad prediction)
    random_points = torch.rand_like(points) * 2 - 1  # Random in [-1, 1]
    
    chamfer_fn = ChamferLoss()
    repulsion_fn = RepulsionLoss(repulsion_radius=0.02)
    grid_density_fn = GridDensityLoss(grid_sizes=(32, 64))
    
    # Compute losses for GT (should be low)
    chamfer_gt = chamfer_fn(points, points)
    repulsion_gt = repulsion_fn(points)
    density_gt = grid_density_fn(points, image)
    
    # Compute losses for random (should be high)
    chamfer_rand = chamfer_fn(random_points, points)
    repulsion_rand = repulsion_fn(random_points)
    density_rand = grid_density_fn(random_points, image)
    
    print(f"\n{'Loss':<20} {'GT -> GT':<15} {'Random -> GT':<15} {'Ratio':<10}")
    print("-" * 60)
    print(f"{'Chamfer':<20} {chamfer_gt.item():<15.6f} {chamfer_rand.item():<15.6f} {chamfer_rand.item() / (chamfer_gt.item() + 1e-8):<10.1f}")
    print(f"{'Repulsion':<20} {repulsion_gt.item():<15.6f} {repulsion_rand.item():<15.6f} {repulsion_rand.item() / (repulsion_gt.item() + 1e-8):<10.1f}")
    print(f"{'GridDensity':<20} {density_gt.item():<15.6f} {density_rand.item():<15.6f} {density_rand.item() / (density_gt.item() + 1e-8):<10.1f}")
    
    print("\nKey insight: The ratio shows how much the loss increases for bad predictions.")
    print("If GridDensity has a high value even for GT, the loss may be poorly calibrated.")
    
    # Analyze gradient direction
    print("\n" + "=" * 60)
    print("GRADIENT ANALYSIS")
    print("=" * 60)
    
    # Perturb a single point and see how losses change
    test_points = points.clone().requires_grad_(True)
    
    # Move a subset of points toward a dark region
    # First find dark and light regions
    img = image[0, 0].cpu().numpy()
    dark_mask = img < 0.3
    light_mask = img > 0.7
    
    dark_y, dark_x = np.where(dark_mask)
    light_y, light_x = np.where(light_mask)
    
    if len(dark_x) > 0 and len(light_x) > 0:
        # Pick a random dark and light pixel
        dark_coord = np.array([dark_x[len(dark_x)//2], dark_y[len(dark_y)//2]])
        light_coord = np.array([light_x[len(light_x)//2], light_y[len(light_y)//2]])
        
        # Convert to [-1, 1]
        H, W = img.shape
        dark_norm = (dark_coord / np.array([W-1, H-1])) * 2 - 1
        light_norm = (light_coord / np.array([W-1, H-1])) * 2 - 1
        
        print(f"Dark region coord (normalized): {dark_norm}")
        print(f"Light region coord (normalized): {light_norm}")
        
        # Create test: put all points at dark location
        all_dark = torch.tensor(dark_norm, dtype=torch.float32, device=device).unsqueeze(0).unsqueeze(0).expand(1, points.shape[1], 2)
        all_light = torch.tensor(light_norm, dtype=torch.float32, device=device).unsqueeze(0).unsqueeze(0).expand(1, points.shape[1], 2)
        
        density_dark = grid_density_fn(all_dark, image)
        density_light = grid_density_fn(all_light, image)
        
        print(f"\nGridDensity loss when ALL points at dark pixel: {density_dark.item():.4f}")
        print(f"GridDensity loss when ALL points at light pixel: {density_light.item():.4f}")
        
        if density_dark < density_light:
            print("✓ GOOD: Loss is lower when points are in dark region (as expected)")
        else:
            print("⚠ BAD: Loss is lower when points are in LIGHT region - inversion is WRONG!")


if __name__ == "__main__":
    analyze_image_data()
    analyze_loss_scales()
