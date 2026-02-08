"""
Stippling Quality Metrics for Point-DiT V6

Two new evaluation metrics inspired by CCVT and Weighted Voronoi Stippling:

1. **Capacity Fulfillment Metric**: 
   Measures how well points fulfill the "capacity" required by the grayscale image.
   - Dark pixels = need more points (high capacity)
   - Light pixels = need fewer points (low capacity)
   - Visualizes fulfilled (blue) vs unfulfilled (red) points

2. **Density Heatmap**:
   Creates a grid at H/2 x W/2 resolution showing point counts per cell.
   Useful for seeing local clustering or gaps.

Usage:
    from stippling_metrics import compute_stippling_metrics, visualize_6panel
    
    metrics_gt, metrics_pred = compute_stippling_metrics(
        gt_points, pred_points, image, grid_size=(256, 256)
    )
    
    visualize_6panel(image, gt_points, pred_points, metrics_gt, metrics_pred, save_path)
"""

import torch
import torch.nn.functional as F
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.colors import Normalize
from typing import Tuple, Dict, Optional
from pathlib import Path


def points_to_pixel_coords(
    points: torch.Tensor, 
    H: int, 
    W: int
) -> torch.Tensor:
    """
    Convert points from [-1, 1] normalized coords to pixel coords.
    
    Args:
        points: (B, N, 2) or (N, 2) points in [-1, 1]
        H, W: Image dimensions
        
    Returns:
        points_pix: Same shape, in [0, W-1] x [0, H-1] pixel coordinates
    """
    # Points are (x, y) in [-1, 1], convert to pixel coords
    # x -> col, y -> row
    points_pix = (points + 1.0) / 2.0  # [0, 1]
    points_pix = points_pix * torch.tensor([W - 1, H - 1], device=points.device, dtype=points.dtype)
    return points_pix


def sample_image_at_points(
    image: torch.Tensor, 
    points: torch.Tensor
) -> torch.Tensor:
    """
    Sample image intensity at point locations using bilinear interpolation.
    
    Args:
        image: (B, 1, H, W) or (1, H, W) grayscale image in [0, 1]
        points: (B, N, 2) or (N, 2) points in [-1, 1]
        
    Returns:
        intensities: (B, N) or (N,) intensity values at each point
    """
    # Ensure batch dimensions
    if image.dim() == 3:
        image = image.unsqueeze(0)
    if points.dim() == 2:
        points = points.unsqueeze(0)
    
    B, N, _ = points.shape
    
    # grid_sample expects grid in [-1, 1], shape (B, H_out, W_out, 2)
    # We have (B, N, 2), so reshape to (B, 1, N, 2) for grid_sample
    grid = points.unsqueeze(1)  # (B, 1, N, 2)
    
    # Sample using bilinear interpolation
    # Note: grid_sample expects (y, x) order, but points are (x, y)
    # Swap to (y, x) for grid_sample
    grid_yx = torch.stack([grid[..., 1], grid[..., 0]], dim=-1)
    
    sampled = F.grid_sample(
        image, 
        grid_yx, 
        mode='bilinear', 
        padding_mode='border', 
        align_corners=True
    )  # (B, 1, 1, N)
    
    intensities = sampled.squeeze(1).squeeze(1)  # (B, N)
    
    if B == 1:
        intensities = intensities.squeeze(0)
    
    return intensities


def compute_grid_capacity(
    points: torch.Tensor,
    image: torch.Tensor,
    grid_size: Tuple[int, int] = (32, 32),
) -> Dict[str, torch.Tensor]:
    """
    Compute CCVT-style grid capacity fulfillment.
    
    This is the PROPER capacity metric used in Weighted Voronoi Stippling:
    - Divide image into grid cells
    - For each cell: expected_count = (1 - avg_intensity) * N * (1/num_cells)
    - For each cell: actual_count = number of points in cell
    - Fulfillment ratio = actual / expected
    
    Args:
        points: (N, 2) points in [-1, 1]
        image: (1, H, W) grayscale image in [0, 1]
        grid_size: (H_grid, W_grid) number of cells
        
    Returns:
        Dictionary with:
        - 'grid_actual': (H_grid, W_grid) actual point counts
        - 'grid_expected': (H_grid, W_grid) expected point counts
        - 'grid_ratio': (H_grid, W_grid) actual/expected ratio
        - 'grid_status': (H_grid, W_grid) -1=under, 0=ok, 1=over
        - 'score': Overall score [0, 1]
        - 'underfilled_pct': % of cells underfilled
        - 'overfilled_pct': % of cells overfilled
    """
    if points.dim() == 3:
        points = points.squeeze(0)
    if image.dim() == 4:
        image = image.squeeze(0)
    
    N = points.shape[0]
    H_grid, W_grid = grid_size
    device = points.device
    
    # 1. Compute expected density per cell from image
    # Downsample image to grid size
    image_down = F.interpolate(
        image.unsqueeze(0), size=grid_size, mode='bilinear', align_corners=True
    ).squeeze(0).squeeze(0)  # (H_grid, W_grid)
    
    # Expected weight = (1 - intensity) for each cell
    # Dark cells (intensity ~0) need more points, light cells (intensity ~1) need fewer
    expected_weight = 1.0 - image_down + 0.01  # Small epsilon
    expected_weight = expected_weight / expected_weight.sum()  # Normalize to sum=1
    grid_expected = expected_weight * N  # Scale by total points
    
    # 2. Count actual points per cell
    points_norm = (points + 1.0) / 2.0  # [0, 1]
    points_norm = points_norm.clamp(0, 1 - 1e-6)
    
    col_idx = (points_norm[:, 0] * W_grid).long()
    row_idx = (points_norm[:, 1] * H_grid).long()
    
    linear_idx = row_idx * W_grid + col_idx
    grid_actual = torch.zeros(H_grid * W_grid, device=device)
    grid_actual.scatter_add_(0, linear_idx, torch.ones(N, device=device))
    grid_actual = grid_actual.view(H_grid, W_grid)
    
    # 3. Compute ratio
    grid_ratio = grid_actual / (grid_expected + 1e-6)
    
    # 4. Classify cells: underfilled (<0.5), ok (0.5-2.0), overfilled (>2.0)
    grid_status = torch.zeros_like(grid_ratio, dtype=torch.int)
    grid_status[grid_ratio < 0.5] = -1  # Underfilled
    grid_status[grid_ratio > 2.0] = 1   # Overfilled
    
    # 5. Compute score
    # Cells where expected > 0.5 (non-trivial cells)
    significant_cells = grid_expected > 0.5
    n_significant = significant_cells.sum().item()
    
    if n_significant > 0:
        # Count correctly filled cells (ratio between 0.5 and 2.0)
        ok_cells = (grid_status == 0) & significant_cells
        score = ok_cells.sum().item() / n_significant
        
        under_cells = (grid_status == -1) & significant_cells
        over_cells = (grid_status == 1) & significant_cells
        underfilled_pct = 100 * under_cells.sum().item() / n_significant
        overfilled_pct = 100 * over_cells.sum().item() / n_significant
    else:
        score = 1.0
        underfilled_pct = 0.0
        overfilled_pct = 0.0
    
    return {
        'grid_actual': grid_actual,
        'grid_expected': grid_expected,
        'grid_ratio': grid_ratio,
        'grid_status': grid_status,
        'score': score,
        'underfilled_pct': underfilled_pct,
        'overfilled_pct': overfilled_pct,
    }


def compute_capacity_fulfillment(
    points: torch.Tensor,
    image: torch.Tensor,
    num_points_expected: int = None,
) -> Dict[str, torch.Tensor]:
    """
    Compute per-point capacity fulfillment (neighbor-based, legacy metric).
    
    NOTE: This is a secondary metric. The primary CCVT-style metric is 
    compute_grid_capacity() which measures regional density balance.
    
    This metric checks if each point's local density (1/NN_distance) matches
    the expected density based on local image darkness.
    
    Args:
        points: (N, 2) or (B, N, 2) points in [-1, 1]
        image: (1, H, W) or (B, 1, H, W) grayscale image in [0, 1]
        num_points_expected: Expected total points (default: actual count)
        
    Returns:
        Dictionary with capacity metrics per point
    """
    # Handle batch dimension
    if points.dim() == 2:
        points = points.unsqueeze(0)
    if image.dim() == 3:
        image = image.unsqueeze(0)
    
    B, N, _ = points.shape
    device = points.device
    
    if num_points_expected is None:
        num_points_expected = N
    
    # 1. Sample image intensity at each point
    intensities = sample_image_at_points(image, points)  # (B, N)
    
    # 2. Compute expected density: darker = higher expected density
    expected_weight = 1.0 - intensities + 0.01
    expected_weight = expected_weight / expected_weight.sum(dim=-1, keepdim=True)
    expected_density = expected_weight * num_points_expected
    
    # 3. Compute actual local density using nearest neighbor distance
    dists = torch.cdist(points, points)  # (B, N, N)
    mask = torch.eye(N, device=device).bool().unsqueeze(0).expand(B, -1, -1)
    dists = dists.masked_fill(mask, float('inf'))
    nn_dists, _ = dists.min(dim=2)  # (B, N)
    
    # Density inversely proportional to NN distance
    actual_density = 1.0 / (nn_dists + 1e-6)
    actual_density = actual_density / actual_density.sum(dim=-1, keepdim=True)
    actual_density = actual_density * num_points_expected
    
    # 4. Compute fulfillment ratio
    fulfillment_ratio = actual_density / (expected_density + 1e-6)
    
    # 5. Determine fulfilled points
    fulfilled_mask = (fulfillment_ratio > 0.5) & (fulfillment_ratio < 2.0)
    
    # 6. Overall score
    score = 1.0 - torch.abs(fulfillment_ratio - 1.0).clamp(0, 1).mean(dim=-1)
    
    return {
        'intensities': intensities.squeeze(0) if B == 1 else intensities,
        'expected_density': expected_density.squeeze(0) if B == 1 else expected_density,
        'actual_density': actual_density.squeeze(0) if B == 1 else actual_density,
        'fulfillment_ratio': fulfillment_ratio.squeeze(0) if B == 1 else fulfillment_ratio,
        'fulfilled_mask': fulfilled_mask.squeeze(0) if B == 1 else fulfilled_mask,
        'score': score.item() if B == 1 else score,
    }


def compute_density_heatmap(
    points: torch.Tensor,
    grid_size: Tuple[int, int] = (256, 256),
    image_size: Tuple[int, int] = (512, 512),
) -> torch.Tensor:
    """
    Compute density heatmap by counting points in each grid cell.
    
    Args:
        points: (N, 2) or (B, N, 2) points in [-1, 1]
        grid_size: (H_grid, W_grid) output heatmap size (default: half of image)
        image_size: (H, W) original image size for reference
        
    Returns:
        heatmap: (H_grid, W_grid) or (B, H_grid, W_grid) point counts per cell
    """
    # Handle batch dimension
    had_batch = points.dim() == 3
    if points.dim() == 2:
        points = points.unsqueeze(0)
    
    B, N, _ = points.shape
    H_grid, W_grid = grid_size
    device = points.device
    
    # Convert points from [-1, 1] to grid cell indices
    # x -> column, y -> row
    points_norm = (points + 1.0) / 2.0  # [0, 1]
    
    # Clamp to valid range
    points_norm = points_norm.clamp(0, 1 - 1e-6)
    
    # Convert to grid indices
    col_idx = (points_norm[..., 0] * W_grid).long()  # (B, N)
    row_idx = (points_norm[..., 1] * H_grid).long()  # (B, N)
    
    # Create heatmap by counting
    heatmaps = []
    for b in range(B):
        heatmap = torch.zeros(H_grid, W_grid, device=device)
        
        # Count points in each cell
        for i in range(N):
            r, c = row_idx[b, i].item(), col_idx[b, i].item()
            if 0 <= r < H_grid and 0 <= c < W_grid:
                heatmap[r, c] += 1
        
        heatmaps.append(heatmap)
    
    heatmaps = torch.stack(heatmaps, dim=0)  # (B, H_grid, W_grid)
    
    if not had_batch:
        heatmaps = heatmaps.squeeze(0)
    
    return heatmaps


def compute_density_heatmap_fast(
    points: torch.Tensor,
    grid_size: Tuple[int, int] = (256, 256),
) -> torch.Tensor:
    """
    Fast GPU version of density heatmap using scatter_add.
    
    Args:
        points: (N, 2) or (B, N, 2) points in [-1, 1]
        grid_size: (H_grid, W_grid) output heatmap size
        
    Returns:
        heatmap: (H_grid, W_grid) or (B, H_grid, W_grid) point counts per cell
    """
    had_batch = points.dim() == 3
    if points.dim() == 2:
        points = points.unsqueeze(0)
    
    B, N, _ = points.shape
    H_grid, W_grid = grid_size
    device = points.device
    
    # Convert points from [-1, 1] to grid cell indices
    points_norm = (points + 1.0) / 2.0  # [0, 1]
    points_norm = points_norm.clamp(0, 1 - 1e-6)
    
    col_idx = (points_norm[..., 0] * W_grid).long()  # (B, N)
    row_idx = (points_norm[..., 1] * H_grid).long()  # (B, N)
    
    # Flatten to linear index
    linear_idx = row_idx * W_grid + col_idx  # (B, N)
    
    # Scatter add to count
    heatmaps = torch.zeros(B, H_grid * W_grid, device=device)
    ones = torch.ones(B, N, device=device)
    heatmaps.scatter_add_(1, linear_idx, ones)
    heatmaps = heatmaps.view(B, H_grid, W_grid)
    
    if not had_batch:
        heatmaps = heatmaps.squeeze(0)
    
    return heatmaps


def compute_spacing_quality(
    points: torch.Tensor,
    image: torch.Tensor = None,
) -> Dict[str, torch.Tensor]:
    """
    Compute spacing quality metrics - measures how well-spaced points are.
    
    Good stippling has "blue noise" properties:
    - No clumping (no points too close together)
    - Uniform spacing relative to local density
    
    Args:
        points: (N, 2) points in [-1, 1]
        image: Optional (1, H, W) grayscale image for density-aware spacing
        
    Returns:
        Dictionary with:
        - 'nn_distances': (N,) nearest neighbor distance for each point
        - 'nn_mean': Mean NN distance
        - 'nn_std': Std of NN distances  
        - 'nn_cv': Coefficient of variation (std/mean) - lower = more uniform
        - 'clumped_mask': (N,) bool mask of points that are "too close"
        - 'clumped_pct': % of clumped points
        - 'spacing_score': Overall spacing quality [0, 1]
    """
    if points.dim() == 3:
        points = points.squeeze(0)
    
    N = points.shape[0]
    device = points.device
    
    # Compute pairwise distances
    dists = torch.cdist(points.unsqueeze(0), points.unsqueeze(0)).squeeze(0)  # (N, N)
    
    # Set diagonal to inf to exclude self
    dists.fill_diagonal_(float('inf'))
    
    # Get nearest neighbor distances
    nn_dists, _ = dists.min(dim=1)  # (N,)
    
    # Statistics
    nn_mean = nn_dists.mean()
    nn_std = nn_dists.std()
    nn_cv = nn_std / (nn_mean + 1e-6)  # Coefficient of variation
    
    # Expected NN distance for uniform distribution in [-1, 1]^2
    # For N points in area 4, expected NN distance ≈ 0.5 / sqrt(N)
    area = 4.0  # [-1, 1] x [-1, 1]
    expected_nn = 0.5 * (area / N) ** 0.5
    
    # Points are "clumped" if NN distance < 0.3 * expected
    clump_threshold = 0.3 * expected_nn
    clumped_mask = nn_dists < clump_threshold
    clumped_pct = 100.0 * clumped_mask.float().mean().item()
    
    # Spacing score: 
    # 1. Penalize high CV (non-uniform spacing)
    # 2. Penalize clumping
    cv_penalty = torch.clamp(nn_cv, 0, 1).item()  # CV > 1 is really bad
    clump_penalty = clumped_pct / 100.0
    spacing_score = max(0, 1.0 - 0.5 * cv_penalty - 0.5 * clump_penalty)
    
    return {
        'nn_distances': nn_dists,
        'nn_mean': nn_mean.item(),
        'nn_std': nn_std.item(),
        'nn_cv': nn_cv.item(),
        'expected_nn': expected_nn,
        'clumped_mask': clumped_mask,
        'clumped_pct': clumped_pct,
        'spacing_score': spacing_score,
    }


def compute_stippling_metrics(
    gt_points: torch.Tensor,
    pred_points: torch.Tensor,
    image: torch.Tensor,
    grid_size: Tuple[int, int] = None,
    capacity_grid_size: Tuple[int, int] = (32, 32),
) -> Tuple[Dict, Dict]:
    """
    Compute all stippling metrics for GT and predicted points.
    
    Metrics computed:
    1. Grid Capacity: Divides image into rectangular cells, compares actual
       vs expected point counts per cell. Expected = ∫(1-intensity) normalized.
       This is a rectangular approximation of CCVT's Voronoi cell capacity.
       Score = % of cells with correct count (within 20% tolerance).
    2. Density Heatmap: Point counts per cell for visualization.
    
    Note: True CCVT measures capacity per Voronoi region (expensive to compute).
    This grid-based metric is a practical approximation.
    
    Args:
        gt_points: (N, 2) or (B, N, 2) ground truth points in [-1, 1]
        pred_points: (N, 2) or (B, N, 2) predicted points in [-1, 1]
        image: (1, H, W) or (B, 1, H, W) grayscale image in [0, 1]
        grid_size: (H_grid, W_grid) for density heatmap (default: H/2, W/2)
        capacity_grid_size: Grid size for CCVT capacity metric (default: 32x32)
        
    Returns:
        metrics_gt: Dict with GT metrics
        metrics_pred: Dict with predicted metrics
    """
    # Get image size for default grid
    if image.dim() == 3:
        _, H, W = image.shape
    else:
        _, _, H, W = image.shape
    
    if grid_size is None:
        grid_size = (H // 8, W // 8)  # 64x64 for 512x512 images
    
    # 1. Compute GRID capacity (CCVT-style - primary metric)
    grid_capacity_gt = compute_grid_capacity(gt_points, image, capacity_grid_size)
    grid_capacity_pred = compute_grid_capacity(pred_points, image, capacity_grid_size)
    
    # 2. Compute spacing quality (blue noise properties)
    spacing_gt = compute_spacing_quality(gt_points, image)
    spacing_pred = compute_spacing_quality(pred_points, image)
    
    # 3. Compute density heatmaps for visualization
    heatmap_gt = compute_density_heatmap_fast(gt_points, grid_size)
    heatmap_pred = compute_density_heatmap_fast(pred_points, grid_size)
    
    metrics_gt = {
        'grid_capacity': grid_capacity_gt,
        'spacing': spacing_gt,
        'heatmap': heatmap_gt,
    }
    
    metrics_pred = {
        'grid_capacity': grid_capacity_pred,
        'spacing': spacing_pred,
        'heatmap': heatmap_pred,
    }
    
    return metrics_gt, metrics_pred


def visualize_8panel(
    image: torch.Tensor,
    gt_points: torch.Tensor,
    pred_points: torch.Tensor,
    metrics_gt: Dict,
    metrics_pred: Dict,
    save_path: Optional[str] = None,
    title: str = "Stippling Quality Metrics",
    figsize: Tuple[int, int] = (14, 16),
) -> plt.Figure:
    """
    Create 8-panel visualization comparing GT and predicted stippling.
    
    Row 1: Point clouds (GT vs Predicted)
    Row 2: Grid capacity (density fulfillment per region)
    Row 3: Spacing quality (color by NN distance - red=clumped, blue=well-spaced)
    Row 4: Density heatmaps
    
    Args:
        image: (1, H, W) grayscale image
        gt_points: (N, 2) GT points in [-1, 1]
        pred_points: (N, 2) predicted points in [-1, 1]
        metrics_gt: Metrics dict from compute_stippling_metrics
        metrics_pred: Metrics dict from compute_stippling_metrics
        save_path: Optional path to save figure
        title: Figure title
        figsize: Figure size
        
    Returns:
        matplotlib Figure
    """
    from matplotlib.patches import Patch
    import matplotlib.colors as mcolors
    
    # Convert to numpy
    if image.dim() == 4:
        image = image.squeeze(0)
    img_np = image.squeeze(0).cpu().numpy()  # (H, W)
    
    if gt_points.dim() == 3:
        gt_np = gt_points.squeeze(0).cpu().numpy()
    else:
        gt_np = gt_points.cpu().numpy()
        
    if pred_points.dim() == 3:
        pred_np = pred_points.squeeze(0).cpu().numpy()
    else:
        pred_np = pred_points.cpu().numpy()
    
    H, W = img_np.shape
    
    fig, axes = plt.subplots(4, 2, figsize=figsize)
    fig.suptitle(title, fontsize=14, fontweight='bold')
    
    # === Row 1: Point clouds ===
    for ax, points, label in [(axes[0, 0], gt_np, 'Ground Truth'), 
                               (axes[0, 1], pred_np, 'Predicted')]:
        ax.imshow(img_np, cmap='gray', extent=[-1, 1, -1, 1], origin='lower')
        ax.scatter(points[:, 0], points[:, 1], c='black', s=1, alpha=0.8)
        ax.set_title(f"{label}\n({len(points)} points)")
        ax.set_xlim(-1, 1)
        ax.set_ylim(-1, 1)
        ax.set_aspect('equal')
        ax.set_xlabel('x')
        ax.set_ylabel('y')
    
    # === Row 2: Grid Capacity (CCVT-style) ===
    for ax, metrics, label in [
        (axes[1, 0], metrics_gt, 'GT Grid Capacity'),
        (axes[1, 1], metrics_pred, 'Pred Grid Capacity')
    ]:
        grid_cap = metrics['grid_capacity']
        grid_status = grid_cap['grid_status']
        if isinstance(grid_status, torch.Tensor):
            grid_status = grid_status.cpu().numpy()
        
        # Create RGB image: Green=ok(0), Red=underfilled(-1), Blue=overfilled(1)
        H_grid, W_grid = grid_status.shape
        status_rgb = np.zeros((H_grid, W_grid, 3), dtype=np.float32)
        status_rgb[grid_status == 0, 1] = 1.0   # Green = correct
        status_rgb[grid_status == -1, 0] = 1.0  # Red = underfilled
        status_rgb[grid_status == 1, 2] = 1.0   # Blue = overfilled
        
        ax.imshow(status_rgb, origin='lower', extent=[-1, 1, -1, 1])
        
        score = grid_cap['score']
        under = grid_cap['underfilled_pct']
        over = grid_cap['overfilled_pct']
        ok = 100.0 - under - over
        
        ax.set_title(f"{label}\nOK:{ok:.0f}% Under:{under:.0f}% Over:{over:.0f}%\nScore: {score:.3f}")
        ax.set_xlim(-1, 1)
        ax.set_ylim(-1, 1)
        ax.set_aspect('equal')
        ax.set_xlabel('x')
        ax.set_ylabel('y')
        
        legend_elements = [
            Patch(facecolor='green', label='Correct'),
            Patch(facecolor='red', label='Underfilled'),
            Patch(facecolor='blue', label='Overfilled'),
        ]
        ax.legend(handles=legend_elements, loc='upper left', bbox_to_anchor=(1.02, 1), 
                  fontsize=7, borderaxespad=0)
    
    # === Row 3: Spacing Quality (color points by NN distance) ===
    # Get NN distances for coloring
    nn_gt = metrics_gt['spacing']['nn_distances']
    nn_pred = metrics_pred['spacing']['nn_distances']
    if isinstance(nn_gt, torch.Tensor):
        nn_gt = nn_gt.cpu().numpy()
    if isinstance(nn_pred, torch.Tensor):
        nn_pred = nn_pred.cpu().numpy()
    
    # Common color scale
    vmin = min(nn_gt.min(), nn_pred.min())
    vmax = max(nn_gt.max(), nn_pred.max())
    
    for ax, points, nn_dists, metrics, label in [
        (axes[2, 0], gt_np, nn_gt, metrics_gt, 'GT Spacing'),
        (axes[2, 1], pred_np, nn_pred, metrics_pred, 'Pred Spacing')
    ]:
        ax.imshow(img_np, cmap='gray', extent=[-1, 1, -1, 1], origin='lower', alpha=0.3)
        
        # Color by NN distance: red=small(clumped), blue=large(well-spaced)
        sc = ax.scatter(points[:, 0], points[:, 1], c=nn_dists, 
                       cmap='RdYlBu', s=3, alpha=0.8, vmin=vmin, vmax=vmax)
        
        spacing = metrics['spacing']
        cv = spacing['nn_cv']
        clumped = spacing['clumped_pct']
        score = spacing['spacing_score']
        
        ax.set_title(f"{label}\nCV:{cv:.3f} Clumped:{clumped:.1f}%\nScore: {score:.3f}")
        ax.set_xlim(-1, 1)
        ax.set_ylim(-1, 1)
        ax.set_aspect('equal')
        ax.set_xlabel('x')
        ax.set_ylabel('y')
        
        cbar = plt.colorbar(sc, ax=ax, shrink=0.8)
        cbar.set_label('NN Distance')
    
    # === Row 4: Simple Density Heatmaps (point counts) ===
    heatmap_gt = metrics_gt['heatmap']
    heatmap_pred = metrics_pred['heatmap']
    if isinstance(heatmap_gt, torch.Tensor):
        heatmap_gt = heatmap_gt.cpu().numpy()
    if isinstance(heatmap_pred, torch.Tensor):
        heatmap_pred = heatmap_pred.cpu().numpy()
    vmax = max(heatmap_gt.max(), heatmap_pred.max(), 1)
    
    for ax, heatmap, label in [
        (axes[3, 0], heatmap_gt, 'GT Density'),
        (axes[3, 1], heatmap_pred, 'Pred Density')
    ]:
        im = ax.imshow(heatmap, cmap='hot', origin='lower', 
                       extent=[-1, 1, -1, 1], vmin=0, vmax=vmax)
        
        total_points = int(heatmap.sum())
        ax.set_title(f"{label} Heatmap\n({total_points} points)")
        ax.set_aspect('equal')
        ax.set_xlabel('x')
        ax.set_ylabel('y')
        
        cbar = plt.colorbar(im, ax=ax, shrink=0.8)
        cbar.set_label('Point Count')
    
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
        print(f"Saved visualization to {save_path}")
    
    return fig


# Keep old name as alias for compatibility
visualize_6panel = visualize_8panel


def quick_eval(
    image_path: str,
    gt_points_path: str,
    pred_points_path: str,
    save_path: str = None,
):
    """
    Quick evaluation from file paths.
    
    Args:
        image_path: Path to grayscale image
        gt_points_path: Path to GT points .npy file
        pred_points_path: Path to predicted points .npy file
        save_path: Where to save visualization
    """
    from PIL import Image
    
    # Load image
    img = Image.open(image_path).convert('L')
    img_np = np.array(img).astype(np.float32) / 255.0
    image = torch.from_numpy(img_np).unsqueeze(0)  # (1, H, W)
    
    # Load points
    gt_points = torch.from_numpy(np.load(gt_points_path)).float()
    pred_points = torch.from_numpy(np.load(pred_points_path)).float()
    
    # Compute metrics
    metrics_gt, metrics_pred = compute_stippling_metrics(gt_points, pred_points, image)
    
    # Visualize
    fig = visualize_6panel(image, gt_points, pred_points, metrics_gt, metrics_pred, save_path)
    
    return fig, metrics_gt, metrics_pred


def extract_points_from_mask(mask_path: str, image_size: int = 512, num_points: int = 5000) -> torch.Tensor:
    """
    Extract stipple points from a binary mask image.
    Black pixels = stipple locations.
    
    Args:
        mask_path: Path to binary mask image
        image_size: Size to resize to
        num_points: Number of points to sample
        
    Returns:
        points: (N, 2) tensor in [-1, 1]
    """
    from PIL import Image
    
    target = Image.open(mask_path).convert('1').resize(
        (image_size, image_size), Image.NEAREST
    )
    target_np = np.asarray(target, dtype=bool)
    
    # Get black pixel coordinates (stipple locations)
    ys, xs = np.where(~target_np)  # ~ inverts: black=True
    
    if len(xs) == 0:
        return torch.zeros(0, 2)
    
    # Sample if too many points
    if len(xs) > num_points:
        indices = np.random.choice(len(xs), size=num_points, replace=False)
        xs, ys = xs[indices], ys[indices]
    
    # Normalize to [-1, 1]
    points = np.stack([xs, ys], axis=1).astype(np.float32)
    points = (points / (image_size - 1)) * 2.0 - 1.0
    
    return torch.from_numpy(points)


def evaluate_dataset_samples(
    source_dir: str,
    target_dir: str,
    output_dir: str,
    checkpoint_path: str = None,
    image_size: int = 512,
    num_points: int = 5000,
    num_inference_steps: int = 50,
    seed: int = 42,
):
    """
    Evaluate stippling metrics on dataset samples using trained model.
    Samples one image from each unique image type (9 total).
    
    Args:
        source_dir: Directory with source grayscale images
        target_dir: Directory with target stippling masks
        output_dir: Where to save visualizations
        checkpoint_path: Path to trained model checkpoint
        image_size: Image size
        num_points: Number of points per sample
        num_inference_steps: Diffusion sampling steps
        seed: Random seed for reproducibility
    """
    from PIL import Image
    import random
    import sys
    
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    print(f"Device: {device}")
    
    if device == 'cpu':
        print("WARNING: Running on CPU - this will be slow!")
    
    # Create output directory
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)
    
    # Load model if checkpoint provided
    model = None
    scheduler = None
    if checkpoint_path:
        print(f"\nLoading model from: {checkpoint_path}")
        
        # Add experiments_pointdit_v5 to path for imports
        v5_path = Path(checkpoint_path).parent.parent / "experiments_pointdit_v5"
        if v5_path.exists():
            sys.path.insert(0, str(v5_path))
        
        from config import Config
        from model import PointDiT
        from diffusion import DDPMScheduler, sample
        
        checkpoint = torch.load(checkpoint_path, map_location=device, weights_only=False)
        config = checkpoint.get('config', Config.default())
        
        model = PointDiT(
            n_points=config.model.n_points,
            dim=config.model.dim,
            n_layers=config.model.n_layers,
            n_heads=config.model.n_heads,
            image_size=config.model.image_size,
            dropout=0.0,
        ).to(device)
        
        model.load_state_dict(checkpoint['model_state_dict'])
        model.eval()
        
        scheduler = DDPMScheduler(
            num_train_timesteps=config.diffusion.num_train_timesteps,
            beta_start=config.diffusion.beta_start,
            beta_end=config.diffusion.beta_end,
            beta_schedule=config.diffusion.beta_schedule,
        )
        
        print(f"  Loaded from epoch {checkpoint['epoch']}")
        print(f"  Model params: {sum(p.numel() for p in model.parameters()):,}")
    else:
        print("\nNo checkpoint provided - will use GT + noise as predictions")
    
    # Find all images
    source_path = Path(source_dir)
    extensions = ['*.png', '*.jpg', '*.jpeg', '*.bmp']
    image_files = []
    for ext in extensions:
        image_files.extend(list(source_path.glob(ext)))
    
    if not image_files:
        raise ValueError(f"No images found in {source_dir}")
    
    print(f"\nFound {len(image_files)} images in dataset")
    
    # Group images by type (extract type from filename pattern: gen_gray_{TYPE}_{id}_{id}.png)
    from collections import defaultdict
    type_to_files = defaultdict(list)
    for f in image_files:
        # Extract type: remove "gen_gray_" prefix and "_number_number.png" suffix
        name = f.stem  # e.g., "gen_gray_Combined_Shape_1332882039_2697"
        parts = name.split('_')
        if len(parts) >= 4 and parts[0] == 'gen' and parts[1] == 'gray':
            # Type is everything between "gen_gray_" and the last two numeric parts
            type_parts = parts[2:-2]  # e.g., ['Combined', 'Shape']
            img_type = '_'.join(type_parts) if type_parts else 'unknown'
        else:
            img_type = 'unknown'
        type_to_files[img_type].append(f)
    
    print(f"Found {len(type_to_files)} unique image types: {list(type_to_files.keys())}")
    
    # Sample one from each type
    selected_files = []
    for img_type, files in sorted(type_to_files.items()):
        selected = random.choice(files)
        selected_files.append(selected)
        print(f"  {img_type}: {selected.name}")
    
    image_files = selected_files
    print(f"\nEvaluating {len(image_files)} samples (one per type)...")
    
    all_metrics = []
    
    for i, source_file in enumerate(image_files):
        print(f"\n[{i+1}/{len(image_files)}] Processing: {source_file.name}", flush=True)
        
        # Load source image
        img = Image.open(source_file).convert('L').resize(
            (image_size, image_size), Image.BILINEAR
        )
        img_np = np.asarray(img, dtype=np.float32) / 255.0
        image = torch.from_numpy(img_np).unsqueeze(0).to(device)  # (1, H, W)
        
        # Load target points from mask
        target_file = Path(target_dir) / source_file.name
        if not target_file.exists():
            print(f"  Warning: Target not found, skipping: {target_file}", flush=True)
            continue
        
        gt_points = extract_points_from_mask(str(target_file), image_size, num_points).to(device)
        print(f"  GT points: {len(gt_points)}", flush=True)
        
        # Generate predictions
        if model is not None:
            print(f"  Generating predictions with model ({num_inference_steps} steps)...", flush=True)
            with torch.no_grad():
                # Need batch dimension for model
                image_batch = image.unsqueeze(0)  # (1, 1, H, W)
                pred_points = sample(
                    model, scheduler, image_batch,
                    num_points=num_points,
                    num_inference_steps=num_inference_steps,
                    device=device,
                    show_progress=False,
                )
                pred_points = pred_points.squeeze(0)  # (N, 2)
            print(f"  Pred points: {len(pred_points)}", flush=True)
        else:
            # Fallback: add noise to GT
            noise_level = 0.02
            pred_points = gt_points + torch.randn_like(gt_points) * noise_level
            pred_points = pred_points.clamp(-1, 1)
        
        # Compute metrics
        print(f"  Computing metrics...", flush=True)
        metrics_gt, metrics_pred = compute_stippling_metrics(gt_points, pred_points, image)
        
        # Print scores
        gt_grid = metrics_gt['grid_capacity']['score']
        pred_grid = metrics_pred['grid_capacity']['score']
        gt_spacing = metrics_gt['spacing']['spacing_score']
        pred_spacing = metrics_pred['spacing']['spacing_score']
        
        print(f"  GT   - Capacity: {gt_grid:.3f}, Spacing: {gt_spacing:.3f}", flush=True)
        print(f"  Pred - Capacity: {pred_grid:.3f}, Spacing: {pred_spacing:.3f}", flush=True)
        
        # Save visualization
        print(f"  Saving visualization...", flush=True)
        save_file = output_path / f"sample_{i+1:02d}_{source_file.stem}.png"
        fig = visualize_8panel(
            image, gt_points, pred_points,
            metrics_gt, metrics_pred,
            save_path=str(save_file),
            title=f"Stippling Metrics: {source_file.name}"
        )
        plt.close(fig)
        plt.close('all')  # Close any remaining figures
        
        all_metrics.append({
            'file': source_file.name,
            'image_type': source_file.stem.replace('gen_gray_', '').rsplit('_', 2)[0],
            'gt_grid_capacity': gt_grid,
            'gt_spacing': gt_spacing,
            'pred_grid_capacity': pred_grid,
            'pred_spacing': pred_spacing,
            'num_gt_points': len(gt_points),
        })
        
        # Cleanup GPU memory
        del image, gt_points, pred_points, metrics_gt, metrics_pred
        if device == 'cuda':
            torch.cuda.empty_cache()
        
        print(f"  Done with sample {i+1}", flush=True)
    
    # Summary
    print("\n" + "="*60)
    print("SUMMARY")
    print("="*60)
    
    if all_metrics:
        avg_gt_grid = np.mean([m['gt_grid_capacity'] for m in all_metrics])
        avg_pred_grid = np.mean([m['pred_grid_capacity'] for m in all_metrics])
        avg_gt_spacing = np.mean([m['gt_spacing'] for m in all_metrics])
        avg_pred_spacing = np.mean([m['pred_spacing'] for m in all_metrics])
        
        print(f"Samples evaluated: {len(all_metrics)}")
        print(f"\nPer-type results (Capacity / Spacing):")
        for m in all_metrics:
            print(f"  {m['image_type']:25s}: GT={m['gt_grid_capacity']:.3f}/{m['gt_spacing']:.3f}, Pred={m['pred_grid_capacity']:.3f}/{m['pred_spacing']:.3f}")
        
        print(f"\nOverall Averages:")
        print(f"  GT   - Capacity: {avg_gt_grid:.4f}, Spacing: {avg_gt_spacing:.4f}")
        print(f"  Pred - Capacity: {avg_pred_grid:.4f}, Spacing: {avg_pred_spacing:.4f}")
    
    print(f"\nResults saved to: {output_path}")
    
    return all_metrics


# Test function
def test_metrics():
    """Test the metrics with synthetic data."""
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    
    # Create synthetic image (gradient)
    H, W = 512, 512
    x = torch.linspace(0, 1, W).unsqueeze(0).expand(H, -1)
    y = torch.linspace(0, 1, H).unsqueeze(1).expand(-1, W)
    image = 0.5 * (x + y)  # Gradient from dark to light
    image = image.unsqueeze(0).to(device)  # (1, H, W)
    
    # Create random points
    N = 5000
    gt_points = torch.rand(N, 2, device=device) * 2 - 1  # [-1, 1]
    pred_points = gt_points + torch.randn_like(gt_points) * 0.1  # Add noise
    pred_points = pred_points.clamp(-1, 1)
    
    print("Computing metrics...")
    metrics_gt, metrics_pred = compute_stippling_metrics(gt_points, pred_points, image)
    
    print(f"\nGT   - Capacity: {metrics_gt['grid_capacity']['score']:.4f}, Spacing: {metrics_gt['spacing']['spacing_score']:.4f}")
    print(f"Pred - Capacity: {metrics_pred['grid_capacity']['score']:.4f}, Spacing: {metrics_pred['spacing']['spacing_score']:.4f}")
    
    print("\nCreating visualization...")
    fig = visualize_8panel(
        image, gt_points, pred_points, 
        metrics_gt, metrics_pred,
        save_path='test_stippling_metrics.png'
    )
    plt.close(fig)
    
    print("Done! Check test_stippling_metrics.png")


if __name__ == "__main__":
    # Evaluate on real dataset samples using trained V5 model
    SOURCE_DIR = "/groups/asharf_group/ofirgila/ControlNet/training/data_grads_v3/source"
    TARGET_DIR = "/groups/asharf_group/ofirgila/ControlNet/training/data_grads_v3/target"
    OUTPUT_DIR = "/groups/asharf_group/ofirgila/projection-conditioned-point-cloud-diffusion/experiments_pointdit_v5/outputs_custom_metrics"
    CHECKPOINT_PATH = "/groups/asharf_group/ofirgila/projection-conditioned-point-cloud-diffusion/outputs_pointdit_v5_hybrid/checkpoint_best.pth"
    
    evaluate_dataset_samples(
        source_dir=SOURCE_DIR,
        target_dir=TARGET_DIR,
        output_dir=OUTPUT_DIR,
        checkpoint_path=CHECKPOINT_PATH,
        image_size=512,
        num_points=5000,
        num_inference_steps=50,
    )
