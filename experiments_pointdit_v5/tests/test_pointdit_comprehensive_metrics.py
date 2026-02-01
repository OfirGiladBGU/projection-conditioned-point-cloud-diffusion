"""
Comprehensive PointDiT evaluation with spectral, spatial, and point-cloud metrics.
Tests PointDiT overfitting on a single sample for three loss combinations,
then evaluates with spectral metrics (radial profile, anisotropy) + spatial metrics.
"""

import pathlib
import sys
import os

sys.path.append(str(pathlib.Path(__file__).parent.parent))

from pathlib import Path
import numpy as np
import torch
import matplotlib.pyplot as plt
from scipy.spatial import cKDTree

from config import Config
from dataset import SimpleImageDataset
from model import PointDiT
from diffusion import DDPMScheduler, train_step, sample, ChamferLoss
from sinkhorn_lloyd_losses import SinkhornDensityLoss, SinkhornDensityLossSimple, HAS_GEOMLOSS


########################
# Spectral Metrics (adapted from metrics_v2.py)
########################

def log_power_spectrum_2d(numpy_2d: np.ndarray, epsilon: float = 1.0) -> np.ndarray:
    """Compute log power spectrum using 2D FFT."""
    tensor_2d = torch.from_numpy(numpy_2d).float()
    tensor_2d -= tensor_2d.mean()
    fft = torch.fft.fftshift(torch.fft.fft2(tensor_2d))
    power = torch.abs(fft) ** 2
    log_power = torch.log(power + epsilon)
    return log_power.numpy()


def radial_profile_2d(numpy_2d: np.ndarray, normalize: bool = True) -> np.ndarray:
    """Compute radial power spectrum profile."""
    spectrum = log_power_spectrum_2d(numpy_2d)
    h, w = spectrum.shape
    cy, cx = h // 2, w // 2

    y, x = np.indices((h, w))
    r = np.sqrt((x - cx)**2 + (y - cy)**2).astype(np.int32)

    tbin = np.bincount(r.ravel(), spectrum.ravel())
    rbin = np.bincount(r.ravel())
    profile = tbin / rbin

    if normalize:
        profile /= np.max(profile)

    return profile


def radial_profile_difference(image_gt: np.ndarray, image_pr: np.ndarray) -> float:
    """Mean absolute difference in radial power spectrum."""
    p_gt = radial_profile_2d(image_gt, normalize=True)
    p_pr = radial_profile_2d(image_pr, normalize=True)
    n = min(len(p_gt), len(p_pr))
    diff = np.abs(p_gt[:n] - p_pr[:n])
    return float(np.mean(diff))


def anisotropy_metric_2d(spectrum_2d: np.ndarray, r_min: int = 5, r_max: int = None) -> np.ndarray:
    """Compute angular power distribution (anisotropy)."""
    h, w = spectrum_2d.shape
    cy, cx = h // 2, w // 2

    if r_max is None:
        r_max = min(cx, cy)

    y, x = np.indices((h, w))
    dx, dy = x - cx, y - cy
    r = np.sqrt(dx**2 + dy**2)
    theta = np.arctan2(dy, dx)

    mask = (r >= r_min) & (r <= r_max)
    theta_vals = theta[mask]
    power_vals = spectrum_2d[mask]

    num_angles = 180
    bins = np.linspace(-np.pi, np.pi, num_angles + 1)
    angular_power = np.zeros(num_angles)

    for i in range(num_angles):
        m = (theta_vals >= bins[i]) & (theta_vals < bins[i + 1])
        angular_power[i] = power_vals[m].mean() if np.any(m) else 0.0

    angular_power /= np.mean(angular_power)
    return angular_power


def anisotropy_score(angular_power: np.ndarray) -> float:
    """Compute anisotropy as std of angular power distribution."""
    return float(np.std(angular_power))


def radial_anisotropy_difference(image_gt: np.ndarray, image_pr: np.ndarray) -> float:
    """Mean absolute difference in anisotropy scores."""
    spectrum_gt = log_power_spectrum_2d(image_gt)
    spectrum_pr = log_power_spectrum_2d(image_pr)
    
    ang_gt = anisotropy_metric_2d(spectrum_gt)
    ang_pr = anisotropy_metric_2d(spectrum_pr)
    
    score_gt = anisotropy_score(ang_gt)
    score_pr = anisotropy_score(ang_pr)
    
    return float(np.abs(score_gt - score_pr))


########################
# Spatial Metrics (Point Distribution)
########################

def pair_correlation_function_2d(points: np.ndarray, r_max: float = 50.0, dr: float = 1.0):
    """Compute pair correlation function g(r) for point distribution."""
    if len(points) == 0:
        return np.array([]), np.array([])
    
    tree = cKDTree(points)
    pairs = tree.query_pairs(r_max, output_type="ndarray")
    
    if len(pairs) == 0:
        return np.array([]), np.array([])
    
    dists = np.linalg.norm(points[pairs[:, 0]] - points[pairs[:, 1]], axis=1)
    bins = np.arange(0, r_max + dr, dr)
    hist, edges = np.histogram(dists, bins=bins)
    r = 0.5 * (edges[:-1] + edges[1:])
    
    # Normalize by density
    area = 100.0  # Assuming normalized [-1, 1] x [-1, 1] -> scale to pixel space
    density = len(points) / area
    shell_areas = 2 * np.pi * r * dr
    hist = hist / (shell_areas * density * len(points) + 1e-8)
    
    return r, hist


def compute_rdf_difference(gt_points: np.ndarray, pred_points: np.ndarray, r_max: float = 50.0, dr: float = 1.0) -> float:
    """Compute mean absolute difference in pair correlation functions."""
    r_gt, g_gt = pair_correlation_function_2d(gt_points, r_max, dr)
    r_pr, g_pr = pair_correlation_function_2d(pred_points, r_max, dr)
    
    if len(g_gt) == 0 or len(g_pr) == 0:
        return 0.0
    
    n = min(len(g_gt), len(g_pr))
    diff = np.abs(g_gt[:n] - g_pr[:n])
    return float(np.mean(diff))


########################
# Point Cloud Metrics
########################

def compute_blue_noise_metrics(points: torch.Tensor, gt_points: torch.Tensor = None):
    """Compute blue-noise metrics (Mean NN, CV, Chamfer)."""
    B, N, _ = points.shape
    dists = torch.cdist(points, points)
    mask = torch.eye(N, device=points.device).bool().unsqueeze(0).expand(B, -1, -1)
    dists = dists.masked_fill(mask, float("inf"))
    nn_dists, _ = dists.min(dim=2)

    metrics = {
        "mean_nn": nn_dists.mean().item(),
        "std_nn": nn_dists.std().item(),
        "min_nn": nn_dists.min().item(),
        "max_nn": nn_dists.max().item(),
    }
    metrics["cv"] = metrics["std_nn"] / (metrics["mean_nn"] + 1e-8)

    if gt_points is not None:
        chamfer = ChamferLoss()
        metrics["chamfer"] = chamfer(points, gt_points).item()

    return metrics


def compute_sinkhorn_metric(points: torch.Tensor, image: torch.Tensor, sinkhorn_loss):
    """Compute Sinkhorn optimal transport metric."""
    return sinkhorn_loss(points, image)


########################
# Image Rendering (convert point cloud to image)
########################

def render_points_to_image(points: np.ndarray, image_size: int = 256, radius: float = 2.0) -> np.ndarray:
    """Render point cloud as binary image (for spectral metrics)."""
    img = np.zeros((image_size, image_size), dtype=np.float32)
    
    # Normalize points from [-1, 1] to [0, image_size]
    pts_norm = (points + 1.0) / 2.0 * image_size
    pts_norm = pts_norm.astype(np.int32)
    
    # Clamp to image bounds
    pts_norm = np.clip(pts_norm, 0, image_size - 1)
    
    # Draw circles at each point
    for pt in pts_norm:
        y, x = pt[1], pt[0]
        if 0 <= x < image_size and 0 <= y < image_size:
            img[y, x] = 1.0
    
    return img


########################
# Main Test
########################

def main():
    config = Config.default()
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Device: {device}")

    dataset = SimpleImageDataset(
        source_dir=config.data.source_dir,
        target_dir=config.data.target_dir,
        image_size=config.data.image_size,
        num_points=config.model.n_points,
    )
    print(f"Dataset loaded: {len(dataset)} samples")

    scheduler = DDPMScheduler(
        num_train_timesteps=config.diffusion.num_train_timesteps,
        beta_start=config.diffusion.beta_start,
        beta_end=config.diffusion.beta_end,
        beta_schedule=config.diffusion.beta_schedule,
    )

    # Initialize sinkhorn metric
    if HAS_GEOMLOSS:
        sinkhorn_metric = SinkhornDensityLoss(blur=0.01, grid_size=64).to(device)
    else:
        sinkhorn_metric = SinkhornDensityLossSimple(grid_size=32, blur=0.05).to(device)
    sample_idx = 0
    data = dataset[sample_idx]
    image = data["image"].unsqueeze(0).to(device)
    gt_points = data["points"].unsqueeze(0).to(device)
    num_points = gt_points.shape[1]

    # Convert GT points to image for spectral metrics
    gt_img = render_points_to_image(gt_points[0].cpu().numpy())

    loss_combos = [
        {"name": "Chamfer Only", "chamfer_weight": 1.0, "sinkhorn_weight": 0.0},
        {"name": "Sinkhorn Only", "chamfer_weight": 0.0, "sinkhorn_weight": 1.0},
        {"name": "Sinkhorn+Chamfer", "chamfer_weight": 1.0, "sinkhorn_weight": 1.0},
    ]

    out_dir = Path(__file__).parent / "outputs_pointdit_v5_comprehensive"
    out_dir.mkdir(exist_ok=True)

    results = []
    steps = 1000

    print("\n" + "="*80)
    print("COMPREHENSIVE POINTDIT EVALUATION")
    print("="*80)

    for combo in loss_combos:
        print(f"\n=== Overfit PointDiT with {combo['name']} ===")
        model = PointDiT(
            n_points=config.model.n_points,
            dim=config.model.dim,
            n_layers=config.model.n_layers,
            n_heads=config.model.n_heads,
            image_size=config.model.image_size,
            dropout=0.0,
        ).to(device)

        optimizer = torch.optim.AdamW(model.parameters(), lr=1e-3, weight_decay=1e-5)

        # Training loop
        for step in range(steps):
            model.train()
            loss_dict = train_step(
                model,
                scheduler,
                gt_points,
                image,
                device=device,
                chamfer_weight=combo["chamfer_weight"],
                sinkhorn_weight=combo["sinkhorn_weight"],
                repulsion_weight=0.0,
                grid_density_weight=0.0,
                use_adaptive_repulsion=False,
            )
            loss = loss_dict["loss"]
            optimizer.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()

            if step % 200 == 0 or step == steps - 1:
                print(
                    f"Step {step:4d} | loss={loss.item():.6f} "
                    f"chamfer={loss_dict['chamfer']:.6f} "
                    f"sinkhorn={loss_dict['sinkhorn']:.6f}"
                )

        # Sampling
        model.eval()
        with torch.no_grad():
            pred_points = sample(
                model,
                scheduler,
                image,
                num_points,
                num_inference_steps=50,
                device=device,
                show_progress=False,
                init_from_density=True,
                init_std=0.02,
                eta=0.0,
            )

        # =====================
        # Point Cloud Metrics
        # =====================
        metrics = compute_blue_noise_metrics(pred_points, gt_points)
        sinkhorn_val = compute_sinkhorn_metric(pred_points, image, sinkhorn_metric).item()
        
        # =====================
        # Spectral Metrics
        # =====================
        pred_img = render_points_to_image(pred_points[0].cpu().numpy())
        radial_diff = radial_profile_difference(gt_img, pred_img)
        anisotropy_diff = radial_anisotropy_difference(gt_img, pred_img)
        
        # =====================
        # Spatial Metrics (RDF)
        # =====================
        gt_pts_np = gt_points[0].cpu().numpy()
        pred_pts_np = pred_points[0].cpu().numpy()
        rdf_diff = compute_rdf_difference(gt_pts_np, pred_pts_np, r_max=0.5, dr=0.01)

        # Print results
        print(f"\n{'Metric':<30} {'Value':<15} {'GT vs Pred':<15}")
        print("-" * 60)
        print(f"{'Mean NN':<30} {metrics['mean_nn']:<15.6f}")
        print(f"{'CV (spacing uniformity)':<30} {metrics['cv']:<15.6f}")
        print(f"{'Chamfer Distance':<30} {metrics['chamfer']:<15.6f}")
        print(f"{'Sinkhorn (OT distance)':<30} {sinkhorn_val:<15.6f}")
        print(f"{'Radial Spectrum Diff':<30} {radial_diff:<15.6f} (lower=better)")
        print(f"{'Anisotropy Diff':<30} {anisotropy_diff:<15.6f} (lower=better)")
        print(f"{'RDF Difference':<30} {rdf_diff:<15.6f} (lower=better)")

        results.append({
            "name": combo["name"],
            "mean_nn": metrics["mean_nn"],
            "cv": metrics["cv"],
            "chamfer": metrics["chamfer"],
            "sinkhorn": sinkhorn_val,
            "radial_diff": radial_diff,
            "anisotropy_diff": anisotropy_diff,
            "rdf_diff": rdf_diff,
        })

        # Visualization
        img = image[0, 0].cpu().numpy()
        gt = gt_points[0].cpu().numpy()
        pred = pred_points[0].cpu().numpy()

        fig, axes = plt.subplots(1, 3, figsize=(18, 5))
        
        # GT
        axes[0].imshow(img, cmap="gray", extent=[-1, 1, -1, 1], origin="lower")
        axes[0].scatter(gt[:, 0], gt[:, 1], c="red", s=2, alpha=0.8)
        axes[0].set_title("Ground Truth", fontsize=12)
        axes[0].set_xlim(-1, 1)
        axes[0].set_ylim(-1, 1)
        
        # Prediction
        axes[1].imshow(img, cmap="gray", extent=[-1, 1, -1, 1], origin="lower")
        axes[1].scatter(pred[:, 0], pred[:, 1], c="blue", s=2, alpha=0.8)
        axes[1].set_title(f"PointDiT: {combo['name']}", fontsize=12)
        axes[1].set_xlim(-1, 1)
        axes[1].set_ylim(-1, 1)
        
        # Overlay
        axes[2].imshow(img, cmap="gray", extent=[-1, 1, -1, 1], origin="lower")
        axes[2].scatter(gt[:, 0], gt[:, 1], c="red", s=1.5, alpha=0.6, label="GT")
        axes[2].scatter(pred[:, 0], pred[:, 1], c="blue", s=1.5, alpha=0.6, label="Pred")
        axes[2].set_title("Overlay (Red=GT, Blue=Pred)", fontsize=12)
        axes[2].legend()
        axes[2].set_xlim(-1, 1)
        axes[2].set_ylim(-1, 1)

        plt.tight_layout()
        filename = f"pointdit_{combo['name'].replace(' ', '_').replace('+', 'plus').lower()}.png"
        plt.savefig(out_dir / filename, dpi=150, bbox_inches="tight")
        plt.close()
        print(f"  Visualization saved to: {out_dir / filename}")

    # =====================
    # Summary Table
    # =====================
    print("\n" + "="*80)
    print("SUMMARY TABLE: All Metrics Comparison")
    print("="*80)
    print(f"{'Method':<20} {'Mean NN':<12} {'CV':<12} {'Chamfer':<12} {'Sinkhorn':<12} {'RadSpec':<12} {'Aniso':<12} {'RDF':<12}")
    print("-" * 112)
    for r in results:
        print(f"{r['name']:<20} {r['mean_nn']:<12.6f} {r['cv']:<12.6f} {r['chamfer']:<12.6f} {r['sinkhorn']:<12.6f} {r['radial_diff']:<12.6f} {r['anisotropy_diff']:<12.6f} {r['rdf_diff']:<12.6f}")

    # Save results to CSV
    import csv
    csv_path = out_dir / "comprehensive_results.csv"
    with open(csv_path, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=results[0].keys())
        writer.writeheader()
        writer.writerows(results)
    print(f"\n✓ Results saved to: {csv_path}")


if __name__ == "__main__":
    main()
