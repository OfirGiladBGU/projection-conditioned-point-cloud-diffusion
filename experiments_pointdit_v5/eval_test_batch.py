"""
Evaluation script for PointDiT v5 on test batch.

Combines spectral metrics (from metrics_v2.py) with NN metrics to evaluate
the model on the test batch and generate comprehensive visualizations.
"""

import sys
import pathlib

sys.path.insert(0, str(pathlib.Path(__file__).parent.parent.parent / "Stable_Diffusion"))
sys.path.insert(0, str(pathlib.Path(__file__).parent))

from pathlib import Path
import numpy as np
import torch
import matplotlib.pyplot as plt
from PIL import Image
from tqdm import tqdm
import json

from config import Config
from model import PointDiT
from diffusion import DDPMScheduler, sample, ChamferLoss

# Import spectral metrics
try:
    from metrics_v2 import (
        radial_profile_2d,
        anisotropy_metric_2d,
        log_power_spectrum_2d,
    )
    HAS_SPECTRAL_METRICS = True
except ImportError:
    print("Warning: metrics_v2 not available, will only compute NN metrics")
    HAS_SPECTRAL_METRICS = False


def compute_blue_noise_metrics(points: torch.Tensor, gt_points: torch.Tensor = None):
    """Compute blue-noise metrics (NN, CV, Chamfer)."""
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


def convert_points_to_image(points: np.ndarray, image_size: int = 512) -> np.ndarray:
    """Convert normalized points [-1, 1] to binary stippling image."""
    # Normalize points from [-1, 1] to [0, image_size]
    points_pixel = (points + 1.0) * (image_size / 2.0)
    points_pixel = np.clip(points_pixel, 0, image_size - 1).astype(np.int32)

    # Create binary image
    img = np.zeros((image_size, image_size), dtype=np.uint8)
    img[points_pixel[:, 1], points_pixel[:, 0]] = 255

    return img


def main():
    config = Config.default()
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Device: {device}")

    # Load model
    model = PointDiT(
        n_points=config.model.n_points,
        dim=config.model.dim,
        n_layers=config.model.n_layers,
        n_heads=config.model.n_heads,
        image_size=config.model.image_size,
        dropout=0.0,
    ).to(device)

    # Load latest checkpoint
    checkpoint_path = Path(__file__).parent.parent / "outputs_pointdit_v5_hybrid" / "checkpoint_latest.pth"
    if checkpoint_path.exists():
        checkpoint = torch.load(checkpoint_path, weights_only=False)
        model.load_state_dict(checkpoint["model_state_dict"])
        print(f"Loaded checkpoint from {checkpoint_path}")
    else:
        print(f"Warning: No checkpoint found at {checkpoint_path}, using untrained model")

    model.eval()

    scheduler = DDPMScheduler(
        num_train_timesteps=config.diffusion.num_train_timesteps,
        beta_start=config.diffusion.beta_start,
        beta_end=config.diffusion.beta_end,
        beta_schedule=config.diffusion.beta_schedule,
    )

    # Load test batch
    test_source_dir = Path("/groups/asharf_group/ofirgila/ControlNet/training/data_grads_v3_test_batch/source")
    test_target_dir = Path("/groups/asharf_group/ofirgila/ControlNet/training/data_grads_v3_test_batch/target")

    source_files = sorted(test_source_dir.glob("*.png"))
    print(f"\nLoaded {len(source_files)} test samples")

    # Output directory
    out_dir = Path(__file__).parent / "evaluation_results"
    out_dir.mkdir(exist_ok=True)

    # Collect metrics for all samples
    all_metrics = {
        "nn_metrics": [],
        "spectral_metrics": [],
        "radial_profiles_gt": [],
        "radial_profiles_pred": [],
    }

    num_samples = min(2000, len(source_files))
    print(f"\nEvaluating on {num_samples} test samples...")
    for idx, source_file in enumerate(tqdm(source_files[:num_samples])):
        # Load image
        img = Image.open(source_file).convert("L")
        img_np = np.array(img) / 255.0
        img_tensor = torch.from_numpy(img_np).float().unsqueeze(0).unsqueeze(0).to(device)

        # Load GT points
        target_file = test_target_dir / source_file.name
        target_img = Image.open(target_file).convert("L")
        target_np = np.array(target_img)
        gt_points_pixel = np.where(target_np > 127)
        gt_points_pixel = np.stack([gt_points_pixel[1], gt_points_pixel[0]], axis=1).astype(np.float32)

        # Convert to normalized coordinates
        image_size = target_np.shape[0]
        gt_points_norm = (gt_points_pixel / (image_size / 2.0)) - 1.0
        gt_points_tensor = torch.from_numpy(gt_points_norm).float().unsqueeze(0).to(device)

        # Predict
        with torch.no_grad():
            pred_points = sample(
                model,
                scheduler,
                img_tensor,
                config.model.n_points,
                num_inference_steps=50,
                device=device,
                show_progress=False,
                init_from_density=True,
                init_std=0.02,
                eta=0.0,
            )

        # Convert to pixel coordinates
        pred_points_np = pred_points[0].cpu().numpy()
        
        # Debug: Check point ranges
        if idx == 0:
            print(f"\nDebug info (Sample 0):")
            print(f"  Pred points shape: {pred_points_np.shape}")
            print(f"  Pred points range: x=[{pred_points_np[:, 0].min():.3f}, {pred_points_np[:, 0].max():.3f}], y=[{pred_points_np[:, 1].min():.3f}, {pred_points_np[:, 1].max():.3f}]")
            print(f"  Image size: {image_size}")
        
        pred_points_pixel = (pred_points_np + 1.0) * (image_size / 2.0)
        pred_img = convert_points_to_image(pred_points_pixel, image_size)

        # Compute NN metrics
        nn_metrics = compute_blue_noise_metrics(pred_points, gt_points_tensor)
        all_metrics["nn_metrics"].append(nn_metrics)

        # Compute spectral metrics
        if HAS_SPECTRAL_METRICS:
            try:
                radial_prof_pred = radial_profile_2d(pred_img, plot=False)
                radial_prof_gt = radial_profile_2d(target_np, plot=False)

                all_metrics["radial_profiles_gt"].append(radial_prof_gt)
                all_metrics["radial_profiles_pred"].append(radial_prof_pred)

                spec_metrics = {
                    "radial_profile_diff": float(np.mean(np.abs(radial_prof_pred - radial_prof_gt))),
                }
                all_metrics["spectral_metrics"].append(spec_metrics)
            except Exception as e:
                print(f"Error computing spectral metrics for sample {idx}: {e}")

        # Save detailed visualization for first 5 samples
        if idx < 5:
            # Create white background image for predicted points (black on white)
            pred_img_white = np.ones((image_size, image_size), dtype=np.uint8) * 255
            pred_points_pixel_int = np.clip(pred_points_pixel, 0, image_size - 1).astype(np.int32)
            pred_img_white[pred_points_pixel_int[:, 1], pred_points_pixel_int[:, 0]] = 0

            # Create layout: 3 columns x 2 rows
            fig = plt.figure(figsize=(16, 10))

            # Row 1: INPUT | GT POINTS | OUTPUT POINTS
            ax1 = plt.subplot(2, 3, 1)
            ax1.imshow(img_np, cmap="gray")
            ax1.set_title("Input Image", fontsize=12, fontweight="bold")
            ax1.axis("off")

            ax2 = plt.subplot(2, 3, 2)
            ax2.imshow(target_np, cmap="gray", vmin=0, vmax=255)
            ax2.set_title("GT Points", fontsize=12, fontweight="bold")
            ax2.axis("off")

            ax3 = plt.subplot(2, 3, 3)
            ax3.imshow(pred_img_white, cmap="gray", vmin=0, vmax=255)
            ax3.set_title(
                f"Output Points\nChamfer={nn_metrics['chamfer']:.4f}, CV={nn_metrics['cv']:.3f}",
                fontsize=11,
                fontweight="bold",
            )
            ax3.axis("off")

            # Row 2: Spectral comparisons
            if HAS_SPECTRAL_METRICS:
                # Spectral difference
                ax4 = plt.subplot(2, 3, 4)
                power_gt = np.abs(np.fft.fftshift(np.fft.fft2(target_np.astype(float) - target_np.mean()))) ** 2
                power_pred = np.abs(np.fft.fftshift(np.fft.fft2(pred_img_white.astype(float) - pred_img_white.mean()))) ** 2
                power_diff = np.abs(power_gt - power_pred)
                power_diff_log = np.log(power_diff + 1.0)
                ax4.imshow(power_diff_log, cmap="hot")
                ax4.set_title("Spectral Difference", fontsize=11, fontweight="bold")
                ax4.axis("off")

                # GT radial profile
                ax5 = plt.subplot(2, 3, 5)
                power_gt_log = np.log(power_gt + 1.0)
                ax5.imshow(power_gt_log, cmap="viridis")
                ax5.set_title("GT Power Spectrum", fontsize=11, fontweight="bold")
                ax5.axis("off")

                # Output radial profile
                ax6 = plt.subplot(2, 3, 6)
                power_pred_log = np.log(power_pred + 1.0)
                ax6.imshow(power_pred_log, cmap="viridis")
                ax6.set_title("Output Power Spectrum", fontsize=11, fontweight="bold")
                ax6.axis("off")

            plt.tight_layout()
            plt.savefig(out_dir / f"sample_{idx:03d}_detailed.png", dpi=150, bbox_inches="tight")
            plt.close()

    # Compute aggregate metrics
    print("\n" + "=" * 80)
    print("AGGREGATE METRICS ON TEST BATCH")
    print("=" * 80)
    
    if all_metrics["nn_metrics"]:
        mean_nn = np.mean([m["mean_nn"] for m in all_metrics["nn_metrics"]])
        std_nn = np.std([m["mean_nn"] for m in all_metrics["nn_metrics"]])
        mean_cv = np.mean([m["cv"] for m in all_metrics["nn_metrics"]])
        std_cv = np.std([m["cv"] for m in all_metrics["nn_metrics"]])
        mean_chamfer = np.mean([m["chamfer"] for m in all_metrics["nn_metrics"]])
        std_chamfer = np.std([m["chamfer"] for m in all_metrics["nn_metrics"]])

        print("\nNearest Neighbor Metrics:")
        print(f"  Mean NN: {mean_nn:.4f} ± {std_nn:.4f}")
        print(f"  CV (Uniformity): {mean_cv:.4f} ± {std_cv:.4f}")
        print(f"  Chamfer Distance: {mean_chamfer:.6f} ± {std_chamfer:.6f}")

    if all_metrics["spectral_metrics"]:
        mean_radial_diff = np.mean([m["radial_profile_diff"] for m in all_metrics["spectral_metrics"]])
        std_radial_diff = np.std([m["radial_profile_diff"] for m in all_metrics["spectral_metrics"]])
        print(f"\nSpectral Metrics:")
        print(f"  Radial Profile Difference: {mean_radial_diff:.6f} ± {std_radial_diff:.6f}")

    # Generate aggregate spectral visualization
    if all_metrics["radial_profiles_gt"] and all_metrics["radial_profiles_pred"]:
        fig = plt.figure(figsize=(16, 12))

        # Plot 1: Average radial profiles
        ax1 = plt.subplot(2, 2, 1)
        avg_radial_gt = np.mean(all_metrics["radial_profiles_gt"], axis=0)
        avg_radial_pred = np.mean(all_metrics["radial_profiles_pred"], axis=0)
        std_radial_gt = np.std(all_metrics["radial_profiles_gt"], axis=0)
        std_radial_pred = np.std(all_metrics["radial_profiles_pred"], axis=0)

        ax1.plot(avg_radial_gt, label="GT", linewidth=2.5, color="red")
        ax1.fill_between(range(len(avg_radial_gt)), avg_radial_gt - std_radial_gt, avg_radial_gt + std_radial_gt, alpha=0.2, color="red")
        ax1.plot(avg_radial_pred, label="Predicted", linewidth=2.5, color="blue")
        ax1.fill_between(range(len(avg_radial_pred)), avg_radial_pred - std_radial_pred, avg_radial_pred + std_radial_pred, alpha=0.2, color="blue")
        ax1.set_xlabel("Frequency Radius", fontsize=11)
        ax1.set_ylabel("Power", fontsize=11)
        ax1.set_title("Average Radial Power Profile (Mean ± Std)", fontsize=12, fontweight="bold")
        ax1.legend(fontsize=10)
        ax1.grid(True, alpha=0.3)

        # Plot 2: Radial profile difference
        ax2 = plt.subplot(2, 2, 2)
        radial_diff = np.abs(avg_radial_gt - avg_radial_pred)
        ax2.bar(range(len(radial_diff)), radial_diff, color="orange", alpha=0.7)
        ax2.set_xlabel("Frequency Radius", fontsize=11)
        ax2.set_ylabel("|Δ Power|", fontsize=11)
        ax2.set_title("Absolute Difference in Radial Power", fontsize=12, fontweight="bold")
        ax2.grid(True, alpha=0.3, axis="y")

        # Plot 3: NN metrics distribution
        ax3 = plt.subplot(2, 2, 3)
        nn_values = [m["mean_nn"] for m in all_metrics["nn_metrics"]]
        ax3.hist(nn_values, bins=10, alpha=0.7, color="green", edgecolor="black")
        ax3.axvline(np.mean(nn_values), color="red", linestyle="--", linewidth=2, label=f"Mean: {np.mean(nn_values):.4f}")
        ax3.set_xlabel("Mean NN Distance", fontsize=11)
        ax3.set_ylabel("Frequency", fontsize=11)
        ax3.set_title("Distribution of Mean NN Distances", fontsize=12, fontweight="bold")
        ax3.legend(fontsize=10)
        ax3.grid(True, alpha=0.3, axis="y")

        # Plot 4: Chamfer distance distribution
        ax4 = plt.subplot(2, 2, 4)
        chamfer_values = [m["chamfer"] for m in all_metrics["nn_metrics"]]
        ax4.hist(chamfer_values, bins=10, alpha=0.7, color="purple", edgecolor="black")
        ax4.axvline(np.mean(chamfer_values), color="red", linestyle="--", linewidth=2, label=f"Mean: {np.mean(chamfer_values):.6f}")
        ax4.set_xlabel("Chamfer Distance", fontsize=11)
        ax4.set_ylabel("Frequency", fontsize=11)
        ax4.set_title("Distribution of Chamfer Distances", fontsize=12, fontweight="bold")
        ax4.legend(fontsize=10)
        ax4.grid(True, alpha=0.3, axis="y")

        plt.tight_layout()
        plt.savefig(out_dir / "aggregate_spectral_analysis.png", dpi=150, bbox_inches="tight")
        plt.close()
        print(f"\nAggregate spectral analysis saved to: {out_dir / 'aggregate_spectral_analysis.png'}")

    # Save metrics to file
    metrics_file = out_dir / "metrics_summary.json"
    with open(metrics_file, "w") as f:
        summary = {
            "nn_metrics_mean": {
                "mean_nn": float(np.mean([m["mean_nn"] for m in all_metrics["nn_metrics"]])),
                "cv": float(np.mean([m["cv"] for m in all_metrics["nn_metrics"]])),
                "chamfer": float(np.mean([m["chamfer"] for m in all_metrics["nn_metrics"]])),
            },
            "num_samples": len(all_metrics["nn_metrics"]),
        }
        if all_metrics["spectral_metrics"]:
            summary["spectral_metrics_mean"] = {
                "radial_profile_diff": float(np.mean([m["radial_profile_diff"] for m in all_metrics["spectral_metrics"]])),
            }
        json.dump(summary, f, indent=2)

    print(f"\nResults saved to: {out_dir}")
    print(f"Metrics summary: {metrics_file}")


if __name__ == "__main__":
    main()
