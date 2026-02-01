"""
Overfit PointDiT on a single sample for each loss combo from test_multi_sample.py,
then evaluate the same metrics on model predictions.
"""

import pathlib
import sys

sys.path.append(str(pathlib.Path(__file__).parent.parent))

from pathlib import Path
import numpy as np
import torch
import matplotlib.pyplot as plt

from config import Config
from dataset import SimpleImageDataset
from model import PointDiT
from diffusion import DDPMScheduler, train_step, sample, ChamferLoss
from sinkhorn_lloyd_losses import SinkhornDensityLoss, SinkhornDensityLossSimple, HAS_GEOMLOSS


def compute_blue_noise_metrics(points: torch.Tensor, gt_points: torch.Tensor = None):
    """Compute blue-noise metrics (same as test_multi_sample.py)."""
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

    # Load multiple test samples
    NUM_TEST_SAMPLES = 6
    test_samples = []
    for i in range(NUM_TEST_SAMPLES):
        data = dataset[i]
        test_samples.append({
            "image": data["image"].unsqueeze(0).to(device),
            "gt_points": data["points"].unsqueeze(0).to(device),
        })
    
    num_points = test_samples[0]["gt_points"].shape[1]

    scheduler = DDPMScheduler(
        num_train_timesteps=config.diffusion.num_train_timesteps,
        beta_start=config.diffusion.beta_start,
        beta_end=config.diffusion.beta_end,
        beta_schedule=config.diffusion.beta_schedule,
    )

    if HAS_GEOMLOSS:
        sinkhorn_metric = SinkhornDensityLoss(blur=0.01, grid_size=64).to(device)
    else:
        sinkhorn_metric = SinkhornDensityLossSimple(grid_size=32, blur=0.05).to(device)

    loss_combos = [
        {"name": "Chamfer Only", "chamfer_weight": 1.0, "sinkhorn_weight": 0.0},
        {"name": "Sinkhorn Only", "chamfer_weight": 0.0, "sinkhorn_weight": 1.0},
        {"name": "Sinkhorn+Chamfer", "chamfer_weight": 1.0, "sinkhorn_weight": 1.0},
    ]

    out_dir = Path(__file__).parent / "outputs_pointdit_v5"
    out_dir.mkdir(exist_ok=True)

    results = []
    steps = 1000

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

        # Train on first sample
        image = test_samples[0]["image"]
        gt_points = test_samples[0]["gt_points"]

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

        # Test on all samples
        model.eval()
        sample_results = []
        
        with torch.no_grad():
            for sample_idx, sample_data in enumerate(test_samples):
                image = sample_data["image"]
                gt_points = sample_data["gt_points"]
                
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

                metrics = compute_blue_noise_metrics(pred_points, gt_points)
                sinkhorn_val = sinkhorn_metric(pred_points, image).item()
                
                sample_results.append({
                    "image": image[0, 0].cpu().numpy(),
                    "gt_points": gt_points[0].cpu().numpy(),
                    "pred_points": pred_points[0].cpu().numpy(),
                    "metrics": metrics,
                    "sinkhorn": sinkhorn_val,
                })
                
                if sample_idx == 0:  # Print first sample metrics
                    print(f"\nResults for {combo['name']} (Sample 0):")
                    print(f"  Mean NN: {metrics['mean_nn']:.4f}")
                    print(f"  CV: {metrics['cv']:.4f}")
                    print(f"  Chamfer: {metrics['chamfer']:.6f}")
                    print(f"  Sinkhorn: {sinkhorn_val:.6f}")

        # Compute average metrics
        avg_metrics = {
            "mean_nn": np.mean([r["metrics"]["mean_nn"] for r in sample_results]),
            "cv": np.mean([r["metrics"]["cv"] for r in sample_results]),
            "chamfer": np.mean([r["metrics"]["chamfer"] for r in sample_results]),
        }
        avg_sinkhorn = np.mean([r["sinkhorn"] for r in sample_results])
        
        print(f"\nAverage across {NUM_TEST_SAMPLES} samples:")
        print(f"  Mean NN: {avg_metrics['mean_nn']:.4f}")
        print(f"  CV: {avg_metrics['cv']:.4f}")
        print(f"  Chamfer: {avg_metrics['chamfer']:.6f}")
        print(f"  Sinkhorn: {avg_sinkhorn:.6f}")

        results.append({
            "name": combo["name"],
            "sample_results": sample_results,
            "avg_metrics": avg_metrics,
            "avg_sinkhorn": avg_sinkhorn,
        })

        # Create grid visualization for this loss combo
        fig, axes = plt.subplots(NUM_TEST_SAMPLES, 3, figsize=(15, NUM_TEST_SAMPLES * 4))
        
        for i, sample_result in enumerate(sample_results):
            img = sample_result["image"]
            gt = sample_result["gt_points"]
            pred = sample_result["pred_points"]
            metrics = sample_result["metrics"]
            
            # Column 0: Ground Truth
            axes[i, 0].imshow(img, cmap="gray", extent=[-1, 1, -1, 1], origin="lower")
            axes[i, 0].scatter(gt[:, 0], gt[:, 1], c="red", s=2, alpha=0.8)
            axes[i, 0].set_title(f"Sample {i} - Ground Truth", fontsize=10)
            axes[i, 0].set_xlim(-1, 1)
            axes[i, 0].set_ylim(-1, 1)
            axes[i, 0].set_aspect("equal")
            axes[i, 0].set_xticks([])
            axes[i, 0].set_yticks([])
            
            # Column 1: Prediction
            axes[i, 1].imshow(img, cmap="gray", extent=[-1, 1, -1, 1], origin="lower")
            axes[i, 1].scatter(pred[:, 0], pred[:, 1], c="blue", s=2, alpha=0.8)
            axes[i, 1].set_title(
                f"PointDiT: {combo['name']}\nNN={metrics['mean_nn']:.4f}, CV={metrics['cv']:.3f}",
                fontsize=10
            )
            axes[i, 1].set_xlim(-1, 1)
            axes[i, 1].set_ylim(-1, 1)
            axes[i, 1].set_aspect("equal")
            axes[i, 1].set_xticks([])
            axes[i, 1].set_yticks([])
            
            # Column 2: Overlay
            axes[i, 2].imshow(img, cmap="gray", extent=[-1, 1, -1, 1], origin="lower")
            axes[i, 2].scatter(gt[:, 0], gt[:, 1], c="red", s=1.5, alpha=0.6, label="GT")
            axes[i, 2].scatter(pred[:, 0], pred[:, 1], c="blue", s=1.5, alpha=0.6, label="Pred")
            axes[i, 2].set_title(
                f"Overlay - Chamfer={metrics['chamfer']:.6f}",
                fontsize=10
            )
            axes[i, 2].legend(loc="upper right", fontsize=8)
            axes[i, 2].set_xlim(-1, 1)
            axes[i, 2].set_ylim(-1, 1)
            axes[i, 2].set_aspect("equal")
            axes[i, 2].set_xticks([])
            axes[i, 2].set_yticks([])

        plt.suptitle(
            f"PointDiT - {combo['name']}\nAvg: NN={avg_metrics['mean_nn']:.4f}, CV={avg_metrics['cv']:.3f}, Chamfer={avg_metrics['chamfer']:.6f}",
            fontsize=14,
            fontweight="bold"
        )
        plt.tight_layout()
        filename = f"pointdit_{combo['name'].replace(' ', '_').replace('+', 'plus').lower()}_grid.png"
        plt.savefig(out_dir / filename, dpi=150, bbox_inches="tight")
        plt.close()
        print(f"  Grid visualization saved to: {out_dir / filename}")

    print("\n" + "="*80)
    print("SUMMARY TABLE - Average Metrics Across All Samples")
    print("="*80)
    print(f"{'Method':<20} {'Mean NN':<12} {'CV':<12} {'Chamfer':<12} {'Sinkhorn':<12}")
    print("-" * 72)
    for result in results:
        print(
            f"{result['name']:<20} "
            f"{result['avg_metrics']['mean_nn']:<12.4f} "
            f"{result['avg_metrics']['cv']:<12.4f} "
            f"{result['avg_metrics']['chamfer']:<12.6f} "
            f"{result['avg_sinkhorn']:<12.6f}"
        )


if __name__ == "__main__":
    main()
