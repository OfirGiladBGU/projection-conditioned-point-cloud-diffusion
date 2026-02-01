"""
PointDiT Multi-Sample Comparison: Train models with different losses, test on 8 samples
Creates grid visualization like multi_sample_comparison.png but using PointDiT models
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
from diffusion import DDPMScheduler, train_step, sample, ChamferLoss, SpectralBluenoiseLoss
from sinkhorn_lloyd_losses import SinkhornDensityLoss, SinkhornDensityLossSimple, HAS_GEOMLOSS


def compute_blue_noise_metrics(points: torch.Tensor, gt_points: torch.Tensor = None):
    """Compute blue-noise metrics."""
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

    scheduler = DDPMScheduler(
        num_train_timesteps=config.diffusion.num_train_timesteps,
        beta_start=config.diffusion.beta_start,
        beta_end=config.diffusion.beta_end,
        beta_schedule=config.diffusion.beta_schedule,
    )

    # Load 8 test samples
    NUM_SAMPLES = 8
    test_samples = []
    for i in range(NUM_SAMPLES):
        data = dataset[i]
        test_samples.append({
            "image": data["image"].unsqueeze(0).to(device),
            "gt_points": data["points"].unsqueeze(0).to(device),
        })
    
    num_points = test_samples[0]["gt_points"].shape[1]

    # Loss combinations to test, including the new spectral loss
    loss_combos = [
        {
            "name": "Chamfer Only",
            "chamfer_weight": 1.0,
            "sinkhorn_weight": 0.0,
            "spectral_weight": 0.0,
        },
        {
            "name": "Sinkhorn Only",
            "chamfer_weight": 0.0,
            "sinkhorn_weight": 1.0,
            "spectral_weight": 0.0,
        },
        {
            "name": "Sinkhorn+Chamfer (Hybrid)",
            "chamfer_weight": 1.0,
            "sinkhorn_weight": 1.0,
            "spectral_weight": 0.0,
        },
        {
            "name": "Sinkhorn+Chamfer+Spectral (New!)",
            "chamfer_weight": 1.0,
            "sinkhorn_weight": 1.0,
            "spectral_weight": 0.5,  # Moderate spectral loss weight
        },
        {
            "name": "Sinkhorn+Spectral (Spectral Focus)",
            "chamfer_weight": 0.0,
            "sinkhorn_weight": 1.0,
            "spectral_weight": 1.0,  # Higher spectral weight
        },
    ]

    out_dir = Path(__file__).parent / "outputs_pointdit_v6_comprehensive"
    out_dir.mkdir(exist_ok=True)

    steps = 1000
    all_predictions = []  # Store predictions for all samples and methods

    print("\n" + "="*80)
    print(f"Testing PointDiT on {NUM_SAMPLES} samples with {len(loss_combos)} loss configurations")
    print("="*80)

    # Train models and collect predictions
    for combo in loss_combos:
        print(f"\n=== Training PointDiT with {combo['name']} ===")
        
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
                spectral_weight=combo["spectral_weight"],
                use_adaptive_repulsion=False,
            )
            loss = loss_dict["loss"]
            optimizer.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()

            if step % 200 == 0 or step == steps - 1:
                log_msg = f"Step {step:4d} | loss={loss.item():.6f} " \
                          f"chamfer={loss_dict['chamfer']:.6f} " \
                          f"sinkhorn={loss_dict['sinkhorn']:.6f}"
                if combo["spectral_weight"] > 0:
                    log_msg += f" spectral={loss_dict['spectral']:.6f}"
                print(log_msg)

        # Test on all samples
        print(f"Testing {combo['name']} on {NUM_SAMPLES} samples...")
        model.eval()
        sample_predictions = []
        
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
                
                sample_predictions.append({
                    "pred_points": pred_points[0].cpu().numpy(),
                    "metrics": metrics,
                })
                
                print(f"  Sample {sample_idx}: NN={metrics['mean_nn']:.4f}, CV={metrics['cv']:.3f}, Chamfer={metrics['chamfer']:.6f}")
        
        all_predictions.append({
            "name": combo["name"],
            "predictions": sample_predictions,
        })

    # Create visualization grid: NUM_SAMPLES rows × (1 GT + len(loss_combos) predictions)
    num_cols = 1 + len(loss_combos)  # GT + predictions from each model
    fig, axes = plt.subplots(NUM_SAMPLES, num_cols, figsize=(num_cols * 4, NUM_SAMPLES * 4))
    
    print("\nGenerating visualization grid...")
    
    for i in range(NUM_SAMPLES):
        img = test_samples[i]["image"][0, 0].cpu().numpy()
        gt_pts = test_samples[i]["gt_points"][0].cpu().numpy()
        
        # Compute GT metrics
        gt_metrics = compute_blue_noise_metrics(test_samples[i]["gt_points"])
        
        # Column 0: Ground Truth
        axes[i, 0].imshow(img, cmap="gray", extent=[-1, 1, -1, 1], origin="lower")
        axes[i, 0].scatter(gt_pts[:, 0], gt_pts[:, 1], c="red", s=2, alpha=0.8)
        axes[i, 0].set_title(f"GT\nNN={gt_metrics['mean_nn']:.4f}\nCV={gt_metrics['cv']:.3f}", fontsize=10)
        axes[i, 0].set_xlim(-1, 1)
        axes[i, 0].set_ylim(-1, 1)
        axes[i, 0].set_aspect("equal")
        axes[i, 0].set_xticks([])
        axes[i, 0].set_yticks([])
        
        # Columns 1+: Predictions from each model
        for col_idx, result in enumerate(all_predictions):
            pred_pts = result["predictions"][i]["pred_points"]
            metrics = result["predictions"][i]["metrics"]
            
            axes[i, col_idx + 1].imshow(img, cmap="gray", extent=[-1, 1, -1, 1], origin="lower")
            axes[i, col_idx + 1].scatter(pred_pts[:, 0], pred_pts[:, 1], c="blue", s=2, alpha=0.8)
            axes[i, col_idx + 1].set_title(
                f"{result['name']}\nNN={metrics['mean_nn']:.4f}\nChamfer={metrics['chamfer']:.6f}",
                fontsize=10
            )
            axes[i, col_idx + 1].set_xlim(-1, 1)
            axes[i, col_idx + 1].set_ylim(-1, 1)
            axes[i, col_idx + 1].set_aspect("equal")
            axes[i, col_idx + 1].set_xticks([])
            axes[i, col_idx + 1].set_yticks([])
    
    plt.tight_layout()
    save_path = out_dir / "multi_sample_comparison.png"
    plt.savefig(save_path, dpi=150, bbox_inches="tight")
    plt.close()
    
    print(f"\n✓ Visualization saved to: {save_path}")
    
    # Print summary statistics
    print("\n" + "="*100)
    print("LOSS COMPARISON SUMMARY")
    print("="*100)
    print(f"{'Loss Configuration':<40} {'Mean NN':<15} {'CV':<15} {'Chamfer':<15}")
    print("-"*100)
    
    best_by_metric = {
        'mean_nn': (float('inf'), None),
        'cv': (float('inf'), None),
        'chamfer': (float('inf'), None),
    }
    
    for result in all_predictions:
        mean_nns = [p["metrics"]["mean_nn"] for p in result["predictions"]]
        cvs = [p["metrics"]["cv"] for p in result["predictions"]]
        chamfers = [p["metrics"]["chamfer"] for p in result["predictions"]]
        
        mean_nn_avg = np.mean(mean_nns)
        cv_avg = np.mean(cvs)
        chamfer_avg = np.mean(chamfers)
        
        print(f"{result['name']:<40} {mean_nn_avg:.6f}±{np.std(mean_nns):.4f}  {cv_avg:.5f}±{np.std(cvs):.4f}  {chamfer_avg:.8f}±{np.std(chamfers):.6f}")
        
        # Track best
        if mean_nn_avg < best_by_metric['mean_nn'][0]:
            best_by_metric['mean_nn'] = (mean_nn_avg, result['name'])
        if cv_avg < best_by_metric['cv'][0]:
            best_by_metric['cv'] = (cv_avg, result['name'])
        if chamfer_avg < best_by_metric['chamfer'][0]:
            best_by_metric['chamfer'] = (chamfer_avg, result['name'])
    
    print("="*100)
    print("\nBEST PERFORMERS:")
    print(f"  ✓ Lowest Mean NN (best spacing):  {best_by_metric['mean_nn'][1]} ({best_by_metric['mean_nn'][0]:.6f})")
    print(f"  ✓ Lowest CV (most uniform):      {best_by_metric['cv'][1]} ({best_by_metric['cv'][0]:.5f})")
    print(f"  ✓ Lowest Chamfer (best match):   {best_by_metric['chamfer'][1]} ({best_by_metric['chamfer'][0]:.8f})")
    print("="*100)
    
    # Detailed breakdown
    print("\nDETAILED BREAKDOWN BY LOSS:")
    for result in all_predictions:
        mean_nns = [p["metrics"]["mean_nn"] for p in result["predictions"]]
        cvs = [p["metrics"]["cv"] for p in result["predictions"]]
        chamfers = [p["metrics"]["chamfer"] for p in result["predictions"]]
        
        print(f"\n{result['name']}:")
        print(f"  Mean NN:  {np.mean(mean_nns):.6f} ± {np.std(mean_nns):.6f}")
        print(f"  CV:       {np.mean(cvs):.5f} ± {np.std(cvs):.5f}")
        print(f"  Chamfer:  {np.mean(chamfers):.8f} ± {np.std(chamfers):.8f}")


if __name__ == "__main__":
    main()
