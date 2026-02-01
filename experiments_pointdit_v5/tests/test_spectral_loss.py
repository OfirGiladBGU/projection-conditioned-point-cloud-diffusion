"""
Test Spectral Loss for PointDiT
Compares Sinkhorn+Chamfer with and without spectral loss component
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

    # Compare two approaches
    methods = [
        {
            "name": "Sinkhorn+Chamfer (baseline)",
            "chamfer_weight": 1.0,
            "sinkhorn_weight": 1.0,
            "spectral_weight": 0.0,
        },
        {
            "name": "Sinkhorn+Chamfer+Spectral (NEW)",
            "chamfer_weight": 1.0,
            "sinkhorn_weight": 1.0,
            "spectral_weight": 0.5,  # Start with moderate weight
        },
    ]

    out_dir = Path(__file__).parent / "outputs_pointdit_v5_comprehensive"
    out_dir.mkdir(exist_ok=True)

    steps = 1000
    all_predictions = []

    print("\n" + "="*80)
    print(f"Testing Spectral Loss on {NUM_SAMPLES} samples")
    print("="*80)

    # Train models and collect predictions
    for method in methods:
        print(f"\n=== Training PointDiT with {method['name']} ===")
        
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
                chamfer_weight=method["chamfer_weight"],
                sinkhorn_weight=method["sinkhorn_weight"],
                spectral_weight=method["spectral_weight"],
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
                    f"sinkhorn={loss_dict['sinkhorn']:.6f} "
                    f"spectral={loss_dict['spectral']:.6f}"
                )

        # Test on all samples
        print(f"Testing {method['name']} on {NUM_SAMPLES} samples...")
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
            "name": method["name"],
            "predictions": sample_predictions,
        })

    # Create side-by-side visualization: NUM_SAMPLES rows × (GT + baseline + spectral)
    num_cols = 1 + len(methods)
    fig, axes = plt.subplots(NUM_SAMPLES, num_cols, figsize=(num_cols * 4, NUM_SAMPLES * 4))
    
    print("\nGenerating comparison visualization...")
    
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
        
        # Columns 1+: Predictions from each method
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
    save_path = out_dir / "spectral_loss_comparison.png"
    plt.savefig(save_path, dpi=150, bbox_inches="tight")
    plt.close()
    
    print(f"\n✓ Comparison visualization saved to: {save_path}")
    
    # Print summary statistics
    print("\n" + "="*80)
    print("SUMMARY STATISTICS - Spectral Loss Impact")
    print("="*80)
    
    for result in all_predictions:
        mean_nns = [p["metrics"]["mean_nn"] for p in result["predictions"]]
        cvs = [p["metrics"]["cv"] for p in result["predictions"]]
        chamfers = [p["metrics"]["chamfer"] for p in result["predictions"]]
        
        print(f"\n{result['name']}:")
        print(f"  Mean NN: {np.mean(mean_nns):.4f} ± {np.std(mean_nns):.4f}")
        print(f"  CV: {np.mean(cvs):.3f} ± {np.std(cvs):.3f}")
        print(f"  Chamfer: {np.mean(chamfers):.6f} ± {np.std(chamfers):.6f}")
    
    # Calculate improvement
    baseline_cvs = [p["metrics"]["cv"] for p in all_predictions[0]["predictions"]]
    spectral_cvs = [p["metrics"]["cv"] for p in all_predictions[1]["predictions"]]
    
    baseline_chamfers = [p["metrics"]["chamfer"] for p in all_predictions[0]["predictions"]]
    spectral_chamfers = [p["metrics"]["chamfer"] for p in all_predictions[1]["predictions"]]
    
    cv_improvement = (np.mean(baseline_cvs) - np.mean(spectral_cvs)) / np.mean(baseline_cvs) * 100
    chamfer_improvement = (np.mean(baseline_chamfers) - np.mean(spectral_chamfers)) / np.mean(baseline_chamfers) * 100
    
    print("\n" + "="*80)
    print("IMPROVEMENT WITH SPECTRAL LOSS")
    print("="*80)
    print(f"CV improvement: {cv_improvement:+.2f}%")
    print(f"Chamfer improvement: {chamfer_improvement:+.2f}%")
    
    if cv_improvement > 0:
        print("✓ Spectral loss improved spacing uniformity!")
    else:
        print("✗ Spectral loss did NOT improve spacing")
    
    if chamfer_improvement > 0:
        print("✓ Spectral loss improved position accuracy!")
    else:
        print("✗ Spectral loss did NOT improve position accuracy")


if __name__ == "__main__":
    main()
