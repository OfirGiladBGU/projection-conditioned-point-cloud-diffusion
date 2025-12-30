"""Overfitting + visualization for Point-DiT V5 on a single image.

Runs training on one dataset sample, performs sampling, saves arrays, and
creates a side-by-side comparison PNG in a sample-specific output folder.
"""

import argparse
import os
import numpy as np
import torch

try:
    import matplotlib.pyplot as plt
    MATPLOTLIB_AVAILABLE = True
except Exception:
    plt = None
    MATPLOTLIB_AVAILABLE = False

from config import Config
from dataset import SimpleImageDataset
from diffusion import DDPMScheduler, sample, train_step
from model import PointDiT


def visualize_sample(img_tensor: torch.Tensor, gt_pts: torch.Tensor, pred_pts: torch.Tensor, save_path: str, show: bool = False):
    """Save side-by-side visualization: input | GT | Pred.

    If show=True and matplotlib is available, display the figure as well.
    """
    img = img_tensor[0, 0].detach().cpu().numpy()
    H, W = img.shape
    gt = gt_pts[0].detach().cpu().numpy()
    pred = pred_pts[0].detach().cpu().numpy()

    gt_pix = np.stack([(gt[:, 0] + 1.0) * (W / 2.0), (gt[:, 1] + 1.0) * (H / 2.0)], axis=1)
    pred_pix = np.stack([(pred[:, 0] + 1.0) * (W / 2.0), (pred[:, 1] + 1.0) * (H / 2.0)], axis=1)

    if MATPLOTLIB_AVAILABLE:
        fig, axes = plt.subplots(1, 3, figsize=(18, 6))

        axes[0].imshow(img, cmap="gray")
        axes[0].set_title("Input Image")
        axes[0].axis("off")

        axes[1].scatter(gt_pix[:, 0], gt_pix[:, 1], c="green", s=1, alpha=0.7)
        axes[1].set_xlim(0, W)
        axes[1].set_ylim(H, 0)
        axes[1].set_aspect("equal")
        axes[1].set_facecolor("white")
        axes[1].set_title(f"GT Stippling ({len(gt_pix)} points)")
        axes[1].axis("off")

        axes[2].scatter(pred_pix[:, 0], pred_pix[:, 1], c="red", s=1, alpha=0.7)
        axes[2].set_xlim(0, W)
        axes[2].set_ylim(H, 0)
        axes[2].set_aspect("equal")
        axes[2].set_facecolor("white")
        axes[2].set_title(f"Predicted Stippling ({len(pred_pix)} points)")
        axes[2].axis("off")

        plt.tight_layout()
        plt.savefig(save_path, dpi=150, bbox_inches="tight")
        print(f"Saved visualization to {save_path}")
        if show:
            plt.show()
        plt.close()
    else:
        np.save(save_path.replace(".png", "_image.npy"), img)
        np.save(save_path.replace(".png", "_gt.npy"), gt_pix)
        np.save(save_path.replace(".png", "_pred.npy"), pred_pix)
        print("Matplotlib not available, saved arrays instead")


def main():
    parser = argparse.ArgumentParser(description="Overfit Point-DiT V5 on a single sample and visualize")
    parser.add_argument("--steps", type=int, default=400, help="Training steps")
    parser.add_argument("--inference-steps", type=int, default=150, help="Sampling steps")
    parser.add_argument("--sample-index", type=int, default=0, help="Dataset sample index")
    parser.add_argument("--lr", type=float, default=2e-4, help="Learning rate")
    parser.add_argument("--seed", type=int, default=42, help="Random seed")
    parser.add_argument("--show", action="store_true", help="Display matplotlib window after saving PNG")
    args = parser.parse_args()

    # Seeds and device
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    torch.cuda.manual_seed_all(args.seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

    config = Config.default()
    device = "cuda" if torch.cuda.is_available() else "cpu"

    model = PointDiT(
        n_points=config.model.n_points,
        dim=config.model.dim,
        n_layers=config.model.n_layers,
        n_heads=config.model.n_heads,
        image_size=config.model.image_size,
        dropout=0.0,
    ).to(device)

    print("Point-DiT V5 model created")
    print(f"Model parameters: {model.get_num_params():,}")

    scheduler = DDPMScheduler(
        num_train_timesteps=config.diffusion.num_train_timesteps,
        beta_start=config.diffusion.beta_start,
        beta_end=config.diffusion.beta_end,
        beta_schedule=config.diffusion.beta_schedule,
    )

    if not os.path.isdir(config.data.source_dir):
        raise FileNotFoundError(f"Source dir not found: {config.data.source_dir}")

    dataset = SimpleImageDataset(
        source_dir=config.data.source_dir,
        target_dir=config.data.target_dir,
        image_size=config.data.image_size,
        num_points=config.model.n_points,
    )

    if args.sample_index < 0 or args.sample_index >= len(dataset):
        raise IndexError(f"sample_index {args.sample_index} out of range for dataset size {len(dataset)}")

    sample_data = dataset[args.sample_index]
    image = sample_data["image"].unsqueeze(0).to(device)
    points = sample_data["points"].unsqueeze(0).to(device)

    output_dir = os.path.join(os.path.dirname(__file__), "outputs_pointdit_v5")
    os.makedirs(output_dir, exist_ok=True)

    print("\n" + "=" * 60)
    print("OVERFITTING TEST: PointDiT V5")
    print("=" * 60)
    print(f"Image shape: {image.shape}")
    print(f"Points shape: {points.shape}")
    print(f"Points range: [{points.min():.3f}, {points.max():.3f}]")
    print(f"Sample index: {args.sample_index} from dataset (total {len(dataset)})")

    # Create sample-specific subdirectory
    sample_dir = os.path.join(output_dir, f"sample_{args.sample_index}")
    os.makedirs(sample_dir, exist_ok=True)

    # Save input image for reproducibility
    img_np = image[0, 0].detach().cpu().numpy()
    np.save(os.path.join(sample_dir, "input_image.npy"), img_np)

    # Optimizer
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=1e-5)

    print("\nTraining on single sample:")
    print(f"{'Step':>5} {'Loss':>12}")
    print("-" * 20)

    for step in range(args.steps):
        model.train()
        loss = train_step(
            model,
            scheduler,
            points,
            image,
            device,
        )

        optimizer.zero_grad()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        optimizer.step()

        if step % 25 == 0 or step == args.steps - 1:
            print(f"{step:5d} {loss.item():12.6f}")

    print("\nNow sampling from the trained model:")
    model.eval()
    with torch.no_grad():
        sampled_points = sample(
            model,
            scheduler,
            image,
            config.model.n_points,
            args.inference_steps,
            device,
            show_progress=False,
            init_from_density=True,
            init_std=0.02,
            eta=0.0,
        )

    # Save normalized coordinates
    np.save(os.path.join(sample_dir, "gt_points_normalized.npy"), points.detach().cpu().numpy())
    np.save(os.path.join(sample_dir, "pred_points_normalized.npy"), sampled_points.detach().cpu().numpy())

    # Save pixel coordinates for visualization
    H, W = img_np.shape
    gt = points[0].detach().cpu().numpy()
    pred = sampled_points[0].detach().cpu().numpy()
    gt_pix = np.stack([(gt[:, 0] + 1.0) * (W / 2.0), (gt[:, 1] + 1.0) * (H / 2.0)], axis=1)
    pred_pix = np.stack([(pred[:, 0] + 1.0) * (W / 2.0), (pred[:, 1] + 1.0) * (H / 2.0)], axis=1)
    np.save(os.path.join(sample_dir, "gt_points_pixels.npy"), gt_pix)
    np.save(os.path.join(sample_dir, "pred_points_pixels.npy"), pred_pix)

    # Create comparison visualization (and optionally display)
    vis_path = os.path.join(sample_dir, "comparison.png")
    visualize_sample(image, points, sampled_points, vis_path, show=args.show)
    print(f"\n✓ Results saved to: {sample_dir}/")


if __name__ == "__main__":
    main()
