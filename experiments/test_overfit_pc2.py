"""Overfitting test for PC^2 Conditional Point Cloud Diffusion on a single image.

This script mirrors experiments_pointdit_v5/test_overfit.py but adapts to the
PC^2 model and conditioning pipeline. It trains on one (image, target) pair
and then samples a point cloud to visually compare.
"""

import argparse
import os
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn

try:
    import matplotlib.pyplot as plt
    MATPLOTLIB_AVAILABLE = True
except Exception:
    plt = None
    MATPLOTLIB_AVAILABLE = False

# Local imports from experiments
from model.model import ConditionalPointCloudDiffusionModel
from pytorch3d.structures import Pointclouds
from pytorch3d.renderer.cameras import PerspectiveCameras


def to_normalized_xy(points_pix: np.ndarray, H: int, W: int) -> np.ndarray:
    """Convert pixel coordinates (x,y) to normalized camera-plane xy in [-1, 1].

    Assumes origin at image center, x right, y down in pixels.
    """
    x = (points_pix[:, 0] - (W / 2.0)) / (W / 2.0)
    y = (points_pix[:, 1] - (H / 2.0)) / (H / 2.0)
    return np.stack([x, y], axis=1)


def extract_points_from_binary(target_img: np.ndarray, threshold: int = 128) -> np.ndarray:
    """Extract 2D point coordinates (pixels) from a binary/monochrome image.

    Returns array of shape (N, 2) with (x, y) pixel coordinates.
    """
    coords = np.where(target_img < threshold)
    if len(coords[0]) == 0:
        return np.zeros((0, 2), dtype=np.float32)
    # x = column index, y = row index
    return np.stack([coords[1], coords[0]], axis=1).astype(np.float32)


def make_camera(batch_size: int, image_size: int, device: torch.device) -> PerspectiveCameras:
    """Create a simple forward-facing perspective camera for conditioning."""
    R = torch.eye(3, device=device).unsqueeze(0).repeat(batch_size, 1, 1)
    T = torch.tensor([[0.0, 0.0, 2.0]], dtype=torch.float32, device=device).repeat(batch_size, 1)
    # image_size expects ((H, W),) per sample; use same for batch by repeating
    image_sizes = tuple((image_size, image_size) for _ in range(batch_size))
    cameras = PerspectiveCameras(R=R, T=T, image_size=image_sizes, device=device)
    return cameras


def visualize_side_by_side(img: np.ndarray, gt_pts_pix: np.ndarray, pred_pts: np.ndarray, save_path: str):
    """Save side-by-side visualization: input | GT | Pred.

    - img: (H, W) float array in [0,1]
    - gt_pts_pix: (N, 2) in pixel coordinates
    - pred_pts: (N, 3) predicted points in camera coords; we project XY to pixels via min-max normalization for quick look
    """
    H, W = img.shape
    if pred_pts.shape[1] < 2:
        raise ValueError("predicted points must have at least XY coordinates")

    # Simple normalization of predicted XY to pixel grid for visualization
    x_pred = pred_pts[:, 0]
    y_pred = pred_pts[:, 1]
    x_min, x_max = float(x_pred.min()), float(x_pred.max())
    y_min, y_max = float(y_pred.min()), float(y_pred.max())
    if x_max - x_min < 1e-6:
        x_pix = np.full_like(x_pred, W / 2.0)
    else:
        x_pix = (x_pred - x_min) / (x_max - x_min) * (W - 1)
    if y_max - y_min < 1e-6:
        y_pix = np.full_like(y_pred, H / 2.0)
    else:
        y_pix = (y_pred - y_min) / (y_max - y_min) * (H - 1)
    pred_pts_pix = np.stack([x_pix, y_pix], axis=1)

    if MATPLOTLIB_AVAILABLE:
        fig, axes = plt.subplots(1, 3, figsize=(18, 6))

        axes[0].imshow(img, cmap="gray")
        axes[0].set_title("Input Image")
        axes[0].axis("off")

        axes[1].scatter(gt_pts_pix[:, 0], gt_pts_pix[:, 1], c="green", s=1, alpha=0.7)
        axes[1].set_xlim(0, W)
        axes[1].set_ylim(H, 0)
        axes[1].set_aspect("equal")
        axes[1].set_facecolor("white")
        axes[1].set_title("GT Stippling")
        axes[1].axis("off")

        axes[2].scatter(pred_pts_pix[:, 0], pred_pts_pix[:, 1], c="red", s=1, alpha=0.7)
        axes[2].set_xlim(0, W)
        axes[2].set_ylim(H, 0)
        axes[2].set_aspect("equal")
        axes[2].set_facecolor("white")
        axes[2].set_title("Predicted Stippling")
        axes[2].axis("off")

        plt.tight_layout()
        plt.savefig(save_path, dpi=150, bbox_inches="tight")
        plt.close()
        print(f"Saved visualization to {save_path}")
    else:
        np.save(save_path.replace(".png", "_image.npy"), img)
        np.save(save_path.replace(".png", "_gt.npy"), gt_pts_pix)
        np.save(save_path.replace(".png", "_pred.npy"), pred_pts_pix)
        print("Matplotlib not available, saved arrays instead")


def main():
    parser = argparse.ArgumentParser(description="Overfit PC^2 on a single sample")
    parser.add_argument("--source-dir", type=str, required=True, help="Directory with source grayscale PNGs")
    parser.add_argument("--target-dir", type=str, required=True, help="Directory with target binary PNGs")
    parser.add_argument("--sample-index", type=int, default=0, help="Sample index to overfit")
    parser.add_argument("--image-size", type=int, default=512, help="Image size to use")
    parser.add_argument("--steps", type=int, default=400, help="Training steps")
    parser.add_argument("--inference-steps", type=int, default=150, help="Sampling steps")
    parser.add_argument("--lr", type=float, default=2e-4, help="Learning rate")
    parser.add_argument("--use-cnn-extractor", action="store_true", help="Use SimpleCNN feature extractor (grayscale)")
    parser.add_argument("--seed", type=int, default=42, help="Random seed")
    args = parser.parse_args()

    # Seeds and device
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    torch.cuda.manual_seed_all(args.seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # List files and pick sample
    source_dir = Path(args.source_dir)
    target_dir = Path(args.target_dir)
    source_files = sorted(list(source_dir.glob("*.png")))
    samples = []
    for src in source_files:
        tgt = target_dir / src.name
        if tgt.exists():
            samples.append((src, tgt))
    if len(samples) == 0:
        raise FileNotFoundError(f"No matched samples in {source_dir} and {target_dir}")
    if not (0 <= args.sample_index < len(samples)):
        raise IndexError(f"sample_index {args.sample_index} out of range for dataset size {len(samples)}")

    src_path, tgt_path = samples[args.sample_index]

    # Load source image (grayscale)
    from PIL import Image
    img = Image.open(src_path).convert("L").resize((args.image_size, args.image_size), Image.Resampling.BILINEAR)
    img_np = np.array(img, dtype=np.float32) / 255.0  # (H, W)
    if args.use_cnn_extractor:
        img_tensor = torch.from_numpy(img_np[np.newaxis, :, :])  # [1, H, W]
        image_color_channels = 1
        use_grayscale_norm = True
    else:
        img_tensor = torch.from_numpy(np.stack([img_np] * 3, axis=0))  # [3, H, W]
        image_color_channels = 3
        use_grayscale_norm = False
    image = img_tensor.unsqueeze(0).to(device)  # [B=1, C, H, W]

    # Load target image (binary) and extract points
    tgt_img = Image.open(tgt_path).convert("L").resize((args.image_size, args.image_size), Image.Resampling.NEAREST)
    tgt_np = np.array(tgt_img, dtype=np.uint8)
    pts_pix = extract_points_from_binary(tgt_np)  # (N, 2)
    if pts_pix.shape[0] == 0:
        raise RuntimeError(f"No points found in target image: {tgt_path}")
    pts_xy_norm = to_normalized_xy(pts_pix, H=args.image_size, W=args.image_size)  # (N, 2)
    pts_z = np.ones((pts_xy_norm.shape[0], 1), dtype=np.float32)  # Z=1 for visibility
    pts_3d = np.concatenate([pts_xy_norm, pts_z], axis=1).astype(np.float32)  # (N, 3)

    # Wrap points in Pointclouds and move to device
    pc = Pointclouds(points=[torch.from_numpy(pts_3d).to(device)])

    # Create simple mask and camera
    mask = torch.ones(1, 1, args.image_size, args.image_size, device=device)
    camera = make_camera(batch_size=1, image_size=args.image_size, device=device)

    # Create model
    model = ConditionalPointCloudDiffusionModel(
        image_size=args.image_size,
        image_feature_model="vit_small_patch16_224_msn",
        use_local_colors=True,
        use_local_features=True,
        use_global_features=False,
        use_mask=True,
        use_distance_transform=True,
        use_grayscale_normalization=use_grayscale_norm,
        use_cnn_extractor=args.use_cnn_extractor,
        image_color_channels=image_color_channels,
        scale_factor=1.0,
        colors_mean=0.5,
        colors_std=0.5,
        color_channels=3,
        predict_shape=True,
        predict_color=False,
        beta_start=1e-5,
        beta_end=8e-3,
        beta_schedule="linear",
        loss_xy_only=True,
        point_cloud_model="simple",
        point_cloud_model_embed_dim=64,
    ).to(device)

    print("PC^2 ConditionalPointCloudDiffusionModel created")
    # ModelMixin doesn't expose get_num_params; use sum
    num_params = sum(p.numel() for p in model.parameters())
    print(f"Model parameters: {num_params:,}")

    # Optimizer
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=1e-5)

    # Output dirs
    output_dir = Path(os.path.join(os.path.dirname(__file__), "outputs_pc2_overfit"))
    sample_dir = output_dir / f"sample_{args.sample_index}"
    sample_dir.mkdir(parents=True, exist_ok=True)

    # Persist input image
    np.save(sample_dir / "input_image.npy", img_np)

    # Training loop on single sample
    print("\n" + "=" * 60)
    print("OVERFITTING TEST: PC^2 (Conditional Diffusion)")
    print("=" * 60)
    print(f"Image shape: {image.shape}")
    print(f"GT points (3D) shape: {pts_3d.shape}")
    print(f"GT points XY range: x[{pts_3d[:,0].min():.3f},{pts_3d[:,0].max():.3f}] y[{pts_3d[:,1].min():.3f},{pts_3d[:,1].max():.3f}] z={pts_3d[:,2].mean():.3f}")
    print(f"Sample index: {args.sample_index} of {len(samples)}")

    print("\nTraining PC^2 on single sample:")
    print(f"{'Step':>5} {'Loss':>12}")
    print("-" * 20)

    for step in range(args.steps):
        model.train()
        # Forward loss
        loss = model.forward_train(
            pc=pc,
            camera=camera,
            image_rgb=image,
            mask=mask,
        )

        optimizer.zero_grad()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        optimizer.step()

        if step % 25 == 0 or step == args.steps - 1:
            print(f"{step:5d} {loss.item():12.6f}")

    # Sampling
    print("\nNow sampling from the trained model:")
    model.eval()
    with torch.no_grad():
        pred_pc = model.forward_sample(
            num_points=pts_3d.shape[0],
            camera=camera,
            image_rgb=image,
            mask=mask,
            scheduler="ddpm",
            num_inference_steps=args.inference_steps,
            eta=0.0,
            disable_tqdm=True,
        )

    # Save normalized coordinates and simple pixel projections
    gt_pts_norm = pts_3d.astype(np.float32)
    pred_pts = pred_pc.points_list()[0].detach().cpu().numpy().astype(np.float32)
    np.save(sample_dir / "gt_points_normalized.npy", gt_pts_norm)
    np.save(sample_dir / "pred_points_normalized.npy", pred_pts)

    # Pixel coordinate versions for quick visualization
    gt_pix = pts_pix.astype(np.float32)
    np.save(sample_dir / "gt_points_pixels.npy", gt_pix)

    # Save side-by-side visualization
    vis_path = sample_dir / "comparison.png"
    visualize_side_by_side(img_np, gt_pix, pred_pts, str(vis_path))
    print(f"\n✓ Results saved to: {sample_dir}/")


if __name__ == "__main__":
    main()
