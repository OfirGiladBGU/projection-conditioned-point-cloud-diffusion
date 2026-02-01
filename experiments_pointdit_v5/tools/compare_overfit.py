"""Compare Point-DiT V4 vs V5 overfit results on a single sample.

Loads saved arrays from two sample directories, computes Chamfer distances
(between predictions and GT), and produces a combined visualization.

Usage:
    python experiments_pointdit_v5/tools/compare_overfit.py \
        --v4-sample-dir experiments_pointdit_v5/outputs_pointdit_v5_V4/sample_0 \
        --v5-sample-dir experiments_pointdit_v5/outputs_pointdit_v5_V5/sample_0 \
        --save-path experiments_pointdit_v5/outputs_pointdit_v5/comparison_V4_V5_sample_0.png
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


def chamfer_distance(p: torch.Tensor, q: torch.Tensor) -> torch.Tensor:
    """Chamfer distance between two point sets (batch-aware or single).
    If inputs are (N,2) tensors, adds a batch dimension.
    """
    if p.dim() == 2:
        p = p.unsqueeze(0)
    if q.dim() == 2:
        q = q.unsqueeze(0)
    dists = torch.cdist(p, q, p=2)
    min_pq = dists.min(dim=2).values
    min_qp = dists.min(dim=1).values
    loss = (min_pq.pow(2).mean(dim=1) + min_qp.pow(2).mean(dim=1)).mean()
    return loss


def load_required_files(sample_dir: str):
    def lf(name: str):
        path = os.path.join(sample_dir, name)
        if not os.path.exists(path):
            raise FileNotFoundError(f"Missing file: {path}")
        return np.load(path)

    image = lf("input_image.npy")  # (H,W)
    gt_norm = lf("gt_points_normalized.npy")  # shape (1,N,2)
    pred_norm = lf("pred_points_normalized.npy")  # shape (1,N,2)
    gt_pix = lf("gt_points_pixels.npy")  # shape (N,2)
    pred_pix = lf("pred_points_pixels.npy")  # shape (N,2)
    return image, gt_norm, pred_norm, gt_pix, pred_pix


def main():
    parser = argparse.ArgumentParser(description="Compare V4 vs V5 overfit outputs")
    parser.add_argument("--v4-sample-dir", type=str, required=True, help="Path to V4 sample directory")
    parser.add_argument("--v5-sample-dir", type=str, required=True, help="Path to V5 sample directory")
    parser.add_argument("--save-path", type=str, required=True, help="Output PNG path for combined visualization")
    args = parser.parse_args()

    # Load V4 and V5 data
    image_v4, gt_norm_v4, pred_norm_v4, gt_pix_v4, pred_pix_v4 = load_required_files(args.v4_sample_dir)
    image_v5, gt_norm_v5, pred_norm_v5, gt_pix_v5, pred_pix_v5 = load_required_files(args.v5_sample_dir)

    # Basic checks
    if image_v4.shape != image_v5.shape:
        print("Warning: image shapes differ between V4 and V5.")
    H, W = image_v5.shape

    # Use GT from V5 as reference (should match V4 if same sample)
    gt_norm = torch.from_numpy(gt_norm_v5.squeeze(0)).float()  # (N,2)
    pred_v4_norm = torch.from_numpy(pred_norm_v4.squeeze(0)).float()  # (N,2)
    pred_v5_norm = torch.from_numpy(pred_norm_v5.squeeze(0)).float()  # (N,2)

    # Compute Chamfer distances (normalized space)
    cd_v4 = chamfer_distance(pred_v4_norm, gt_norm).item()
    cd_v5 = chamfer_distance(pred_v5_norm, gt_norm).item()
    cd_v4_v5 = chamfer_distance(pred_v4_norm, pred_v5_norm).item()

    print("Comparison metrics (normalized coordinates):")
    print(f"  Chamfer(V4 pred, GT):  {cd_v4:.6f}")
    print(f"  Chamfer(V5 pred, GT):  {cd_v5:.6f}")
    print(f"  Chamfer(V4 pred, V5):  {cd_v4_v5:.6f}")

    # Also save metrics to JSON next to the figure for easy reading
    import json
    metrics = {
        "same_image_shape": image_v4.shape == image_v5.shape,
        "same_gt_norm": bool(np.allclose(gt_norm_v4, gt_norm_v5)),
        "cd_v4_vs_gt": cd_v4,
        "cd_v5_vs_gt": cd_v5,
        "cd_v4_vs_v5": cd_v4_v5,
        "H": int(H),
        "W": int(W),
    }
    metrics_path = os.path.splitext(args.save_path)[0] + "_metrics.json"
    os.makedirs(os.path.dirname(metrics_path), exist_ok=True)
    with open(metrics_path, "w") as f:
        json.dump(metrics, f, indent=2)
    print(f"Saved metrics to {metrics_path}")

    # Combined visualization
    if MATPLOTLIB_AVAILABLE:
        fig, axes = plt.subplots(1, 4, figsize=(24, 6))

        axes[0].imshow(image_v5, cmap="gray")
        axes[0].set_title("Input Image")
        axes[0].axis("off")

        axes[1].scatter(gt_pix_v5[:, 0], gt_pix_v5[:, 1], c="green", s=2, alpha=0.7)
        axes[1].set_xlim(0, W)
        axes[1].set_ylim(H, 0)
        axes[1].set_aspect("equal")
        axes[1].set_facecolor("white")
        axes[1].set_title(f"GT ({len(gt_pix_v5)} pts)")
        axes[1].axis("off")

        axes[2].scatter(pred_pix_v5[:, 0], pred_pix_v5[:, 1], c="red", s=2, alpha=0.7)
        axes[2].set_xlim(0, W)
        axes[2].set_ylim(H, 0)
        axes[2].set_aspect("equal")
        axes[2].set_facecolor("white")
        axes[2].set_title(f"V5 Pred ({len(pred_pix_v5)} pts)\nCD={cd_v5:.4f}")
        axes[2].axis("off")

        axes[3].scatter(pred_pix_v4[:, 0], pred_pix_v4[:, 1], c="blue", s=2, alpha=0.7)
        axes[3].set_xlim(0, W)
        axes[3].set_ylim(H, 0)
        axes[3].set_aspect("equal")
        axes[3].set_facecolor("white")
        axes[3].set_title(f"V4 Pred ({len(pred_pix_v4)} pts)\nCD={cd_v4:.4f}")
        axes[3].axis("off")

        plt.suptitle("Point-DiT Overfit Comparison: V5 vs V4", fontsize=14, fontweight="bold")
        plt.tight_layout()
        os.makedirs(os.path.dirname(args.save_path), exist_ok=True)
        plt.savefig(args.save_path, dpi=150, bbox_inches="tight")
        plt.close()
        print(f"Saved comparison figure to {args.save_path}")
    else:
        print("Matplotlib not available; skipping visualization.")


if __name__ == "__main__":
    main()
