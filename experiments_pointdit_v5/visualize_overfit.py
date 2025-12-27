"""Visualize saved overfit results (supports index-specific files)."""

import argparse
import numpy as np
import matplotlib.pyplot as plt
import os

parser = argparse.ArgumentParser(description="Visualize Point-DiT overfit outputs")
from pathlib import Path
default_dir = Path(__file__).resolve().parent / "outputs_pointdit_v5"
parser.add_argument("--output-dir", type=str, default=str(default_dir), help="Directory with saved outputs")
parser.add_argument("--index", type=int, required=True, help="Sample index used during overfit run")
args = parser.parse_args()

output_dir = args.output_dir
sample_dir = os.path.join(output_dir, f"sample_{args.index}")

def load_file(filename):
    path = os.path.join(sample_dir, filename)
    if not os.path.exists(path):
        raise FileNotFoundError(f"Missing file: {path}\nEnsure test_overfit.py ran for sample {args.index}")
    return np.load(path)

# Load data (image in pixel space, points in pixel coords)
try:
    image = load_file("input_image.npy")
    gt_pts = load_file("gt_points_pixels.npy")
    pred_pts = load_file("pred_points_pixels.npy")
except FileNotFoundError as e:
    raise FileNotFoundError(f"Missing visualization files in {sample_dir}.\nError: {e}")

print(f"Image shape: {image.shape}")
print(f"GT points shape: {gt_pts.shape}")
print(f"Pred points shape: {pred_pts.shape}")
print(f"GT points range: [{gt_pts.min():.1f}, {gt_pts.max():.1f}]")
print(f"Pred points range: [{pred_pts.min():.1f}, {pred_pts.max():.1f}]")

H, W = image.shape

fig, axes = plt.subplots(1, 3, figsize=(18, 6))

# Input image
axes[0].imshow(image, cmap='gray')
axes[0].set_title('Input Image')
axes[0].axis('off')

# GT stippling
axes[1].scatter(gt_pts[:, 0], gt_pts[:, 1], c='green', s=2, alpha=0.7)
axes[1].set_xlim(0, W)
axes[1].set_ylim(H, 0)
axes[1].set_aspect('equal')
axes[1].set_facecolor('white')
axes[1].set_title(f'GT Stippling ({len(gt_pts)} points)')
axes[1].axis('off')

# Predicted stippling
axes[2].scatter(pred_pts[:, 0], pred_pts[:, 1], c='red', s=2, alpha=0.7)
axes[2].set_xlim(0, W)
axes[2].set_ylim(H, 0)
axes[2].set_aspect('equal')
axes[2].set_facecolor('white')
axes[2].set_title(f'Predicted Stippling ({len(pred_pts)} points)')
axes[2].axis('off')

plt.suptitle('Point-DiT Overfit Test Results', fontsize=14, fontweight='bold')
plt.tight_layout()

# Display only, comparison.png already saved by test_overfit.py
print(f"\n✓ Visualization ready (see {sample_dir}/comparison.png)")
plt.show()
