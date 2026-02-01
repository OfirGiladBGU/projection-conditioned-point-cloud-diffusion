"""Validate that split_scripts (test_overfit.py + visualize_overfit.py) 
produces the same results as single_script (test_overfit_pointdit.py).
"""

import os
import numpy as np
import torch
import json

def chamfer_distance(p: torch.Tensor, q: torch.Tensor) -> torch.Tensor:
    """Chamfer distance between two point sets."""
    if p.dim() == 2:
        p = p.unsqueeze(0)
    if q.dim() == 2:
        q = q.unsqueeze(0)
    dists = torch.cdist(p, q, p=2)
    min_pq = dists.min(dim=2).values
    min_qp = dists.min(dim=1).values
    loss = (min_pq.pow(2).mean(dim=1) + min_qp.pow(2).mean(dim=1)).mean()
    return loss

def load_sample(sample_dir: str):
    """Load sample outputs."""
    def load(name):
        path = os.path.join(sample_dir, name)
        if not os.path.exists(path):
            raise FileNotFoundError(f"Missing: {path}")
        return np.load(path)
    
    image = load("input_image.npy")
    gt_norm = load("gt_points_normalized.npy")
    pred_norm = load("pred_points_normalized.npy")
    gt_pix = load("gt_points_pixels.npy")
    pred_pix = load("pred_points_pixels.npy")
    return image, gt_norm, pred_norm, gt_pix, pred_pix

print("=" * 70)
print("VALIDATION: split_scripts vs single_script")
print("=" * 70)

# Load both
split_image, split_gt_norm, split_pred_norm, split_gt_pix, split_pred_pix = load_sample("split_scripts/sample_0")
single_image, single_gt_norm, single_pred_norm, single_gt_pix, single_pred_pix = load_sample("single_script/sample_0")

# Compare images
images_match = np.allclose(split_image, single_image, rtol=1e-5)
print(f"\nImages match: {images_match}")

# Compare GT
gt_match = np.allclose(split_gt_norm, single_gt_norm, rtol=1e-5)
print(f"GT points match: {gt_match}")

# Compare predictions
pred_match = np.allclose(split_pred_norm, single_pred_norm, rtol=1e-5)
print(f"Pred points match (exact): {pred_match}")

# Compute distances to GT for each
split_gt = torch.from_numpy(split_gt_norm.squeeze(0)).float()
split_pred = torch.from_numpy(split_pred_norm.squeeze(0)).float()
single_gt = torch.from_numpy(single_gt_norm.squeeze(0)).float()
single_pred = torch.from_numpy(single_pred_norm.squeeze(0)).float()

split_cd = chamfer_distance(split_pred, split_gt).item()
single_cd = chamfer_distance(single_pred, single_gt).item()

print(f"\nChamfer distances to GT:")
print(f"  Split scripts:  {split_cd:.6f}")
print(f"  Single script:  {single_cd:.6f}")
print(f"  Difference:     {abs(split_cd - single_cd):.6f}")

# Summary
print("\n" + "=" * 70)
if images_match and gt_match:
    if abs(split_cd - single_cd) < 1e-4:
        print("✓ RESULTS ARE ESSENTIALLY IDENTICAL (same data, same quality)")
    else:
        print("⚠ SAME INPUTS but DIFFERENT PREDICTIONS (model randomness or different code path)")
        print(f"  Quality difference: {abs(split_cd - single_cd):.6f}")
else:
    print("✗ DIFFERENT INPUTS (not a fair comparison)")

print("=" * 70)

# Save summary
summary = {
    "images_match": bool(images_match),
    "gt_match": bool(gt_match),
    "pred_exact_match": bool(pred_match),
    "split_chamfer": split_cd,
    "single_chamfer": single_cd,
    "chamfer_diff": abs(split_cd - single_cd),
}
with open("validation_summary.json", "w") as f:
    json.dump(summary, f, indent=2)
print(f"\nSaved summary to validation_summary.json")
