import os
import numpy as np
import torch

v4_dir = 'experiments_pointdit_v5/outputs_pointdit_v5_V4/sample_0'
v5_dir = 'experiments_pointdit_v5/outputs_pointdit_v5_V5/sample_0'


def chamfer_distance(p: torch.Tensor, q: torch.Tensor) -> torch.Tensor:
    if p.dim() == 2:
        p = p.unsqueeze(0)
    if q.dim() == 2:
        q = q.unsqueeze(0)
    dists = torch.cdist(p, q, p=2)
    min_pq = dists.min(dim=2).values
    min_qp = dists.min(dim=1).values
    loss = (min_pq.pow(2).mean(dim=1) + min_qp.pow(2).mean(dim=1)).mean()
    return loss

image_v4 = np.load(os.path.join(v4_dir, 'input_image.npy'))
image_v5 = np.load(os.path.join(v5_dir, 'input_image.npy'))

# Normalized points (with batch dim)
gt_norm_v5 = np.load(os.path.join(v5_dir, 'gt_points_normalized.npy'))
pred_norm_v5 = np.load(os.path.join(v5_dir, 'pred_points_normalized.npy'))
gt_norm_v4 = np.load(os.path.join(v4_dir, 'gt_points_normalized.npy'))
pred_norm_v4 = np.load(os.path.join(v4_dir, 'pred_points_normalized.npy'))

H, W = image_v5.shape
same_image_shape = image_v4.shape == image_v5.shape
same_gt_norm = np.allclose(gt_norm_v4, gt_norm_v5)

GT = torch.from_numpy(gt_norm_v5.squeeze(0)).float()
V5 = torch.from_numpy(pred_norm_v5.squeeze(0)).float()
V4 = torch.from_numpy(pred_norm_v4.squeeze(0)).float()

cd_v5 = chamfer_distance(V5, GT).item()
cd_v4 = chamfer_distance(V4, GT).item()
cd_v4_v5 = chamfer_distance(V4, V5).item()

print({
    'same_image_shape': same_image_shape,
    'same_gt_norm': bool(same_gt_norm),
    'cd_v5_vs_gt': cd_v5,
    'cd_v4_vs_gt': cd_v4,
    'cd_v4_vs_v5': cd_v4_v5,
    'H': int(H), 'W': int(W),
})
