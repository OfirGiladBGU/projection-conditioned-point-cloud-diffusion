"""Evaluation script for Point-RT."""

import torch
import torch.nn as nn
from pathlib import Path
from tqdm import tqdm
import json
import numpy as np

from model.point_rt import PointRT
from fast_init import fast_density_initialization, create_point_input_vectors
from dataset import PointRTDataset


class PointCloudMetrics:
    """Compute metrics for point clouds."""
    
    @staticmethod
    def chamfer_distance(pred: torch.Tensor, target: torch.Tensor) -> float:
        """Compute bidirectional Chamfer distance."""
        # (N, 2) and (M, 2)
        dist_pred_to_target = torch.min(torch.cdist(pred, target), dim=1)[0]
        dist_target_to_pred = torch.min(torch.cdist(target, pred), dim=1)[0]
        
        return (dist_pred_to_target.mean() + dist_target_to_pred.mean()).item() / 2
    
    @staticmethod
    def blue_noise_cv(points: torch.Tensor, grid_size: int = 32) -> float:
        """Compute coefficient of variation for blue noise quality.
        
        CV < 0.5: Crystalline (very uniform)
        CV ~ 0.6: Good blue noise
        CV > 0.8: Uniform random
        """
        # Map to grid
        grid_coords = (points + 1) / 2 * (grid_size - 1)
        grid_x = torch.round(grid_coords[:, 0]).long().clamp(0, grid_size - 1)
        grid_y = torch.round(grid_coords[:, 1]).long().clamp(0, grid_size - 1)
        
        # Count points in each cell
        histogram = torch.zeros(grid_size, grid_size, device=points.device)
        for i, j in zip(grid_x, grid_y):
            histogram[i, j] += 1
        
        # Compute CV = std / mean
        mean = histogram.mean()
        std = histogram.std()
        
        if mean > 0:
            cv = (std / mean).item()
        else:
            cv = float('inf')
        
        return cv
    
    @staticmethod
    def minimum_distance(points: torch.Tensor) -> float:
        """Compute minimum pairwise distance."""
        dist = torch.cdist(points, points)
        
        # Set diagonal to inf (self-distance)
        dist.fill_diagonal_(float('inf'))
        
        return dist.min().item()


def evaluate_model(
    model: PointRT,
    dataset: PointRTDataset,
    device: str,
    num_samples: int = 50,
) -> dict:
    """Evaluate model on dataset.
    
    Args:
        model: PointRT model
        dataset: PointRTDataset
        device: 'cuda' or 'cpu'
        num_samples: Number of samples to evaluate on
    
    Returns:
        Metrics dict
    """
    model.eval()
    
    metrics = {
        'chamfer_distance': [],
        'blue_noise_cv': [],
        'min_distance': [],
    }
    
    num_eval = min(num_samples, len(dataset))
    
    with torch.no_grad():
        for idx in tqdm(range(num_eval), desc="Evaluating"):
            sample = dataset[idx]
            image = sample['image'].unsqueeze(0).to(device)  # (1, 1, H, W)
            lloyd_points = sample['lloyd_points']
            
            if lloyd_points is None:
                continue
            
            lloyd_points = lloyd_points.to(device)
            
            # Forward pass
            init_points = fast_density_initialization(image, model.n_points)
            input_vecs = create_point_input_vectors(init_points, image)
            pred_points = model(input_vecs, image)
            pred_points = pred_points.squeeze(0)  # (N, 2)
            
            # Compute metrics
            chamfer = PointCloudMetrics.chamfer_distance(pred_points, lloyd_points)
            cv = PointCloudMetrics.blue_noise_cv(pred_points)
            min_dist = PointCloudMetrics.minimum_distance(pred_points)
            
            metrics['chamfer_distance'].append(chamfer)
            metrics['blue_noise_cv'].append(cv)
            metrics['min_distance'].append(min_dist)
    
    # Compute statistics
    results = {}
    for key, values in metrics.items():
        if values:
            values = np.array(values)
            results[key] = {
                'mean': float(values.mean()),
                'std': float(values.std()),
                'min': float(values.min()),
                'max': float(values.max()),
            }
    
    return results


def main():
    """Run evaluation."""
    
    # Load model
    model_path = Path("./outputs_point_detr_v7/model_final.pt")
    if not model_path.exists():
        print(f"Model not found at {model_path}")
        return
    
    device = "cuda" if torch.cuda.is_available() else "cpu"
    model = PointRT().to(device)
    model.load_state_dict(torch.load(model_path, map_location=device))
    
    # Create dataset
    dataset = PointRTDataset(
        source_dir="/groups/asharf_group/ofirgila/ControlNet/training/data_grads_v3/source",
        lloyd_dir=None,  # Would need to be set
    )
    
    # Evaluate
    results = evaluate_model(model, dataset, device, num_samples=50)
    
    print("\n" + "="*60)
    print("Evaluation Results")
    print("="*60)
    for metric, stats in results.items():
        print(f"{metric}:")
        for key, val in stats.items():
            print(f"  {key}: {val:.4f}")
    
    # Save results
    output_path = Path("./outputs_point_detr_v7/eval_results.json")
    with open(output_path, 'w') as f:
        json.dump(results, f, indent=2)
    print(f"\nResults saved to {output_path}")


if __name__ == "__main__":
    main()
