#!/usr/bin/env python3
"""
Similarity Metrics for Stippling Quality Evaluation
Based on GaussianBlueNoise evaluation methodology.

Metrics:
1. NN distance mean - Average nearest neighbor distance
2. NN distance std - Standard deviation of NN distances  
3. Histogram intersection - Overlap between density histograms

Usage:
    python similarity_metrics.py --reference path/to/reference.txt --output path/to/output.txt
    python similarity_metrics.py --eval-sample 0  # Evaluate Point-DiT on sample
"""

import argparse
import numpy as np
from pathlib import Path
from scipy.spatial import cKDTree
from PIL import Image
import sys
sys.path.insert(0, str(Path(__file__).parent.parent))


def load_points_from_txt(txt_path):
    """Load points from a text file (GBN format: N points, then x y per line)."""
    with open(txt_path, 'r') as f:
        lines = f.readlines()
    
    # Try to detect format
    first_line = lines[0].strip().split()
    
    if len(first_line) == 1:
        # GBN format: first line is count
        n_points = int(first_line[0])
        points = np.array([list(map(float, line.split())) for line in lines[1:n_points+1]])
    else:
        # Direct format: each line is x y
        points = np.array([list(map(float, line.split())) for line in lines if line.strip()])
    
    return points


def compute_nn_distances(points):
    """
    Compute nearest neighbor distances for all points.
    
    Returns:
        nn_distances: Array of NN distances for each point
    """
    if len(points) < 2:
        return np.array([0])
    
    tree = cKDTree(points)
    # Query k=2 to get self (distance=0) and nearest neighbor
    distances, _ = tree.query(points, k=2)
    nn_distances = distances[:, 1]  # Second closest (first is self)
    
    return nn_distances


def compute_nn_metrics(points):
    """
    Compute NN distance statistics.
    
    Returns:
        mean: Mean NN distance
        std: Std of NN distances
        min: Min NN distance
        max: Max NN distance
    """
    nn_distances = compute_nn_distances(points)
    
    return {
        'mean': np.mean(nn_distances),
        'std': np.std(nn_distances),
        'min': np.min(nn_distances),
        'max': np.max(nn_distances)
    }


def histogram_intersection(hist1, hist2):
    """
    Compute histogram intersection (overlap) between two histograms.
    Returns value in [0, 1] where 1 = identical distributions.
    
    Formula: sum(min(h1, h2)) / min(sum(h1), sum(h2))
    """
    # Normalize histograms
    h1 = hist1 / (hist1.sum() + 1e-10)
    h2 = hist2 / (hist2.sum() + 1e-10)
    
    # Intersection
    intersection = np.minimum(h1, h2).sum()
    
    return intersection


def compute_density_histogram(points, image_size, n_bins=32):
    """
    Compute a 2D density histogram from points.
    
    Args:
        points: Nx2 array of (x, y) coordinates in [0, 1] or image space
        image_size: (width, height) tuple
        n_bins: Number of bins per dimension
        
    Returns:
        2D histogram (density)
    """
    w, h = image_size
    
    # Normalize to [0, 1] if needed
    pts = points.copy()
    if pts.max() > 1.0:
        pts[:, 0] /= w
        pts[:, 1] /= h
    
    # Compute 2D histogram
    hist, _, _ = np.histogram2d(
        pts[:, 0], pts[:, 1],
        bins=n_bins,
        range=[[0, 1], [0, 1]]
    )
    
    return hist


def compute_1d_nn_histogram(nn_distances, n_bins=50, range_max=None):
    """
    Compute histogram of nearest neighbor distances.
    
    Args:
        nn_distances: Array of NN distances
        n_bins: Number of bins
        range_max: Max distance for binning (auto if None)
    
    Returns:
        Histogram array
    """
    if range_max is None:
        range_max = nn_distances.max() * 1.1
    
    hist, _ = np.histogram(nn_distances, bins=n_bins, range=(0, range_max))
    return hist


def compute_similarity_score(ref_value, output_value):
    """Compute similarity as percentage (100% = identical)."""
    if ref_value == 0:
        return 100.0 if output_value == 0 else 0.0
    
    # Use min/max ratio for positive values
    similarity = min(ref_value, output_value) / max(ref_value, output_value) * 100
    return similarity


def compare_point_sets(reference_points, output_points, image_size=(512, 512)):
    """
    Compare two point sets using GaussianBlueNoise metrics.
    
    Args:
        reference_points: Nx2 reference points
        output_points: Mx2 output points  
        image_size: Image dimensions for normalization
        
    Returns:
        Dictionary with all metrics and similarities
    """
    # Compute NN metrics
    ref_nn = compute_nn_metrics(reference_points)
    out_nn = compute_nn_metrics(output_points)
    
    # Compute NN distance histograms
    ref_nn_dists = compute_nn_distances(reference_points)
    out_nn_dists = compute_nn_distances(output_points)
    
    # Use shared range for fair comparison
    max_dist = max(ref_nn_dists.max(), out_nn_dists.max()) * 1.1
    ref_nn_hist = compute_1d_nn_histogram(ref_nn_dists, range_max=max_dist)
    out_nn_hist = compute_1d_nn_histogram(out_nn_dists, range_max=max_dist)
    
    # Histogram intersection on NN distance distributions
    nn_hist_intersection = histogram_intersection(ref_nn_hist, out_nn_hist) * 100
    
    # Compute 2D density histograms
    ref_density_hist = compute_density_histogram(reference_points, image_size)
    out_density_hist = compute_density_histogram(output_points, image_size)
    
    # Histogram intersection on 2D density
    density_hist_intersection = histogram_intersection(
        ref_density_hist.flatten(), 
        out_density_hist.flatten()
    ) * 100
    
    # Compute similarities
    mean_sim = compute_similarity_score(ref_nn['mean'], out_nn['mean'])
    std_sim = compute_similarity_score(ref_nn['std'], out_nn['std'])
    
    # Overall score (weighted average of key metrics)
    overall = (mean_sim + std_sim + density_hist_intersection) / 3
    
    return {
        'reference': {
            'n_points': len(reference_points),
            'nn_mean': ref_nn['mean'],
            'nn_std': ref_nn['std'],
            'nn_min': ref_nn['min'],
            'nn_max': ref_nn['max'],
        },
        'output': {
            'n_points': len(output_points),
            'nn_mean': out_nn['mean'],
            'nn_std': out_nn['std'],
            'nn_min': out_nn['min'],
            'nn_max': out_nn['max'],
        },
        'similarity': {
            'nn_mean': mean_sim,
            'nn_std': std_sim,
            'histogram_intersection': density_hist_intersection,
            'nn_histogram_intersection': nn_hist_intersection,
            'overall': overall,
        }
    }


def print_comparison_table(results):
    """Print comparison in the GaussianBlueNoise format."""
    print("\n" + "="*70)
    print("SIMILARITY METRICS (GaussianBlueNoise Format)")
    print("="*70)
    
    print(f"\n{'Metric':<25} {'Reference':<15} {'Output':<15} {'Similarity':<12}")
    print("-"*70)
    
    ref = results['reference']
    out = results['output']
    sim = results['similarity']
    
    print(f"{'Number of points':<25} {ref['n_points']:<15} {out['n_points']:<15} "
          f"{compute_similarity_score(ref['n_points'], out['n_points']):.1f}%")
    
    print(f"{'NN distance mean':<25} {ref['nn_mean']:<15.6f} {out['nn_mean']:<15.6f} "
          f"**{sim['nn_mean']:.1f}%**")
    
    print(f"{'NN distance std':<25} {ref['nn_std']:<15.6f} {out['nn_std']:<15.6f} "
          f"**{sim['nn_std']:.1f}%**")
    
    print(f"{'NN distance min':<25} {ref['nn_min']:<15.6f} {out['nn_min']:<15.6f} "
          f"{compute_similarity_score(ref['nn_min'], out['nn_min']):.1f}%")
    
    print(f"{'NN distance max':<25} {ref['nn_max']:<15.6f} {out['nn_max']:<15.6f} "
          f"{compute_similarity_score(ref['nn_max'], out['nn_max']):.1f}%")
    
    print(f"{'Histogram intersection':<25} {'-':<15} {'-':<15} "
          f"**{sim['histogram_intersection']:.1f}%**")
    
    print("-"*70)
    print(f"{'**Overall similarity**':<25} {'-':<15} {'-':<15} "
          f"**{sim['overall']:.1f}%**")
    print("="*70)


def evaluate_pointdit_sample(sample_idx, checkpoint_path=None):
    """
    Evaluate Point-DiT model on a specific sample.
    
    Args:
        sample_idx: Index of sample to evaluate
        checkpoint_path: Path to model checkpoint (default: latest)
    """
    import torch
    from experiments_pointdit_v5.model import StipplingPointDiTModel
    from experiments_pointdit_v5.dataset import StipplingDataset
    
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    
    # Load dataset
    data_dir = Path(__file__).parent.parent / "co3d" / "data_grads_v3"
    dataset = StipplingDataset(data_dir, num_samples=1000)
    
    if sample_idx >= len(dataset):
        print(f"Error: Sample {sample_idx} out of range (max {len(dataset)-1})")
        return
    
    # Load model
    if checkpoint_path is None:
        ckpt_dir = Path(__file__).parent / "checkpoints"
        ckpts = sorted(ckpt_dir.glob("pointdit_*.pt"))
        if not ckpts:
            print("No checkpoints found!")
            return
        checkpoint_path = ckpts[-1]
    
    print(f"Loading model from: {checkpoint_path}")
    model = StipplingPointDiTModel(
        point_dim=2,
        latent_dim=256,
        n_heads=8,
        n_layers=6,
        image_encoder_dim=256,
        timesteps=1000
    ).to(device)
    
    state = torch.load(checkpoint_path, map_location=device)
    model.load_state_dict(state['model_state_dict'])
    model.eval()
    
    # Get sample
    sample = dataset[sample_idx]
    grayscale = sample['grayscale'].unsqueeze(0).to(device)
    target_points = sample['target_points'].numpy()  # [N, 2]
    
    print(f"\nEvaluating sample {sample_idx}...")
    print(f"  Target points: {len(target_points)}")
    
    # Generate points
    with torch.no_grad():
        # Start from noise
        n_points = len(target_points)
        noise = torch.randn(1, n_points, 2, device=device)
        
        # Denoise
        generated = model.sample(grayscale, noise, steps=50)
        generated_points = generated[0].cpu().numpy()  # [N, 2]
    
    # Denormalize to image space (assuming 512x512)
    # Points are in [-1, 1] -> convert to [0, 512]
    target_denorm = (target_points + 1) / 2 * 512
    generated_denorm = (generated_points + 1) / 2 * 512
    
    # Compare
    results = compare_point_sets(target_denorm, generated_denorm, image_size=(512, 512))
    print_comparison_table(results)
    
    return results


def main():
    parser = argparse.ArgumentParser(description="Stippling Similarity Metrics")
    parser.add_argument("--reference", type=Path, help="Reference points file (.txt)")
    parser.add_argument("--output", type=Path, help="Output points file (.txt)")
    parser.add_argument("--eval-sample", type=int, help="Evaluate Point-DiT on sample index")
    parser.add_argument("--checkpoint", type=Path, help="Model checkpoint path")
    parser.add_argument("--image-size", type=int, default=512, help="Image size for normalization")
    
    args = parser.parse_args()
    
    if args.eval_sample is not None:
        # Evaluate Point-DiT model
        evaluate_pointdit_sample(args.eval_sample, args.checkpoint)
    
    elif args.reference and args.output:
        # Compare two point files
        print(f"Loading reference: {args.reference}")
        ref_points = load_points_from_txt(args.reference)
        
        print(f"Loading output: {args.output}")
        out_points = load_points_from_txt(args.output)
        
        image_size = (args.image_size, args.image_size)
        results = compare_point_sets(ref_points, out_points, image_size)
        print_comparison_table(results)
    
    else:
        parser.print_help()
        print("\nExample usage:")
        print("  python similarity_metrics.py --reference ref.txt --output out.txt")
        print("  python similarity_metrics.py --eval-sample 0")


if __name__ == "__main__":
    main()
