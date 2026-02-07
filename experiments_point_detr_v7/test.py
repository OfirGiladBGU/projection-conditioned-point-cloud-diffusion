"""Quick test script for Point-RT components."""

import torch
import torch.nn as nn
import sys
from pathlib import Path

# Add to path
repo_root = Path(__file__).parent
sys.path.insert(0, str(repo_root))

from model.point_rt import PointRT, FourierEmbedder
from fast_init import fast_density_initialization, sample_density_at_points, create_point_input_vectors
from losses import ChamferLoss, RepulsionLoss, HungarianMSELoss, PointRTLoss


def test_fourier_embedder():
    """Test Fourier embeddings."""
    print("\n" + "="*60)
    print("Testing FourierEmbedder...")
    print("="*60)
    
    embedder = FourierEmbedder(num_freqs=32)
    
    # Test single coordinate
    x = torch.tensor([[0.0], [0.5], [-0.5]])  # 3 points
    emb = embedder(x)
    
    print(f"Input shape: {x.shape}")
    print(f"Output shape: {emb.shape}")
    print(f"Expected: (3, 64) - 32 freqs * 2 (sin/cos)")
    assert emb.shape == (3, 64), "Shape mismatch!"
    print("✓ FourierEmbedder test passed")


def test_fast_init():
    """Test fast density initialization."""
    print("\n" + "="*60)
    print("Testing fast_density_initialization...")
    print("="*60)
    
    device = "cuda" if torch.cuda.is_available() else "cpu"
    
    # Create mock circular image
    B, H, W = 4, 512, 512
    image = torch.ones(B, 1, H, W, device=device) * 0.8  # Bright background
    
    # Add dark circle
    y = torch.linspace(-1, 1, H, device=device)
    x = torch.linspace(-1, 1, W, device=device)
    yy, xx = torch.meshgrid(y, x, indexing='ij')
    circle = (xx**2 + yy**2) < 0.25
    image[:, :, circle] = 0.1  # Dark circle
    
    # Initialize points
    num_points = 2000
    points = fast_density_initialization(image, num_points)
    
    print(f"Image shape: {image.shape}")
    print(f"Points shape: {points.shape}")
    print(f"Points range: [{points.min():.3f}, {points.max():.3f}]")
    
    # Check that more points are in circle
    circle_normalized = (points[:, :, 0]**2 + points[:, :, 1]**2) < 0.25
    circle_fraction = circle_normalized.float().mean().item()
    print(f"Fraction of points in circle: {circle_fraction:.3f} (should be ~0.25)")
    
    assert points.shape == (B, num_points, 2), "Shape mismatch!"
    assert points.min() >= -1.0 and points.max() <= 1.0, "Out of bounds!"
    print("✓ Fast initialization test passed")


def test_point_input_vectors():
    """Test creation of input vectors (x, y, intensity)."""
    print("\n" + "="*60)
    print("Testing create_point_input_vectors...")
    print("="*60)
    
    device = "cuda" if torch.cuda.is_available() else "cpu"
    
    B, N = 2, 1000
    image = torch.rand(B, 1, 512, 512, device=device)
    points = torch.rand(B, N, 2, device=device) * 2 - 1  # [-1, 1]
    
    input_vecs = create_point_input_vectors(points, image)
    
    print(f"Points shape: {points.shape}")
    print(f"Image shape: {image.shape}")
    print(f"Input vectors shape: {input_vecs.shape}")
    
    assert input_vecs.shape == (B, N, 3), "Shape mismatch!"
    print("✓ Input vectors test passed")


def test_model():
    """Test Point-RT model."""
    print("\n" + "="*60)
    print("Testing Point-RT Model...")
    print("="*60)
    
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Device: {device}")
    
    # Create model (with pretrained=False for speed)
    model = PointRT(
        n_points=1024,
        dim=128,
        n_layers=2,
        n_heads=4,
        use_resnet_backbone=True,
        resnet_pretrained=False,
    )
    model = model.to(device)
    
    params = model.get_num_params()
    print(f"Model parameters: {params:,}")
    
    # Create mock input
    B, N = 2, 1024
    x_init = torch.rand(B, N, 3, device=device)
    x_init[:, :, :2] = x_init[:, :, :2] * 2 - 1  # [-1, 1]
    
    image = torch.rand(B, 1, 512, 512, device=device)
    
    # Forward pass
    with torch.no_grad():
        output = model(x_init, image)
    
    print(f"Input shape: {x_init.shape}")
    print(f"Image shape: {image.shape}")
    print(f"Output shape: {output.shape}")
    print(f"Output range: [{output.min():.3f}, {output.max():.3f}]")
    
    assert output.shape == (B, N, 2), "Shape mismatch!"
    print("✓ Model test passed")


def test_losses():
    """Test loss functions."""
    print("\n" + "="*60)
    print("Testing Loss Functions...")
    print("="*60)
    
    device = "cuda" if torch.cuda.is_available() else "cpu"
    
    B, N = 2, 500
    pred = torch.rand(B, N, 2, device=device) * 2 - 1
    target = torch.rand(B, N, 2, device=device) * 2 - 1
    
    # Test Chamfer
    chamfer = ChamferLoss().to(device)
    chamfer_loss = chamfer(pred, target)
    print(f"Chamfer loss: {chamfer_loss.item():.4f}")
    
    # Test Repulsion
    repulsion = RepulsionLoss().to(device)
    repulsion_loss = repulsion(pred)
    print(f"Repulsion loss: {repulsion_loss.item():.4f}")
    
    # Test Hungarian (if scipy available)
    try:
        hungarian = HungarianMSELoss().to(device)
        hungarian_loss, _ = hungarian(pred, target)
        print(f"Hungarian MSE loss: {hungarian_loss.item():.4f}")
    except Exception as e:
        print(f"Hungarian loss skipped (scipy not available): {e}")
    
    # Test combined
    combined = PointRTLoss().to(device)
    total_loss, loss_dict = combined(pred, target)
    print(f"Combined loss: {total_loss.item():.4f}")
    print(f"  - MSE: {loss_dict['mse_loss']:.4f}")
    print(f"  - Repulsion: {loss_dict['repulsion_loss']:.4f}")
    print(f"  - Diversity: {loss_dict['diversity_loss']:.4f}")
    
    print("✓ Losses test passed")


def test_end_to_end():
    """End-to-end pipeline test."""
    print("\n" + "="*60)
    print("Testing End-to-End Pipeline...")
    print("="*60)
    
    device = "cuda" if torch.cuda.is_available() else "cpu"
    
    # 1. Initialize
    B = 2
    image = torch.rand(B, 1, 512, 512, device=device)
    
    print("Step 1: Fast initialization...")
    points_init = fast_density_initialization(image, 1024)
    print(f"  → {points_init.shape}")
    
    # 2. Create input vectors
    print("Step 2: Create input vectors...")
    input_vecs = create_point_input_vectors(points_init, image)
    print(f"  → {input_vecs.shape}")
    
    # 3. Model forward pass
    print("Step 3: Model forward pass...")
    model = PointRT(n_points=1024, dim=128, n_layers=2, n_heads=4, 
                    resnet_pretrained=False).to(device)
    with torch.no_grad():
        output = model(input_vecs, image)
    print(f"  → {output.shape}")
    
    # 4. Compute loss (with dummy target)
    print("Step 4: Compute loss...")
    target = points_init + torch.randn_like(points_init) * 0.05  # Slightly perturbed
    loss_fn = PointRTLoss().to(device)
    loss, loss_dict = loss_fn(output, target)
    print(f"  → {loss.item():.4f}")
    
    print("✓ End-to-end test passed!")


if __name__ == "__main__":
    print("\n" + "="*70)
    print("Point-RT Component Tests")
    print("="*70)
    
    try:
        test_fourier_embedder()
        test_fast_init()
        test_point_input_vectors()
        test_model()
        test_losses()
        test_end_to_end()
        
        print("\n" + "="*70)
        print("All tests passed! ✓")
        print("="*70)
    except Exception as e:
        print(f"\n✗ Test failed with error: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)
