"""
Test Script for Neural-Newton Pipeline (Sinkhorn + Lloyd)

This script evaluates the new Sinkhorn Loss and Differentiable Lloyd's Step
against the current Chamfer+Repulsion baseline.

Tests:
1. Loss function sanity checks
2. Gradient flow verification
3. Comparison on real data samples
4. Training step integration test
"""

import torch
import torch.nn.functional as F
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path
import sys

# Add parent to path for imports
sys.path.insert(0, str(Path(__file__).parent))

from config import Config
from model import PointDiT
from diffusion import DDPMScheduler, ChamferLoss, RepulsionLoss, GridDensityLoss
from sinkhorn_lloyd_losses import (
    SinkhornDensityLoss, 
    SinkhornDensityLossSimple,
    DifferentiableLloydStep,
    NeuralNewtonLoss,
    HAS_GEOMLOSS
)
from dataset import create_dataloaders


def visualize_points_comparison(
    image: torch.Tensor,
    pred_points: torch.Tensor, 
    refined_points: torch.Tensor,
    gt_points: torch.Tensor,
    save_path: str,
    title: str = ""
):
    """Visualize comparison between predicted, refined, and GT points."""
    fig, axes = plt.subplots(1, 4, figsize=(16, 4))
    
    # Get numpy arrays
    img = image[0, 0].cpu().numpy()
    pred = pred_points[0].cpu().numpy()
    refined = refined_points[0].cpu().numpy()
    gt = gt_points[0].cpu().numpy()
    
    # Convert from [-1, 1] to [0, 1]
    pred_scaled = (pred + 1) / 2
    refined_scaled = (refined + 1) / 2
    gt_scaled = (gt + 1) / 2
    
    # 1. Input image
    axes[0].imshow(img, cmap='gray', origin='lower')
    axes[0].set_title('Input Image')
    axes[0].axis('off')
    
    # 2. Predicted points
    axes[1].imshow(img, cmap='gray', origin='lower', alpha=0.3)
    axes[1].scatter(pred_scaled[:, 0] * img.shape[1], 
                    pred_scaled[:, 1] * img.shape[0], 
                    s=1, c='blue', alpha=0.8)
    axes[1].set_title('Predicted Points')
    axes[1].axis('off')
    axes[1].set_xlim(0, img.shape[1])
    axes[1].set_ylim(0, img.shape[0])
    
    # 3. Lloyd-refined points
    axes[2].imshow(img, cmap='gray', origin='lower', alpha=0.3)
    axes[2].scatter(refined_scaled[:, 0] * img.shape[1], 
                    refined_scaled[:, 1] * img.shape[0], 
                    s=1, c='green', alpha=0.8)
    axes[2].set_title('Lloyd-Refined Points')
    axes[2].axis('off')
    axes[2].set_xlim(0, img.shape[1])
    axes[2].set_ylim(0, img.shape[0])
    
    # 4. Ground truth
    axes[3].imshow(img, cmap='gray', origin='lower', alpha=0.3)
    axes[3].scatter(gt_scaled[:, 0] * img.shape[1], 
                    gt_scaled[:, 1] * img.shape[0], 
                    s=1, c='red', alpha=0.8)
    axes[3].set_title('Ground Truth')
    axes[3].axis('off')
    axes[3].set_xlim(0, img.shape[1])
    axes[3].set_ylim(0, img.shape[0])
    
    if title:
        fig.suptitle(title)
    
    plt.tight_layout()
    plt.savefig(save_path, dpi=150, bbox_inches='tight')
    plt.close()
    print(f"Saved: {save_path}")


def test_loss_scale_comparison(device: str = 'cuda'):
    """Compare loss scales across different loss functions."""
    print("\n" + "="*60)
    print("TEST 1: Loss Scale Comparison")
    print("="*60)
    
    B, N = 4, 512
    H, W = 256, 256
    
    # Generate test data
    torch.manual_seed(42)
    
    # Points from density (realistic)
    image = torch.rand(B, 1, H, W, device=device)
    
    # Good prediction (similar to GT)
    gt_points = (torch.rand(B, N, 2, device=device) * 2 - 1)
    good_pred = gt_points + torch.randn_like(gt_points) * 0.05  # Small noise
    good_pred = good_pred.clamp(-1, 1)
    
    # Bad prediction (random)
    bad_pred = (torch.rand(B, N, 2, device=device) * 2 - 1)
    
    # Initialize losses
    chamfer = ChamferLoss()
    repulsion = RepulsionLoss(repulsion_radius=0.02)
    grid_density = GridDensityLoss(grid_sizes=(32, 64))
    
    if HAS_GEOMLOSS:
        sinkhorn = SinkhornDensityLoss(blur=0.01, grid_size=64).to(device)
    else:
        sinkhorn = SinkhornDensityLossSimple(grid_size=32, blur=0.05).to(device)
    
    lloyd = DifferentiableLloydStep(grid_size=32, tau=0.01, num_steps=1).to(device)
    
    print("\nLoss comparison (Good vs Bad prediction):")
    print("-" * 50)
    
    losses = {
        'Chamfer': chamfer,
        'Repulsion': repulsion,
        'GridDensity': grid_density,
        'Sinkhorn': sinkhorn,
    }
    
    for name, loss_fn in losses.items():
        if name in ['Chamfer', 'Repulsion']:
            good_loss = loss_fn(good_pred, gt_points) if name == 'Chamfer' else loss_fn(good_pred)
            bad_loss = loss_fn(bad_pred, gt_points) if name == 'Chamfer' else loss_fn(bad_pred)
        else:
            good_loss = loss_fn(good_pred, image)
            bad_loss = loss_fn(bad_pred, image)
        
        ratio = bad_loss.item() / (good_loss.item() + 1e-8)
        print(f"{name:15s}: Good={good_loss.item():.6f}, Bad={bad_loss.item():.6f}, Ratio={ratio:.2f}x")
    
    # Test Lloyd refinement effect
    print("\n--- Lloyd Refinement Effect ---")
    refined_good = lloyd(good_pred, image)
    refined_bad = lloyd(bad_pred, image)
    
    movement_good = (refined_good - good_pred).norm(dim=-1).mean()
    movement_bad = (refined_bad - bad_pred).norm(dim=-1).mean()
    
    print(f"Point movement (good pred): {movement_good.item():.6f}")
    print(f"Point movement (bad pred):  {movement_bad.item():.6f}")
    
    # Check if Lloyd improves Chamfer
    chamfer_before_good = chamfer(good_pred, gt_points).item()
    chamfer_after_good = chamfer(refined_good, gt_points).item()
    chamfer_before_bad = chamfer(bad_pred, gt_points).item()
    chamfer_after_bad = chamfer(refined_bad, gt_points).item()
    
    print(f"\nChamfer before/after Lloyd (good): {chamfer_before_good:.6f} -> {chamfer_after_good:.6f}")
    print(f"Chamfer before/after Lloyd (bad):  {chamfer_before_bad:.6f} -> {chamfer_after_bad:.6f}")


def test_gradient_flow(device: str = 'cuda'):
    """Verify gradient flow through all loss functions."""
    print("\n" + "="*60)
    print("TEST 2: Gradient Flow Verification")
    print("="*60)
    
    B, N = 2, 256
    H, W = 128, 128
    
    torch.manual_seed(42)
    points = (torch.rand(B, N, 2, device=device) * 2 - 1)
    gt_points = (torch.rand(B, N, 2, device=device) * 2 - 1)
    image = torch.rand(B, 1, H, W, device=device)
    
    # Test NeuralNewtonLoss
    print("\n--- NeuralNewtonLoss Gradient Test ---")
    
    nn_loss = NeuralNewtonLoss(
        sinkhorn_weight=1.0,
        chamfer_weight=1.0,
        use_lloyd_refinement=True,
        lloyd_steps=1,
        grid_size=32,
    ).to(device)
    
    points_test = points.clone().requires_grad_(True)
    result = nn_loss(points_test, gt_points, image)
    result['loss'].backward()
    
    grad_norm = points_test.grad.norm().item()
    grad_mean = points_test.grad.abs().mean().item()
    grad_max = points_test.grad.abs().max().item()
    
    print(f"Total loss: {result['loss'].item():.6f}")
    print(f"Sinkhorn:   {result['sinkhorn']:.6f}")
    print(f"Chamfer:    {result['chamfer']:.6f}")
    print(f"Gradient norm: {grad_norm:.6f}")
    print(f"Gradient mean: {grad_mean:.6f}")
    print(f"Gradient max:  {grad_max:.6f}")
    
    # Check gradient is not NaN or Inf
    assert not torch.isnan(points_test.grad).any(), "NaN in gradients!"
    assert not torch.isinf(points_test.grad).any(), "Inf in gradients!"
    print("✓ Gradients are valid (no NaN/Inf)")


def test_on_real_data(device: str = 'cuda', num_samples: int = 3):
    """Test on real dataset samples."""
    print("\n" + "="*60)
    print("TEST 3: Evaluation on Real Data")
    print("="*60)
    
    config = Config.default()
    
    train_loader, val_loader = create_dataloaders(
        source_dir=config.data.source_dir,
        target_dir=config.data.target_dir,
        batch_size=1,  # Small batch for testing
        image_size=config.data.image_size,
        num_points=config.data.num_points,
        val_split=config.data.val_split,
    )
    
    # Load model (if checkpoint exists)
    model = PointDiT(
        n_points=config.model.n_points,
        dim=config.model.dim,
        n_layers=config.model.n_layers,
        n_heads=config.model.n_heads,
        image_size=config.model.image_size,
        dropout=config.model.dropout,
    ).to(device)
    
    # Try to load latest checkpoint
    ckpt_path = Path(__file__).parent / 'outputs_pointdit_v5' / 'latest.pt'
    if ckpt_path.exists():
        print(f"Loading checkpoint: {ckpt_path}")
        ckpt = torch.load(ckpt_path, map_location=device)
        model.load_state_dict(ckpt['model'])
        print("✓ Checkpoint loaded")
    else:
        print(f"[Warning] No checkpoint found at {ckpt_path}, using random weights")
    
    scheduler = DDPMScheduler()
    
    # Initialize losses
    chamfer = ChamferLoss()
    
    if HAS_GEOMLOSS:
        sinkhorn = SinkhornDensityLoss(blur=0.01, grid_size=64).to(device)
    else:
        sinkhorn = SinkhornDensityLossSimple(grid_size=32, blur=0.05).to(device)
    
    lloyd = DifferentiableLloydStep(grid_size=64, tau=0.01, num_steps=2).to(device)
    
    # Create output dir
    output_dir = Path(__file__).parent / 'outputs_sinkhorn_lloyd_test'
    output_dir.mkdir(exist_ok=True)
    
    model.eval()
    results = []
    
    print(f"\nEvaluating {num_samples} samples...")
    
    for i, batch in enumerate(val_loader):
        if i >= num_samples:
            break
        
        image = batch['image'].to(device)
        gt_points = batch['points'].to(device)
        
        with torch.no_grad():
            # Simple forward pass (no diffusion, just direct prediction)
            # Use t=0 for direct prediction
            t = torch.zeros(1, device=device, dtype=torch.long)
            noise = torch.randn_like(gt_points) * 0.1
            x_noisy = gt_points + noise
            
            pred_points = model(x_noisy, t, image)
            pred_points = pred_points.clamp(-1, 1)
            
            # Apply Lloyd refinement
            refined_points = lloyd(pred_points, image)
            
            # Compute losses
            chamfer_pred = chamfer(pred_points, gt_points).item()
            chamfer_refined = chamfer(refined_points, gt_points).item()
            sinkhorn_pred = sinkhorn(pred_points, image).item()
            sinkhorn_refined = sinkhorn(refined_points, image).item()
            
            results.append({
                'sample': i,
                'chamfer_pred': chamfer_pred,
                'chamfer_refined': chamfer_refined,
                'sinkhorn_pred': sinkhorn_pred,
                'sinkhorn_refined': sinkhorn_refined,
            })
            
            print(f"\nSample {i}:")
            print(f"  Chamfer:  pred={chamfer_pred:.6f}, refined={chamfer_refined:.6f}, "
                  f"change={100*(chamfer_refined-chamfer_pred)/chamfer_pred:+.1f}%")
            print(f"  Sinkhorn: pred={sinkhorn_pred:.6f}, refined={sinkhorn_refined:.6f}, "
                  f"change={100*(sinkhorn_refined-sinkhorn_pred)/sinkhorn_pred:+.1f}%")
            
            # Visualize
            visualize_points_comparison(
                image, pred_points, refined_points, gt_points,
                str(output_dir / f'sample_{i}_comparison.png'),
                title=f'Sample {i}: Chamfer {chamfer_pred:.4f} -> {chamfer_refined:.4f}'
            )
    
    # Summary
    print("\n--- Summary ---")
    avg_chamfer_pred = np.mean([r['chamfer_pred'] for r in results])
    avg_chamfer_refined = np.mean([r['chamfer_refined'] for r in results])
    avg_sinkhorn_pred = np.mean([r['sinkhorn_pred'] for r in results])
    avg_sinkhorn_refined = np.mean([r['sinkhorn_refined'] for r in results])
    
    print(f"Average Chamfer:  pred={avg_chamfer_pred:.6f}, refined={avg_chamfer_refined:.6f}")
    print(f"Average Sinkhorn: pred={avg_sinkhorn_pred:.6f}, refined={avg_sinkhorn_refined:.6f}")
    print(f"\nVisualizations saved to: {output_dir}")


def test_training_integration(device: str = 'cuda', num_steps: int = 10):
    """Test integration with training loop."""
    print("\n" + "="*60)
    print("TEST 4: Training Integration Test")
    print("="*60)
    
    config = Config.default()
    
    train_loader, _ = create_dataloaders(
        source_dir=config.data.source_dir,
        target_dir=config.data.target_dir,
        batch_size=2,  # Small batch for testing
        image_size=config.data.image_size,
        num_points=config.data.num_points,
        val_split=config.data.val_split,
    )
    
    model = PointDiT(
        n_points=config.model.n_points,
        dim=config.model.dim,
        n_layers=config.model.n_layers,
        n_heads=config.model.n_heads,
        image_size=config.model.image_size,
        dropout=config.model.dropout,
    ).to(device)
    
    scheduler = DDPMScheduler()
    optimizer = torch.optim.AdamW(model.parameters(), lr=1e-4)
    
    # Neural-Newton loss
    nn_loss = NeuralNewtonLoss(
        sinkhorn_weight=0.1,  # Start with lower weight for stability
        chamfer_weight=1.0,
        use_lloyd_refinement=True,
        lloyd_steps=1,
        grid_size=32,  # Smaller for training efficiency
    ).to(device)
    
    chamfer_fn = ChamferLoss()
    
    model.train()
    losses = []
    
    print(f"\nRunning {num_steps} training steps with Neural-Newton loss...")
    
    for step, batch in enumerate(train_loader):
        if step >= num_steps:
            break
        
        image = batch['image'].to(device)
        gt_points = batch['points'].to(device)
        
        # Standard diffusion training
        noise = torch.randn_like(gt_points)
        t = torch.randint(0, scheduler.num_train_timesteps, (image.shape[0],), device=device)
        x_t = scheduler.add_noise(gt_points, noise, t)
        
        # Model prediction
        pred_x0 = model(x_t, t, image)
        
        # Neural-Newton loss
        result = nn_loss(pred_x0, gt_points, image)
        loss = result['loss']
        
        # Backward
        optimizer.zero_grad()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        optimizer.step()
        
        losses.append(loss.item())
        
        print(f"Step {step}: loss={loss.item():.6f}, "
              f"sinkhorn={result['sinkhorn']:.6f}, chamfer={result['chamfer']:.6f}")
    
    # Check loss decreased
    if len(losses) >= 5:
        first_half = np.mean(losses[:len(losses)//2])
        second_half = np.mean(losses[len(losses)//2:])
        print(f"\nLoss trend: {first_half:.6f} -> {second_half:.6f}")
        if second_half < first_half:
            print("✓ Loss is decreasing")
        else:
            print("⚠ Loss not decreasing (may need more steps or tuning)")


def main():
    """Run all tests."""
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    print(f"Device: {device}")
    print(f"geomloss available: {HAS_GEOMLOSS}")
    
    # Test 1: Loss scale comparison
    test_loss_scale_comparison(device)
    
    # Test 2: Gradient flow
    test_gradient_flow(device)
    
    # Test 3: Real data evaluation
    test_on_real_data(device, num_samples=3)
    
    # Test 4: Training integration
    test_training_integration(device, num_steps=10)
    
    print("\n" + "="*60)
    print("ALL TESTS COMPLETED!")
    print("="*60)


if __name__ == '__main__':
    main()
