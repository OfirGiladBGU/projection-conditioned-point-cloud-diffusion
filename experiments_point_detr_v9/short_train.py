"""
Short Training Script for Point-RT V9 - Quick Unsupervised Prototype Testing

This script runs a quick UNSUPERVISED training to verify the architecture
without needing ground truth points. Perfect for testing physics-based losses!

Key advantages over V7:
- No GT points needed (use Sinkhorn + AdaptiveRepulsion instead)
- Can train on ANY images
- Faster iteration: validate on spacing quality (blue noise)

Usage:
    python short_train.py
    
    # On GPU node:
    bash /home/ofirgila/scripts/pycharm_rtx_6000_32h.sh
    cd experiments_point_detr_v8 && python short_train.py
"""

import torch
from pathlib import Path
import matplotlib.pyplot as plt
import numpy as np
from tqdm import tqdm
import sys
import time
from PIL import Image

sys.path.insert(0, str(Path(__file__).parent))

from config import Config
from model import StippleRefiner
from fast_init import fast_density_initialization, create_point_input_vectors
from losses import UnsupervisedStipplingLoss


def custom_collate_fn(batch):
    """Custom collate that keeps only images."""
    return {
        'image': torch.stack([item['image'] for item in batch]),
        'image_path': [item['image_path'] for item in batch],
    }


def compute_spacing_metrics(pred_points: torch.Tensor):
    """Compute blue noise spacing metrics (no GT needed!)."""
    B, N, _ = pred_points.shape
    
    # Nearest neighbor distances
    dists = torch.cdist(pred_points, pred_points)
    mask = torch.eye(N, device=pred_points.device).bool().unsqueeze(0).expand(B, -1, -1)
    dists = dists.masked_fill(mask, float('inf'))
    nn_dists, _ = dists.min(dim=2)
    
    mean_nn = nn_dists.mean().item()
    cv = nn_dists.std().item() / (mean_nn + 1e-8)  # Coefficient of variation
    
    return {'mean_nn': mean_nn, 'cv': cv, 'nn_dists': nn_dists}


def visualize_results(
    image: torch.Tensor,
    pred_points: torch.Tensor,
    spacing_metrics: dict,
    epoch: int,
    output_dir: Path,
):
    """Create visualization of results (unsupervised - no GT)."""
    fig, axes = plt.subplots(1, 2, figsize=(12, 5))
    
    img = image[0, 0].cpu().numpy()
    pred = pred_points[0].cpu().numpy()
    
    # Density-colored points
    nn_dists = spacing_metrics['nn_dists'][0].cpu().numpy()
    axes[0].imshow(img, cmap='gray', extent=[-1, 1, -1, 1], origin='lower')
    scatter0 = axes[0].scatter(pred[:, 0], pred[:, 1], c=nn_dists, cmap='viridis', s=3, alpha=0.7)
    axes[0].set_title(f"Predicted Points (Epoch {epoch})\nMean NN={spacing_metrics['mean_nn']:.4f}, CV={spacing_metrics['cv']:.3f}")
    axes[0].set_xlim(-1, 1)
    axes[0].set_ylim(-1, 1)
    axes[0].set_aspect('equal')
    cbar0 = plt.colorbar(scatter0, ax=axes[0], label='NN Distance')
    
    # Density heatmap
    axes[1].imshow(img, cmap='gray', extent=[-1, 1, -1, 1], origin='lower')
    # Create density heatmap from points
    H, W = 64, 64
    density_map = np.zeros((H, W))
    for point in pred:
        x_idx = int((point[0] + 1) / 2 * (W - 1))
        y_idx = int((point[1] + 1) / 2 * (H - 1))
        x_idx = np.clip(x_idx, 0, W - 1)
        y_idx = np.clip(y_idx, 0, H - 1)
        density_map[y_idx, x_idx] += 1
    
    from scipy.ndimage import gaussian_filter
    density_map = gaussian_filter(density_map.astype(float), sigma=3)
    scatter1 = axes[1].imshow(density_map, extent=[-1, 1, -1, 1], origin='lower', cmap='hot', alpha=0.6)
    axes[1].set_title("Point Density Heatmap")
    axes[1].set_xlim(-1, 1)
    axes[1].set_ylim(-1, 1)
    axes[1].set_aspect('equal')
    plt.colorbar(scatter1, ax=axes[1], label='Density')
    
    plt.tight_layout()
    plt.savefig(output_dir / f'epoch_{epoch:03d}.png', dpi=150)
    plt.close()


def main():
    """Quick UNSUPERVISED training loop for Point-RT V8."""
    
    # Configuration - small batch for quick testing
    BATCH_SIZE = 4
    NUM_EPOCHS = 20
    NUM_TRAIN_SAMPLES = 200  # Small subset
    EVAL_EVERY = 2
    
    # UNSUPERVISED loss weights
    SINKHORN_WEIGHT = 1.0
    REPULSION_WEIGHT = 0.5
    
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    print(f"Device: {device}")
    print(f"UNSUPERVISED Loss weights - Sinkhorn: {SINKHORN_WEIGHT}, Repulsion: {REPULSION_WEIGHT}")
    print("(No ground truth points needed!)")
    
    # Create output directory
    output_dir = Path(__file__).parent / 'outputs_short_train'
    output_dir.mkdir(exist_ok=True)
    
    # Load config and data
    config = Config.default()
    print(f"\nLoading dataset from: {config.data.source_dir}")
    
    # Create dataset manually with custom collate function
    from torch.utils.data import random_split
    from dataset import PointRTDataset
    
    dataset = PointRTDataset(
        source_dir=config.data.source_dir,
        image_size=config.data.image_size,
        num_points=config.data.num_points,
        lloyd_dir=None,  # No GT needed for unsupervised!
    )
    
    n_val = int(len(dataset) * 0.1)
    n_train = len(dataset) - n_val
    train_dataset, val_dataset = random_split(dataset, [n_train, n_val])
    
    train_loader = torch.utils.data.DataLoader(
        train_dataset,
        batch_size=BATCH_SIZE,
        shuffle=True,
        num_workers=4,
        pin_memory=True,
        collate_fn=custom_collate_fn,
    )
    
    val_loader = torch.utils.data.DataLoader(
        val_dataset,
        batch_size=BATCH_SIZE,
        shuffle=False,
        num_workers=4,
        pin_memory=True,
        collate_fn=custom_collate_fn,
    )
    
    # Limit training data
    train_iter = iter(train_loader)
    train_data = []
    for _ in range(NUM_TRAIN_SAMPLES // BATCH_SIZE):
        try:
            batch = next(train_iter)
            train_data.append(batch)
        except StopIteration:
            break
    
    print(f"Training on {len(train_data) * BATCH_SIZE} samples")

    # Get validation sample for consistent evaluation
    val_batch = next(iter(val_loader))
    val_image = val_batch['image'].to(device)
    
    # Model and optimizer
    model = StippleRefiner(
        n_points=config.data.num_points,
        dim=config.model.dim,
        n_layers=config.model.n_layers,
        n_heads=config.model.n_heads,
        image_size=config.data.image_size,
        use_resnet_backbone=config.model.use_resnet_backbone,
        resnet_pretrained=False,  # False for quick test
        dropout=config.model.dropout,
        n_refine_steps=config.model.n_refine_steps,
        share_weights=config.model.share_weights,
        delta_scale=config.model.delta_scale,
    ).to(device)
    
    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=config.training.learning_rate,
        weight_decay=config.training.weight_decay,
    )
    
    loss_fn = UnsupervisedStipplingLoss(
        sinkhorn_weight=SINKHORN_WEIGHT,
        repulsion_weight=REPULSION_WEIGHT,
        repulsion_base_radius=config.training.base_radius,
        sinkhorn_blur=config.training.sinkhorn_blur,
    ).to(device)
    
    print(f"Model parameters: {model.get_num_params():,}")
    
    # Training loop
    history = {
        'train_loss': [],
        'train_sinkhorn': [],
        'train_repulsion': [],
        'val_mean_nn': [],
        'val_cv': [],
    }
    
    start_time = time.time()
    
    for epoch in range(1, NUM_EPOCHS + 1):
        model.train()
        epoch_losses = {'loss': [], 'sinkhorn': [], 'repulsion': []}
        
        pbar = tqdm(train_data, desc=f"Epoch {epoch}/{NUM_EPOCHS}")
        for batch in pbar:
            image = batch['image'].to(device)
            
            # Fast initialization
            with torch.no_grad():
                init_points = fast_density_initialization(image, model.n_points)
                input_vecs = create_point_input_vectors(init_points, image)
            
            optimizer.zero_grad()
            
            # Forward pass (SINGLE PASS - single-pass refinement!)
            pred_points = model(input_vecs, image)
            
            # UNSUPERVISED loss computation (no GT needed!)
            loss, loss_dict = loss_fn(pred_points, image)
            
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), config.training.max_grad_norm)
            optimizer.step()
            
            epoch_losses['loss'].append(loss.item())
            epoch_losses['sinkhorn'].append(loss_dict['sinkhorn_loss'])
            epoch_losses['repulsion'].append(loss_dict['repulsion_loss'])
            
            pbar.set_postfix({
                'loss': f"{loss.item():.4f}",
                'sinkhorn': f"{loss_dict['sinkhorn_loss']:.6f}",
                'rep': f"{loss_dict['repulsion_loss']:.6f}",
            })
        
        # Record training metrics
        avg_loss = np.mean(epoch_losses['loss'])
        avg_sinkhorn = np.mean(epoch_losses['sinkhorn'])
        avg_repulsion = np.mean(epoch_losses['repulsion'])
        
        history['train_loss'].append(avg_loss)
        history['train_sinkhorn'].append(avg_sinkhorn)
        history['train_repulsion'].append(avg_repulsion)
        
        print(f"Epoch {epoch}: Loss={avg_loss:.4f}, Sinkhorn={avg_sinkhorn:.6f}, Repulsion={avg_repulsion:.6f}")
        
        # Evaluation
        if epoch % EVAL_EVERY == 0 or epoch == 1:
            model.eval()
            with torch.no_grad():
                # Initialize and refine (single pass)
                init_points = fast_density_initialization(val_image, model.n_points)
                input_vecs = create_point_input_vectors(init_points, val_image)
                pred_points = model(input_vecs, val_image)  # SINGLE FORWARD PASS!
                
                spacing_metrics = compute_spacing_metrics(pred_points)
                
                history['val_mean_nn'].append(spacing_metrics['mean_nn'])
                history['val_cv'].append(spacing_metrics['cv'])
                
                print(f"  Eval: Mean NN={spacing_metrics['mean_nn']:.4f}, CV={spacing_metrics['cv']:.3f}")
                
                # Save visualization
                visualize_results(
                    val_image, pred_points,
                    spacing_metrics, epoch, output_dir
                )
    
    elapsed = time.time() - start_time
    print(f"\nTraining completed in {elapsed/60:.1f} minutes")
    
    # Plot training curves
    fig, axes = plt.subplots(2, 2, figsize=(12, 10))
    
    # Training loss
    axes[0, 0].plot(history['train_loss'], label='Total Loss', linewidth=2)
    axes[0, 0].set_xlabel('Epoch')
    axes[0, 0].set_ylabel('Loss')
    axes[0, 0].set_title('Training Loss (Unsupervised)')
    axes[0, 0].legend()
    axes[0, 0].grid(True, alpha=0.3)
    
    # Individual losses
    axes[0, 1].plot(history['train_sinkhorn'], label='Sinkhorn', linewidth=2)
    axes[0, 1].plot([x*2 for x in history['train_repulsion']], label='Repulsion (×2)', linewidth=2)
    axes[0, 1].set_xlabel('Epoch')
    axes[0, 1].set_ylabel('Loss')
    axes[0, 1].set_title('Loss Components')
    axes[0, 1].legend()
    axes[0, 1].grid(True, alpha=0.3)
    
    # Blue noise quality (Mean NN Distance)
    eval_epochs = list(range(EVAL_EVERY, NUM_EPOCHS + 1, EVAL_EVERY))
    if 1 not in eval_epochs:
        eval_epochs = [1] + eval_epochs
    
    axes[1, 0].plot(eval_epochs[:len(history['val_mean_nn'])], history['val_mean_nn'], 'b-o', linewidth=2, markersize=6)
    axes[1, 0].set_xlabel('Epoch')
    axes[1, 0].set_ylabel('Mean NN Distance')
    axes[1, 0].set_title('Blue Noise Quality (Spacing)')
    axes[1, 0].grid(True, alpha=0.3)
    
    # Coefficient of Variation
    axes[1, 1].plot(eval_epochs[:len(history['val_cv'])], history['val_cv'], 'g-o', linewidth=2, markersize=6)
    axes[1, 1].set_xlabel('Epoch')
    axes[1, 1].set_ylabel('Coefficient of Variation')
    axes[1, 1].set_title('Blue Noise Regularity (lower = better)')
    axes[1, 1].grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig(output_dir / 'training_curves.png', dpi=150)
    plt.close()
    
    print(f"\nResults saved to: {output_dir}")
    print(f"  - Training curves: {output_dir}/training_curves.png")
    print(f"  - Per-epoch visualizations: {output_dir}/epoch_*.png")
    
    # Final summary
    print("\n" + "="*60)
    print("FINAL SUMMARY - UNSUPERVISED TRAINING")
    print("="*60)
    print(f"Training time: {elapsed/60:.1f} minutes")
    
    if history['val_mean_nn']:
        final_nn = history['val_mean_nn'][-1]
        final_cv = history['val_cv'][-1]
        initial_nn = history['val_mean_nn'][0]
        print(f"\nFinal Metrics (Epoch {NUM_EPOCHS}):")
        print(f"  Mean NN Distance: {final_nn:.4f} (initial: {initial_nn:.4f})")
        print(f"  CV (regularity):  {final_cv:.3f}")
    
    print(f"\nKey observations:")
    print(f"  - Single forward pass per image (no diffusion loop)")
    print(f"  - NO GROUND TRUTH POINTS NEEDED! (physics-based losses only)")
    print(f"  - Sinkhorn loss: Capacity/density constraint")
    print(f"  - Repulsion loss: Adaptive spacing with blue noise")
    print(f"  - Expected CV target: 0.60-0.70 (good blue noise)")
    print(f"  - Expected Mean NN: 0.015-0.025 (reasonable spacing)")
    
    # Save model weights
    weights_path = output_dir / 'model_weights.pt'
    torch.save({
        'epoch': NUM_EPOCHS,
        'model_state_dict': model.state_dict(),
        'optimizer_state_dict': optimizer.state_dict(),
        'history': history,
        'config': {
            'batch_size': BATCH_SIZE,
            'num_epochs': NUM_EPOCHS,
            'num_train_samples': NUM_TRAIN_SAMPLES,
            'sinkhorn_weight': SINKHORN_WEIGHT,
            'repulsion_weight': REPULSION_WEIGHT,
        },
    }, weights_path)
    print(f"  - Model weights: {weights_path}")
    
    print("\n✓ Unsupervised short training complete!")


if __name__ == "__main__":
    main()
