"""
Short Training Script for Quick Sinkhorn+Chamfer Loss Evaluation

This script runs a quick training on a small subset to verify the new loss combination.
Use this before committing to a full training run.

Usage:
    python short_train.py
    
    # Or on GPU node:
    ssh ise-cpu256-26 "cd /groups/asharf_group/ofirgila/projection-conditioned-point-cloud-diffusion/experiments_pointdit_v6 && source ~/.bashrc && conda activate pc2 && python short_train.py"
"""

import torch
import torch.nn as nn
from pathlib import Path
import matplotlib.pyplot as plt
import numpy as np
from tqdm import tqdm
import sys

sys.path.insert(0, str(Path(__file__).parent))

from config import Config
from dataset import create_dataloaders
from model.point_dit import PointDiT
from diffusion import DDPMScheduler, train_step, sample, ChamferLoss


def compute_metrics(pred_points: torch.Tensor, gt_points: torch.Tensor):
    """Compute evaluation metrics."""
    B, N, _ = pred_points.shape
    
    # NN distances for blue noise quality
    dists = torch.cdist(pred_points, pred_points)
    mask = torch.eye(N, device=pred_points.device).bool().unsqueeze(0).expand(B, -1, -1)
    dists = dists.masked_fill(mask, float('inf'))
    nn_dists, _ = dists.min(dim=2)
    
    mean_nn = nn_dists.mean().item()
    cv = nn_dists.std().item() / (mean_nn + 1e-8)
    
    # Chamfer distance
    chamfer_fn = ChamferLoss()
    chamfer = chamfer_fn(pred_points, gt_points).item()
    
    return {'mean_nn': mean_nn, 'cv': cv, 'chamfer': chamfer}


def visualize_results(
    image: torch.Tensor,
    pred_points: torch.Tensor,
    gt_points: torch.Tensor,
    pred_metrics: dict,
    gt_metrics: dict,
    epoch: int,
    output_dir: Path,
):
    """Create visualization of results."""
    fig, axes = plt.subplots(1, 3, figsize=(15, 5))
    
    img = image[0, 0].cpu().numpy()
    pred = pred_points[0].cpu().numpy()
    gt = gt_points[0].cpu().numpy()
    
    # GT
    axes[0].imshow(img, cmap='gray', extent=[-1, 1, -1, 1], origin='lower')
    axes[0].scatter(gt[:, 0], gt[:, 1], c='red', s=2, alpha=0.8)
    axes[0].set_title(f"Ground Truth\nNN={gt_metrics['mean_nn']:.4f}, CV={gt_metrics['cv']:.3f}")
    axes[0].set_xlim(-1, 1)
    axes[0].set_ylim(-1, 1)
    
    # Predicted
    axes[1].imshow(img, cmap='gray', extent=[-1, 1, -1, 1], origin='lower')
    axes[1].scatter(pred[:, 0], pred[:, 1], c='blue', s=2, alpha=0.8)
    axes[1].set_title(f"Predicted (Epoch {epoch})\nNN={pred_metrics['mean_nn']:.4f}, CV={pred_metrics['cv']:.3f}\nChamfer={pred_metrics['chamfer']:.6f}")
    axes[1].set_xlim(-1, 1)
    axes[1].set_ylim(-1, 1)
    
    # Side by side
    axes[2].imshow(img, cmap='gray', extent=[-1, 1, -1, 1], origin='lower')
    axes[2].scatter(gt[:, 0], gt[:, 1], c='red', s=2, alpha=0.5, label='GT')
    axes[2].scatter(pred[:, 0], pred[:, 1], c='blue', s=2, alpha=0.5, label='Pred')
    axes[2].set_title("Overlay")
    axes[2].legend()
    axes[2].set_xlim(-1, 1)
    axes[2].set_ylim(-1, 1)
    
    for ax in axes:
        ax.set_aspect('equal')
    
    plt.tight_layout()
    plt.savefig(output_dir / f'epoch_{epoch:03d}.png', dpi=150)
    plt.close()


def main():
    # Configuration - small batch for quick testing
    BATCH_SIZE = 4
    NUM_EPOCHS = 20
    NUM_TRAIN_SAMPLES = 200  # Small subset
    EVAL_EVERY = 5
    NUM_INFERENCE_STEPS = 30
    
    # Loss weights (from our evaluation)
    SINKHORN_WEIGHT = 1.0
    CHAMFER_WEIGHT = 10.0
    
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    print(f"Device: {device}")
    print(f"Sinkhorn weight: {SINKHORN_WEIGHT}, Chamfer weight: {CHAMFER_WEIGHT}")
    
    # Create output directory
    output_dir = Path(__file__).parent / 'outputs_short_train'
    output_dir.mkdir(exist_ok=True)
    
    # Load config and data
    config = Config.default()
    train_loader, val_loader = create_dataloaders(
        source_dir=config.data.source_dir,
        target_dir=config.data.target_dir,
        batch_size=BATCH_SIZE,
        image_size=config.data.image_size,
        num_points=config.data.num_points,
        val_split=config.data.val_split,
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
    val_gt_points = val_batch['points'].to(device)
    
    # Model and optimizer
    model = PointDiT(
        n_points=config.data.num_points,
        dim=config.model.dim,
        n_layers=config.model.n_layers,
        n_heads=config.model.n_heads,
        image_size=config.data.image_size,
        dropout=config.model.dropout,
    ).to(device)
    
    scheduler = DDPMScheduler(
        num_train_timesteps=config.diffusion.num_train_timesteps,
        beta_start=config.diffusion.beta_start,
        beta_end=config.diffusion.beta_end,
        beta_schedule=config.diffusion.beta_schedule,
    )
    
    optimizer = torch.optim.AdamW(model.parameters(), lr=config.training.learning_rate)
    
    print(f"Model parameters: {sum(p.numel() for p in model.parameters()):,}")
    
    # Training loop
    history = {
        'train_loss': [], 'chamfer': [], 'sinkhorn': [],
        'val_mean_nn': [], 'val_cv': [], 'val_chamfer': []
    }
    
    gt_metrics = compute_metrics(val_gt_points, val_gt_points)
    print(f"\nGT Metrics: Mean NN={gt_metrics['mean_nn']:.4f}, CV={gt_metrics['cv']:.3f}")
    
    for epoch in range(1, NUM_EPOCHS + 1):
        model.train()
        epoch_losses = {'loss': [], 'chamfer': [], 'sinkhorn': []}
        
        pbar = tqdm(train_data, desc=f"Epoch {epoch}/{NUM_EPOCHS}")
        for batch in pbar:
            image = batch['image'].to(device)
            points = batch['points'].to(device)
            
            optimizer.zero_grad()
            
            result = train_step(
                model, scheduler, points, image, device,
                sinkhorn_weight=SINKHORN_WEIGHT,
                chamfer_weight=CHAMFER_WEIGHT,
            )
            
            result['loss'].backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()
            
            epoch_losses['loss'].append(result['loss'].item())
            epoch_losses['chamfer'].append(result['chamfer'])
            epoch_losses['sinkhorn'].append(result['sinkhorn'])
            
            pbar.set_postfix({
                'loss': f"{result['loss'].item():.4f}",
                'chamfer': f"{result['chamfer']:.6f}",
                'sinkhorn': f"{result['sinkhorn']:.4f}",
            })
        
        # Record training metrics
        avg_loss = np.mean(epoch_losses['loss'])
        avg_chamfer = np.mean(epoch_losses['chamfer'])
        avg_sinkhorn = np.mean(epoch_losses['sinkhorn'])
        
        history['train_loss'].append(avg_loss)
        history['chamfer'].append(avg_chamfer)
        history['sinkhorn'].append(avg_sinkhorn)
        
        print(f"Epoch {epoch}: Loss={avg_loss:.4f}, Chamfer={avg_chamfer:.6f}, Sinkhorn={avg_sinkhorn:.4f}")
        
        # Evaluation
        if epoch % EVAL_EVERY == 0 or epoch == 1:
            model.eval()
            with torch.no_grad():
                pred_points = sample(
                    model, scheduler, val_image,
                    num_points=config.data.num_points,
                    num_inference_steps=NUM_INFERENCE_STEPS,
                    device=device,
                    show_progress=False,
                    init_from_density=True,
                )
                
                pred_metrics = compute_metrics(pred_points, val_gt_points)
                
                history['val_mean_nn'].append(pred_metrics['mean_nn'])
                history['val_cv'].append(pred_metrics['cv'])
                history['val_chamfer'].append(pred_metrics['chamfer'])
                
                nn_ratio = pred_metrics['mean_nn'] / gt_metrics['mean_nn']
                print(f"  Eval: Mean NN={pred_metrics['mean_nn']:.4f} (ratio={nn_ratio:.3f}), "
                      f"CV={pred_metrics['cv']:.3f}, Chamfer={pred_metrics['chamfer']:.6f}")
                
                # Save visualization
                visualize_results(
                    val_image, pred_points, val_gt_points,
                    pred_metrics, gt_metrics, epoch, output_dir
                )
    
    # Plot training curves
    fig, axes = plt.subplots(2, 2, figsize=(12, 10))
    
    # Training loss
    axes[0, 0].plot(history['train_loss'], label='Total Loss')
    axes[0, 0].set_xlabel('Epoch')
    axes[0, 0].set_ylabel('Loss')
    axes[0, 0].set_title('Training Loss')
    axes[0, 0].legend()
    
    # Individual losses
    axes[0, 1].plot(history['chamfer'], label='Chamfer')
    axes[0, 1].plot(history['sinkhorn'], label='Sinkhorn')
    axes[0, 1].set_xlabel('Epoch')
    axes[0, 1].set_ylabel('Loss')
    axes[0, 1].set_title('Individual Losses')
    axes[0, 1].legend()
    
    # Blue noise quality
    eval_epochs = list(range(EVAL_EVERY, NUM_EPOCHS + 1, EVAL_EVERY))
    if 1 not in eval_epochs:
        eval_epochs = [1] + eval_epochs
    
    axes[1, 0].plot(eval_epochs[:len(history['val_mean_nn'])], history['val_mean_nn'], 'b-o', label='Pred Mean NN')
    axes[1, 0].axhline(y=gt_metrics['mean_nn'], color='r', linestyle='--', label=f"GT Mean NN: {gt_metrics['mean_nn']:.4f}")
    axes[1, 0].set_xlabel('Epoch')
    axes[1, 0].set_ylabel('Mean NN Distance')
    axes[1, 0].set_title('Blue Noise Quality (higher = better spaced)')
    axes[1, 0].legend()
    
    # Chamfer distance
    axes[1, 1].plot(eval_epochs[:len(history['val_chamfer'])], history['val_chamfer'], 'g-o')
    axes[1, 1].set_xlabel('Epoch')
    axes[1, 1].set_ylabel('Chamfer Distance')
    axes[1, 1].set_title('Position Accuracy (lower = better)')
    
    plt.tight_layout()
    plt.savefig(output_dir / 'training_curves.png', dpi=150)
    plt.close()
    
    print(f"\nResults saved to: {output_dir}")
    print(f"  - Training curves: {output_dir}/training_curves.png")
    print(f"  - Per-epoch visualizations: {output_dir}/epoch_*.png")
    
    # Final summary
    print("\n" + "="*60)
    print("FINAL SUMMARY")
    print("="*60)
    print(f"GT Mean NN: {gt_metrics['mean_nn']:.4f}")
    if history['val_mean_nn']:
        final_nn = history['val_mean_nn'][-1]
        final_chamfer = history['val_chamfer'][-1]
        print(f"Final Mean NN: {final_nn:.4f} (ratio={final_nn/gt_metrics['mean_nn']:.3f})")
        print(f"Final Chamfer: {final_chamfer:.6f}")


if __name__ == "__main__":
    main()
