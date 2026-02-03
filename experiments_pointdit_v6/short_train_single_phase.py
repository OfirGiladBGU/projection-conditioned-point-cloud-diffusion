"""
Short Training Script for V6.1 Proof of Concept (PoC)

V6.1 SINGLE-PHASE STRATEGY:
- With Rectified Flow + Hard Sinkhorn Matching, the model learns structure and
  spacing SIMULTANEOUSLY. No need for two-phase training.
- The "Straight Paths" from exact matching mean the velocity vectors already
  point perfectly to correct locations - implicit curriculum built-in.

LOSS FORMULA (Unified from Epoch 0):
  Loss = MSE (velocity) + 10.0 * Chamfer + 0.1 * Sinkhorn
  - MSE: Main engine (Flow Matching velocity prediction)
  - Chamfer (10.0): High weight prevents "drifting" off shapes
  - Sinkhorn (0.1): Low weight "vibe check" for global density
  - Repulsion (0.0): OFF - breaks the straight paths

PoC STRATEGY (Overfit Test):
- 50 epochs on 200 images
- If loss drops and training samples look crystalline, architecture is validated
- Evaluate on TRAINING SET (not validation) - we're testing capacity, not generalization

Usage:
    python short_train.py
    
    # On GPU node:
    ssh cs-6000-03 "cd /groups/asharf_group/ofirgila/projection-conditioned-point-cloud-diffusion/experiments_pointdit_v6 && conda run -n pc2 python short_train.py"
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
    # =================================================================
    # V6.1 SINGLE-PHASE CONFIGURATION (PoC / Overfit Test)
    # =================================================================
    BATCH_SIZE = 4           # Small batch for quick testing
    NUM_EPOCHS = 50          # Single unified phase
    NUM_TRAIN_SAMPLES = 200  # Small subset for capacity test
    EVAL_EVERY = 5           # Evaluate on training set
    NUM_INFERENCE_STEPS = 30
    
    # UNIFIED LOSS WEIGHTS (From Epoch 0)
    # These work because Hard Sinkhorn matching already solves OT at input level
    CHAMFER_WEIGHT = 10.0    # High - prevents drifting off shapes
    SINKHORN_WEIGHT = 0.1    # Low - just a "vibe check" for global density
    REPULSION_WEIGHT = 0.0   # OFF - harmful, breaks straight paths
    
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    print(f"Device: {device}")
    print(f"\n" + "="*60)
    print(f"V6.1 SINGLE-PHASE PoC (Overfit Test)")
    print(f"="*60)
    print(f"Strategy: Test model CAPACITY on {NUM_TRAIN_SAMPLES} images")
    print(f"If training samples look crystalline -> architecture validated")
    print(f"\nConfiguration:")
    print(f"  - Epochs: {NUM_EPOCHS} (single phase, no switching)")
    print(f"  - Matching: Hard Sinkhorn (GPU, argmax, no ghost points)")
    print(f"  - Loss: MSE + {CHAMFER_WEIGHT}*Chamfer + {SINKHORN_WEIGHT}*Sinkhorn")
    print(f"  - Repulsion: OFF (breaks straight paths)")
    
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
    
    # Randomly sample training data from full dataset
    print(f"Randomly sampling {NUM_TRAIN_SAMPLES} samples from {len(train_loader.dataset)} total training samples")
    
    import random
    random.seed(42)  # For reproducibility
    torch.manual_seed(42)
    
    # Get random indices
    total_samples = len(train_loader.dataset)
    random_indices = random.sample(range(total_samples), NUM_TRAIN_SAMPLES)
    
    # Create subset dataset
    from torch.utils.data import Subset, DataLoader
    train_subset = Subset(train_loader.dataset, random_indices)
    train_subset_loader = DataLoader(
        train_subset,
        batch_size=BATCH_SIZE,
        shuffle=True,  # Shuffle each epoch
        num_workers=0,
    )
    
    train_data = list(train_subset_loader)
    print(f"Training on {len(train_data) * BATCH_SIZE} randomly sampled samples")
    
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
    
    # Single-phase config (no switching needed with V6.1)
    loss_config = {
        'chamfer_weight': CHAMFER_WEIGHT,
        'repulsion_weight': REPULSION_WEIGHT,
        'sinkhorn_weight': SINKHORN_WEIGHT,
    }
    
    # Training loop
    history = {
        'train_loss': [], 'chamfer': [], 'sinkhorn': [], 'repulsion': [],
        'val_mean_nn': [], 'val_cv': [], 'val_chamfer': []
    }
    
    gt_metrics = compute_metrics(val_gt_points, val_gt_points)
    print(f"\nGT Metrics: Mean NN={gt_metrics['mean_nn']:.4f}, CV={gt_metrics['cv']:.3f}")
    
    for epoch in range(1, NUM_EPOCHS + 1):
        model.train()
        epoch_losses = {'loss': [], 'chamfer': [], 'sinkhorn': [], 'repulsion': []}
        
        pbar = tqdm(train_data, desc=f"Epoch {epoch}/{NUM_EPOCHS} (Single Phase)")
        for batch in pbar:
            image = batch['image'].to(device)
            points = batch['points'].to(device)
            
            optimizer.zero_grad()
            
            # V6.1: Hard Sinkhorn matching (GPU, argmax, no ghost points)
            # Loss: MSE + 10.0*Chamfer + 0.1*Sinkhorn (unified from epoch 0)
            result = train_step(
                model, scheduler, points, image, device,
                sinkhorn_weight=loss_config['sinkhorn_weight'],
                chamfer_weight=loss_config['chamfer_weight'],
                repulsion_weight=loss_config['repulsion_weight'],
                use_ot_matching=True,    # Enable trajectory straightening
                use_gpu_sinkhorn=True,   # V6.1: Hard Sinkhorn (argmax, no averaging)
                sinkhorn_epsilon=0.01,   # Sharp matching
                sinkhorn_iterations=5,   # 5 iterations enough in log-space
            )
            
            result['loss'].backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()
            
            epoch_losses['loss'].append(result['loss'].item())
            epoch_losses['chamfer'].append(result['chamfer'])
            epoch_losses['sinkhorn'].append(result['sinkhorn'])
            if 'repulsion' in result:
                epoch_losses['repulsion'].append(result['repulsion'])
            
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
            'chamfer_weight': CHAMFER_WEIGHT,
            'sinkhorn_weight': SINKHORN_WEIGHT,
            'repulsion_weight': REPULSION_WEIGHT,
            'strategy': 'single_phase_v6.1',
        },
        'gt_metrics': gt_metrics,
    }, weights_path)
    print(f"  - Model weights: {weights_path}")


if __name__ == "__main__":
    main()
