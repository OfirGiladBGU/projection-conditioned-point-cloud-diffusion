"""Training script for Unsupervised Point-RT V8.

Key difference from V7: Uses physics-based losses (Sinkhorn + AdaptiveRepulsion)
instead of GT supervision. No Lloyd points needed!

Benefits:
- Can train on any image (infinite data)
- Not limited by GT quality artifacts
- Learns true equilibrium directly
"""

import torch
import torch.nn as nn
from torch.optim import AdamW
from torch.optim.lr_scheduler import CosineAnnealingLR, LinearLR, SequentialLR
from tqdm import tqdm
import os
import numpy as np
from pathlib import Path
import time

from config import Config
from point_rt import PointRT
from fast_init import fast_density_initialization, create_point_input_vectors
from losses import UnsupervisedStipplingLoss, HybridStipplingLoss
from dataset import create_dataloaders

# Optional wandb logging
try:
    import wandb
    HAS_WANDB = True
except ImportError:
    HAS_WANDB = False


def create_optimizer(model: nn.Module, config: Config):
    """Create optimizer with optional backbone freezing."""
    
    # Optionally freeze ResNet backbone to preserve pretrained features
    if hasattr(model, 'freeze_backbone'):
        freeze_backbone = not config.model.resnet_pretrained
        if freeze_backbone:
            model.freeze_backbone(True)
            print("Frozen ResNet backbone")
    
    optimizer = AdamW(
        model.parameters(),
        lr=config.training.learning_rate,
        weight_decay=config.training.weight_decay,
    )
    
    return optimizer


def create_scheduler(optimizer, config: Config, num_batches_per_epoch: int):
    """Create learning rate scheduler."""
    
    if config.training.lr_scheduler == "constant":
        return None
    
    total_steps = int(num_batches_per_epoch * config.training.num_epochs)
    warmup_steps = int(num_batches_per_epoch * config.training.warmup_epochs)
    
    if config.training.lr_scheduler == "warmup_cosine":
        warmup_scheduler = LinearLR(
            optimizer,
            start_factor=0.1,
            total_iters=warmup_steps,
        )
        cosine_scheduler = CosineAnnealingLR(
            optimizer,
            T_max=total_steps - warmup_steps,
        )
        scheduler = SequentialLR(
            optimizer,
            [warmup_scheduler, cosine_scheduler],
            milestones=[warmup_steps],
        )
    elif config.training.lr_scheduler == "cosine":
        scheduler = CosineAnnealingLR(
            optimizer,
            T_max=total_steps,
        )
    else:
        scheduler = None
    
    return scheduler


def train_step_unsupervised(
    batch: dict,
    model: nn.Module,
    loss_fn: nn.Module,
    device: str,
) -> tuple:
    """Single UNSUPERVISED training step.
    
    Uses physics-based losses (Sinkhorn + AdaptiveRepulsion) - no GT needed!
    
    Args:
        batch: dict with 'image' (and optionally 'lloyd_points' for hybrid)
        model: PointRT model
        loss_fn: UnsupervisedStipplingLoss or HybridStipplingLoss
        device: 'cuda' or 'cpu'
    
    Returns:
        (loss, loss_dict)
    """
    image = batch['image'].to(device)
    
    # Fast initialization (density-based point placement)
    with torch.no_grad():
        init_points = fast_density_initialization(image, model.n_points)
        input_vecs = create_point_input_vectors(init_points, image)
    
    # Forward pass
    pred_points = model(input_vecs, image)
    
    # Compute UNSUPERVISED loss (Sinkhorn + AdaptiveRepulsion)
    # No GT needed - loss purely from physics constraints!
    if isinstance(loss_fn, HybridStipplingLoss) and 'lloyd_points' in batch and batch['lloyd_points'] is not None:
        lloyd_points = batch['lloyd_points'].to(device)
        loss, loss_dict = loss_fn(pred_points, image, lloyd_points)
    else:
        loss, loss_dict = loss_fn(pred_points, image)
    
    return loss, loss_dict


def train_epoch(
    model: nn.Module,
    train_loader,
    optimizer,
    loss_fn: nn.Module,
    device: str,
    config: Config,
    epoch: int,
    scheduler=None,
) -> dict:
    """Train for one epoch.
    
    Returns:
        Metrics dict with average loss components
    """
    model.train()
    
    losses_accum = {
        'total_loss': 0.0,
        'sinkhorn_loss': 0.0,
        'repulsion_loss': 0.0,
    }
    num_batches = 0
    
    pbar = tqdm(train_loader, desc=f"Epoch {epoch}")
    
    for batch_idx, batch in enumerate(pbar):
        try:
            # Forward pass (UNSUPERVISED)
            loss, loss_dict = train_step_unsupervised(batch, model, loss_fn, device)
            
            # Backward pass
            optimizer.zero_grad()
            loss.backward()
            
            # Gradient clipping
            nn.utils.clip_grad_norm_(model.parameters(), config.training.max_grad_norm)
            
            # Update
            optimizer.step()
            if scheduler is not None:
                scheduler.step()
            
            # Accumulate losses
            for key in losses_accum:
                if key in loss_dict:
                    losses_accum[key] += loss_dict[key]
            num_batches += 1
            
            # Update progress bar
            pbar.set_postfix({
                'loss': loss.item(),
                'sinkhorn': loss_dict.get('sinkhorn_loss', 0),
                'repulsion': loss_dict.get('repulsion_loss', 0),
                'lr': optimizer.param_groups[0]['lr'],
            })
            
            # Log
            if (batch_idx + 1) % config.training.log_every == 0:
                log_dict = {
                    'epoch': epoch,
                    'batch': batch_idx,
                    'learning_rate': optimizer.param_groups[0]['lr'],
                }
                log_dict.update(loss_dict)
                
                if HAS_WANDB and config.training.use_wandb:
                    wandb.log(log_dict)
        
        except Exception as e:
            print(f"Error in batch {batch_idx}: {e}")
            import traceback
            traceback.print_exc()
            continue
    
    # Average metrics
    metrics = {}
    for key in losses_accum:
        metrics[key] = losses_accum[key] / max(num_batches, 1)
    
    return metrics


def val_step(
    model: nn.Module,
    val_loader,
    loss_fn: nn.Module,
    device: str,
    config: Config,
) -> dict:
    """Validation loop (unsupervised)."""
    
    model.eval()
    
    losses_accum = {
        'total_loss': 0.0,
        'sinkhorn_loss': 0.0,
        'repulsion_loss': 0.0,
    }
    num_batches = 0
    
    with torch.no_grad():
        for batch in tqdm(val_loader, desc="Validation"):
            try:
                loss, loss_dict = train_step_unsupervised(batch, model, loss_fn, device)
                
                for key in losses_accum:
                    if key in loss_dict:
                        losses_accum[key] += loss_dict[key]
                num_batches += 1
            except Exception as e:
                print(f"Error in val batch: {e}")
                continue
    
    metrics = {}
    for key in losses_accum:
        metrics[key] = losses_accum[key] / max(num_batches, 1)
    
    return metrics


def main():
    """Main UNSUPERVISED training script.
    
    Uses Sinkhorn + AdaptiveRepulsion losses instead of GT supervision.
    """
    
    # Load config
    config = Config.default()
    
    # Setup
    device = config.device
    print(f"Using device: {device}")
    print(f"\n{'='*60}")
    print("UNSUPERVISED POINT-RT TRAINING (V8)")
    print("Using physics-based losses: Sinkhorn + AdaptiveRepulsion")
    print("No ground truth points needed!")
    print(f"{'='*60}\n")
    
    # Create output directory
    output_dir = Path(config.training.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # Initialize wandb
    if HAS_WANDB and config.training.use_wandb:
        wandb.init(
            project=config.training.wandb_project,
            name=config.training.wandb_run_name,
            config=config.to_dict(),
        )
    
    # Create model
    print("Creating model...")
    model = PointRT(
        n_points=config.model.n_points,
        dim=config.model.dim,
        n_layers=config.model.n_layers,
        n_heads=config.model.n_heads,
        image_size=config.model.image_size,
        use_resnet_backbone=config.model.use_resnet_backbone,
        resnet_pretrained=config.model.resnet_pretrained,
        use_displacement=config.model.use_displacement,
        fourier_freqs=config.model.fourier_freqs,
        dropout=config.model.dropout,
    )
    model = model.to(device)
    
    print(f"Model parameters: {model.get_num_params():,}")
    
    # Create dataloaders
    # For unsupervised, lloyd_dir can be None!
    print("\nCreating dataloaders...")
    print("NOTE: Ground truth (lloyd_dir) optional for unsupervised training")
    
    train_loader, val_loader = create_dataloaders(
        source_dir=config.data.source_dir,
        batch_size=config.training.batch_size,
        num_workers=config.data.num_workers,
        image_size=config.data.image_size,
        num_points=config.data.num_points,
        lloyd_dir=config.data.lloyd_dir,  # Can be None for pure unsupervised
        val_split=config.data.val_split,
    )
    
    # Create optimizer and scheduler
    optimizer = create_optimizer(model, config)
    scheduler = create_scheduler(optimizer, config, len(train_loader))
    
    # Create loss function (UNSUPERVISED)
    print("\nCreating UNSUPERVISED loss function...")
    print(f"  Sinkhorn weight: {config.training.sinkhorn_weight}")
    print(f"  Repulsion weight: {config.training.repulsion_weight}")
    
    if config.training.gt_weight > 0 and config.data.lloyd_dir is not None:
        print(f"  GT weight: {config.training.gt_weight} (hybrid mode)")
        loss_fn = HybridStipplingLoss(
            sinkhorn_weight=config.training.sinkhorn_weight,
            repulsion_weight=config.training.repulsion_weight,
            gt_weight=config.training.gt_weight,
            base_radius=config.training.base_radius,
            sinkhorn_blur=config.training.sinkhorn_blur,
        ).to(device)
    else:
        print("  Pure unsupervised mode (no GT)")
        loss_fn = UnsupervisedStipplingLoss(
            sinkhorn_weight=config.training.sinkhorn_weight,
            repulsion_weight=config.training.repulsion_weight,
            base_radius=config.training.base_radius,
            sinkhorn_blur=config.training.sinkhorn_blur,
        ).to(device)
    
    # Training loop
    print(f"\nStarting UNSUPERVISED training for {config.training.num_epochs} epochs...")
    print(f"Train batches per epoch: {len(train_loader)}")
    print(f"Val batches: {len(val_loader)}")
    
    start_time = time.time()
    
    for epoch in range(config.training.num_epochs):
        # Train
        train_metrics = train_epoch(
            model, train_loader, optimizer, loss_fn, device, config, epoch, scheduler
        )
        
        # Validate
        val_metrics = val_step(model, val_loader, loss_fn, device, config)
        
        # Log
        log_dict = {
            'epoch': epoch,
            'train/loss': train_metrics['total_loss'],
            'train/sinkhorn_loss': train_metrics['sinkhorn_loss'],
            'train/repulsion_loss': train_metrics['repulsion_loss'],
            'val/loss': val_metrics['total_loss'],
            'val/sinkhorn_loss': val_metrics['sinkhorn_loss'],
            'val/repulsion_loss': val_metrics['repulsion_loss'],
        }
        
        print(f"\nEpoch {epoch}")
        print(f"  Train Loss: {train_metrics['total_loss']:.4f} (Sinkhorn: {train_metrics['sinkhorn_loss']:.4f}, Repulsion: {train_metrics['repulsion_loss']:.4f})")
        print(f"  Val Loss: {val_metrics['total_loss']:.4f}")
        
        if HAS_WANDB and config.training.use_wandb:
            wandb.log(log_dict)
        
        # Save checkpoint
        if (epoch + 1) % 10 == 0:
            ckpt_path = output_dir / f"checkpoint_epoch_{epoch}.pt"
            torch.save({
                'epoch': epoch,
                'model_state': model.state_dict(),
                'optimizer_state': optimizer.state_dict(),
                'config': config,
            }, ckpt_path)
            print(f"  Saved checkpoint: {ckpt_path}")
    
    elapsed = time.time() - start_time
    print(f"\nTraining complete! Elapsed time: {elapsed/3600:.1f} hours")
    
    # Save final model
    final_path = output_dir / "model_final.pt"
    torch.save(model.state_dict(), final_path)
    print(f"Final model saved: {final_path}")
    
    if HAS_WANDB and config.training.use_wandb:
        wandb.finish()


if __name__ == "__main__":
    main()
