"""Training script for Point-RT."""

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
from model.point_rt import PointRT
from fast_init import fast_density_initialization, create_point_input_vectors
from losses import PointRTLoss
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


def train_step(
    batch: dict,
    model: nn.Module,
    loss_fn: nn.Module,
    device: str,
) -> tuple:
    """Single training step.
    
    Args:
        batch: dict with 'image' and 'lloyd_points'
        model: PointRT model
        loss_fn: PointRTLoss
        device: 'cuda' or 'cpu'
    
    Returns:
        (loss, loss_dict)
    """
    image = batch['image'].to(device)
    lloyd_points = batch['lloyd_points']
    
    # Fast initialization
    with torch.no_grad():
        init_points = fast_density_initialization(image, model.n_points)
        input_vecs = create_point_input_vectors(init_points, image)
    
    # Forward pass
    pred_points = model(input_vecs, image)
    
    # Handle ground truth
    if lloyd_points is not None:
        lloyd_points = lloyd_points.to(device)
    else:
        # Fallback: use perturbed initialization as target
        lloyd_points = init_points + torch.randn_like(init_points) * 0.1
    
    # Compute loss
    loss, loss_dict = loss_fn(pred_points, lloyd_points)
    
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
        'mse_loss': 0.0,
        'repulsion_loss': 0.0,
        'diversity_loss': 0.0,
    }
    num_batches = 0
    
    pbar = tqdm(train_loader, desc=f"Epoch {epoch}")
    
    for batch_idx, batch in enumerate(pbar):
        try:
            # Forward pass
            loss, loss_dict = train_step(batch, model, loss_fn, device)
            
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
                losses_accum[key] += loss_dict[key]
            num_batches += 1
            
            # Update progress bar
            pbar.set_postfix({
                'loss': loss.item(),
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
    """Validation loop."""
    
    model.eval()
    
    losses_accum = {
        'total_loss': 0.0,
        'mse_loss': 0.0,
        'repulsion_loss': 0.0,
        'diversity_loss': 0.0,
    }
    num_batches = 0
    
    with torch.no_grad():
        for batch in tqdm(val_loader, desc="Validation"):
            try:
                loss, loss_dict = train_step(batch, model, loss_fn, device)
                
                for key in losses_accum:
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
    """Main training script."""
    
    # Load config
    config = Config.default()
    
    # Setup
    device = config.device
    print(f"Using device: {device}")
    
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
    print("Creating dataloaders...")
    train_loader, val_loader = create_dataloaders(
        source_dir=config.data.source_dir,
        batch_size=config.training.batch_size,
        num_workers=config.data.num_workers,
        image_size=config.data.image_size,
        num_points=config.data.num_points,
        lloyd_dir=config.data.lloyd_dir,
        val_split=config.data.val_split,
    )
    
    # Create optimizer and scheduler
    optimizer = create_optimizer(model, config)
    scheduler = create_scheduler(optimizer, config, len(train_loader))
    
    # Create loss function
    loss_fn = PointRTLoss(
        mse_weight=config.training.mse_weight,
        repulsion_weight=config.training.repulsion_weight,
        diversity_weight=config.training.diversity_weight,
    ).to(device)
    
    # Training loop
    print(f"\nStarting training for {config.training.num_epochs} epochs...")
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
            'train/mse_loss': train_metrics['mse_loss'],
            'train/repulsion_loss': train_metrics['repulsion_loss'],
            'val/loss': val_metrics['total_loss'],
            'val/mse_loss': val_metrics['mse_loss'],
        }
        
        print(f"\nEpoch {epoch}")
        print(f"  Train Loss: {train_metrics['total_loss']:.4f}")
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
