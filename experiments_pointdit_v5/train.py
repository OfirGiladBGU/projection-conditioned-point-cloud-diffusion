"""Training script for Point-DiT."""

import torch
import torch.nn as nn
from torch.optim import AdamW
from torch.optim.lr_scheduler import CosineAnnealingLR
from tqdm import tqdm
import os
import numpy as np
from pathlib import Path

from config import Config
from model import PointDiT
from diffusion import DDPMScheduler, train_step, sample
from dataset import create_dataloaders


def train_epoch(
    model: nn.Module,
    scheduler: DDPMScheduler,
    dataloader,
    optimizer,
    device: str,
    epoch: int,
    config: Config,
):
    """Train for one epoch."""
    model.train()
    total_loss = 0
    
    pbar = tqdm(dataloader, desc=f"Epoch {epoch}")
    for step, batch in enumerate(pbar):
        image = batch['image'].to(device)
        points = batch['points'].to(device)
        
        # Forward pass
        loss = train_step(model, scheduler, points, image, device)
        
        # Backward pass
        optimizer.zero_grad()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), config.training.max_grad_norm)
        optimizer.step()
        
        # Logging
        total_loss += loss.item()
        pbar.set_postfix({'loss': loss.item()})
        
        if step % config.training.log_every == 0:
            avg_loss = total_loss / (step + 1)
            print(f"Epoch {epoch}, Step {step}, Loss: {loss.item():.6f}, Avg: {avg_loss:.6f}")
    
    return total_loss / len(dataloader)


@torch.no_grad()
def validate(
    model: nn.Module,
    scheduler: DDPMScheduler,
    dataloader,
    device: str,
):
    """Validate the model."""
    model.eval()
    total_loss = 0
    
    for batch in tqdm(dataloader, desc="Validating"):
        image = batch['image'].to(device)
        points = batch['points'].to(device)
        
        loss = train_step(model, scheduler, points, image, device)
        total_loss += loss.item()
    
    return total_loss / len(dataloader)


def main():
    # Load config
    config = Config.default()
    
    # Set seed
    torch.manual_seed(config.seed)
    np.random.seed(config.seed)
    
    # Device
    device = torch.device(config.device if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")
    
    # Create output directory
    os.makedirs(config.training.output_dir, exist_ok=True)
    
    # Create model
    model = PointDiT(
        n_points=config.model.n_points,
        dim=config.model.dim,
        n_layers=config.model.n_layers,
        n_heads=config.model.n_heads,
        image_size=config.model.image_size,
        dropout=config.model.dropout,
    ).to(device)
    
    print(f"Model parameters: {model.get_num_params():,}")
    
    # Create scheduler
    scheduler = DDPMScheduler(
        num_train_timesteps=config.diffusion.num_train_timesteps,
        beta_start=config.diffusion.beta_start,
        beta_end=config.diffusion.beta_end,
        beta_schedule=config.diffusion.beta_schedule,
    )
    
    # Create dataloaders
    print(f"Loading dataset from: {config.data.source_dir}")
    print(f"Target directory: {config.data.target_dir}")
    
    train_loader, val_loader = create_dataloaders(
        source_dir=config.data.source_dir,
        target_dir=config.data.target_dir,
        batch_size=config.training.batch_size,
        num_workers=config.num_workers,
        val_split=config.data.val_split,
        image_size=config.data.image_size,
        num_points=config.data.num_points,
    )
    
    print(f"Train batches: {len(train_loader)}, Val batches: {len(val_loader)}")
    
    # Optimizer
    optimizer = AdamW(
        model.parameters(),
        lr=config.training.learning_rate,
        weight_decay=config.training.weight_decay,
    )
    
    # LR Scheduler
    if config.training.lr_scheduler == 'cosine':
        lr_scheduler = CosineAnnealingLR(optimizer, T_max=config.training.num_epochs)
    elif config.training.lr_scheduler == 'warmup_cosine':
        # Simple warmup + cosine
        def lr_lambda(epoch):
            if epoch < config.training.warmup_epochs:
                return epoch / config.training.warmup_epochs
            else:
                return 0.5 * (1 + np.cos(
                    np.pi * (epoch - config.training.warmup_epochs) / 
                    (config.training.num_epochs - config.training.warmup_epochs)
                ))
        lr_scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda)
    else:
        lr_scheduler = None
    
    # Load checkpoint if provided
    start_epoch = 0
    best_val_loss = float('inf')
    
    if config.training.checkpoint_path and os.path.exists(config.training.checkpoint_path):
        print(f"Loading checkpoint: {config.training.checkpoint_path}")
        checkpoint = torch.load(config.training.checkpoint_path, map_location=device)
        model.load_state_dict(checkpoint['model_state_dict'])
        optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        start_epoch = checkpoint['epoch'] + 1
        best_val_loss = checkpoint.get('best_val_loss', float('inf'))
        print(f"Resuming from epoch {start_epoch}")
    
    # Training loop
    print(f"\nStarting training for {config.training.num_epochs} epochs...")
    
    for epoch in range(start_epoch, config.training.num_epochs):
        # Train
        train_loss = train_epoch(
            model, scheduler, train_loader, optimizer, device, epoch, config
        )
        
        # Validate
        val_loss = validate(model, scheduler, val_loader, device)
        
        print(f"\nEpoch {epoch}: Train Loss = {train_loss:.6f}, Val Loss = {val_loss:.6f}")
        
        # Update LR
        if lr_scheduler is not None:
            lr_scheduler.step()
            print(f"Learning rate: {optimizer.param_groups[0]['lr']:.6e}")
        
        # Save checkpoint
        checkpoint = {
            'epoch': epoch,
            'model_state_dict': model.state_dict(),
            'optimizer_state_dict': optimizer.state_dict(),
            'train_loss': train_loss,
            'val_loss': val_loss,
            'best_val_loss': best_val_loss,
            'config': config,
        }
        
        # Save latest
        torch.save(checkpoint, os.path.join(config.training.output_dir, 'checkpoint_latest.pth'))
        
        # Save best
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            checkpoint['best_val_loss'] = best_val_loss
            torch.save(checkpoint, os.path.join(config.training.output_dir, 'checkpoint_best.pth'))
            print(f"✓ New best model saved (val_loss: {val_loss:.6f})")
        
        # Save periodic
        if (epoch + 1) % 10 == 0:
            torch.save(checkpoint, os.path.join(config.training.output_dir, f'checkpoint_epoch_{epoch+1}.pth'))
    
    print("\n✓ Training complete!")


if __name__ == '__main__':
    main()
