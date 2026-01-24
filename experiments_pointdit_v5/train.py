"""Training script for Point-DiT."""

import torch
import torch.nn as nn
from torch.optim import AdamW
from torch.optim.lr_scheduler import CosineAnnealingLR
from tqdm import tqdm
import os
import numpy as np
from pathlib import Path

# Load wandb API key from .env if present
try:
    from dotenv import load_dotenv
    env_path = Path(__file__).parent.parent / '.env'
    if env_path.exists():
        load_dotenv(env_path)
except ImportError:
    pass

try:
    import wandb
    HAS_WANDB = True
except ImportError:
    HAS_WANDB = False

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
    total_chamfer = 0
    total_repulsion = 0
    
    pbar = tqdm(dataloader, desc=f"Epoch {epoch}")
    for step, batch in enumerate(pbar):
        image = batch['image'].to(device)
        points = batch['points'].to(device)
        
        # Forward pass (returns dict with loss components)
        loss_dict = train_step(model, scheduler, points, image, device)
        loss = loss_dict['loss']
        
        # Backward pass
        optimizer.zero_grad()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), config.training.max_grad_norm)
        optimizer.step()
        
        # Logging
        total_loss += loss.item()
        total_chamfer += loss_dict['chamfer']
        total_repulsion += loss_dict['repulsion']
        pbar.set_postfix({'loss': loss.item()})
        
        if step % config.training.log_every == 0:
            avg_loss = total_loss / (step + 1)
            print(f"Epoch {epoch}, Step {step}, Loss: {loss.item():.6f}, Avg: {avg_loss:.6f}")
    
    n_steps = len(dataloader)
    return {
        'loss': total_loss / n_steps,
        'chamfer': total_chamfer / n_steps,
        'repulsion': total_repulsion / n_steps,
    }


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
    total_chamfer = 0
    total_repulsion = 0
    
    for batch in tqdm(dataloader, desc="Validating"):
        image = batch['image'].to(device)
        points = batch['points'].to(device)
        
        loss_dict = train_step(model, scheduler, points, image, device)
        total_loss += loss_dict['loss'].item()
        total_chamfer += loss_dict['chamfer']
        total_repulsion += loss_dict['repulsion']
    
    n_steps = len(dataloader)
    return {
        'loss': total_loss / n_steps,
        'chamfer': total_chamfer / n_steps,
        'repulsion': total_repulsion / n_steps,
    }


def main():
    # Load config
    config = Config.default()
    
    # Initialize wandb
    if config.training.use_wandb and HAS_WANDB:
        wandb.init(
            project=config.training.wandb_project,
            name=config.training.wandb_run_name,
            config={
                'model_n_points': config.model.n_points,
                'model_dim': config.model.dim,
                'model_n_layers': config.model.n_layers,
                'model_n_heads': config.model.n_heads,
                'batch_size': config.training.batch_size,
                'learning_rate': config.training.learning_rate,
                'num_epochs': config.training.num_epochs,
                'data_num_points': config.data.num_points,
            }
        )
        print(f"Wandb initialized: project={config.training.wandb_project}")
    elif config.training.use_wandb and not HAS_WANDB:
        print("[Warning] wandb requested but not installed. Skipping wandb logging.")
    
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
        train_metrics = train_epoch(
            model, scheduler, train_loader, optimizer, device, epoch, config
        )
        
        # Validate
        val_metrics = validate(model, scheduler, val_loader, device)
        
        print(f"\nEpoch {epoch}:")
        print(f"  Train Loss = {train_metrics['loss']:.6f} (Chamfer: {train_metrics['chamfer']:.6f}, Repulsion: {train_metrics['repulsion']:.6f})")
        print(f"  Val Loss = {val_metrics['loss']:.6f} (Chamfer: {val_metrics['chamfer']:.6f}, Repulsion: {val_metrics['repulsion']:.6f})")
        
        # Log to wandb
        if config.training.use_wandb and HAS_WANDB:
            wandb.log({
                'epoch': epoch,
                'train_loss': train_metrics['loss'],
                'train_chamfer': train_metrics['chamfer'],
                'train_repulsion': train_metrics['repulsion'],
                'val_loss': val_metrics['loss'],
                'val_chamfer': val_metrics['chamfer'],
                'val_repulsion': val_metrics['repulsion'],
                'learning_rate': optimizer.param_groups[0]['lr'],
            })
        
        # Update LR
        if lr_scheduler is not None:
            lr_scheduler.step()
            print(f"Learning rate: {optimizer.param_groups[0]['lr']:.6e}")
        
        # Save checkpoint
        checkpoint = {
            'epoch': epoch,
            'model_state_dict': model.state_dict(),
            'optimizer_state_dict': optimizer.state_dict(),
            'train_loss': train_metrics['loss'],
            'val_loss': val_metrics['loss'],
            'best_val_loss': best_val_loss,
            'config': config,
        }
        
        # Save latest
        torch.save(checkpoint, os.path.join(config.training.output_dir, 'checkpoint_latest.pth'))
        
        # Save best
        if val_metrics['loss'] < best_val_loss:
            best_val_loss = val_metrics['loss']
            checkpoint['best_val_loss'] = best_val_loss
            torch.save(checkpoint, os.path.join(config.training.output_dir, 'checkpoint_best.pth'))
            print(f"✓ New best model saved (val_loss: {val_metrics['loss']:.6f})")
        
        # Save periodic
        if (epoch + 1) % 10 == 0:
            torch.save(checkpoint, os.path.join(config.training.output_dir, f'checkpoint_epoch_{epoch+1}.pth'))
    
    print("\n✓ Training complete!")
    
    # Finish wandb run
    if config.training.use_wandb and HAS_WANDB:
        wandb.finish()


if __name__ == '__main__':
    main()
