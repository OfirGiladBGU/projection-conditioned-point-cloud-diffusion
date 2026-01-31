"""Training script for Point-DiT with Hybrid Loss Strategy.

Hybrid Training Strategy (for best quality + reasonable speed):
- Phase 1 (epochs 0-79): Chamfer + Repulsion loss (~46 min/epoch)
  - Teaches position accuracy and basic spacing
- Phase 2 (epochs 80-99): Sinkhorn + Chamfer loss (~103 min/epoch)
  - Refines blue noise distribution quality

Total: ~2.5 days for Phase 1 + ~1.5 days for Phase 2 = ~4 days
vs 7 days for pure Sinkhorn or 3 days for pure Chamfer+Repulsion

Checkpoint includes training phase info for robust crash recovery.
"""

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


# =============================================================================
# Hybrid Training Configuration
# =============================================================================
PHASE1_EPOCHS = 80   # Chamfer + Repulsion (fast, position learning)
PHASE2_EPOCHS = 20   # Sinkhorn + Chamfer (slower, blue noise refinement)
TOTAL_EPOCHS = PHASE1_EPOCHS + PHASE2_EPOCHS  # 100


def get_training_phase(epoch: int) -> dict:
    """Determine training phase and loss configuration for given epoch.
    
    Phase 1 (0-79): Chamfer + Repulsion - fast position learning
    Phase 2 (80-99): Sinkhorn + Chamfer - blue noise refinement
    """
    if epoch < PHASE1_EPOCHS:
        return {
            'phase': 1,
            'name': 'chamfer_repulsion',
            'chamfer_weight': 1.0,
            'repulsion_weight': 1.0,
            'sinkhorn_weight': 0.0,
        }
    else:
        return {
            'phase': 2,
            'name': 'sinkhorn_chamfer',
            'chamfer_weight': 10.0,
            'repulsion_weight': 0.0,
            'sinkhorn_weight': 1.0,
        }


def train_epoch(
    model: nn.Module,
    scheduler: DDPMScheduler,
    dataloader,
    optimizer,
    device: str,
    epoch: int,
    config: Config,
):
    """Train for one epoch with phase-appropriate loss."""
    model.train()
    
    # Get loss configuration for this epoch
    phase_config = get_training_phase(epoch)
    
    total_loss = 0
    total_chamfer = 0
    total_secondary = 0  # repulsion or sinkhorn depending on phase
    
    phase_name = phase_config['name']
    pbar = tqdm(dataloader, desc=f"Epoch {epoch} [{phase_name}]")
    
    for step, batch in enumerate(pbar):
        image = batch['image'].to(device)
        points = batch['points'].to(device)
        
        # Forward pass with phase-appropriate loss
        loss_dict = train_step(
            model, scheduler, points, image, device,
            sinkhorn_weight=phase_config['sinkhorn_weight'],
            chamfer_weight=phase_config['chamfer_weight'],
            repulsion_weight=phase_config['repulsion_weight'],
        )
        loss = loss_dict['loss']
        
        # Backward pass
        optimizer.zero_grad()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), config.training.max_grad_norm)
        optimizer.step()
        
        # Logging - determine secondary loss based on weights
        total_loss += loss.item()
        total_chamfer += loss_dict['chamfer']
        
        if phase_config['sinkhorn_weight'] > 0:
            total_secondary += loss_dict.get('sinkhorn', 0.0)
            secondary_name = 'sinkhorn'
            secondary_val = loss_dict.get('sinkhorn', 0.0)
        else:
            total_secondary += loss_dict.get('repulsion', 0.0)
            secondary_name = 'repulsion'
            secondary_val = loss_dict.get('repulsion', 0.0)
        
        pbar.set_postfix({
            'loss': f"{loss.item():.4f}",
            'chamfer': f"{loss_dict['chamfer']:.6f}",
            secondary_name: f"{secondary_val:.4f}",
        })
        
        if step % config.training.log_every == 0:
            print(f"Epoch {epoch} [{phase_name}], Step {step}, Loss: {loss.item():.4f}, "
                  f"Chamfer: {loss_dict['chamfer']:.6f}, {secondary_name}: {secondary_val:.4f}")
    
    n_steps = len(dataloader)
    return {
        'loss': total_loss / n_steps,
        'chamfer': total_chamfer / n_steps,
        'secondary': total_secondary / n_steps,
        'phase': phase_config['phase'],
        'phase_name': phase_name,
    }


@torch.no_grad()
def validate(
    model: nn.Module,
    scheduler: DDPMScheduler,
    dataloader,
    device: str,
    epoch: int,
):
    """Validate the model with phase-appropriate loss."""
    model.eval()
    
    phase_config = get_training_phase(epoch)
    
    total_loss = 0
    total_chamfer = 0
    total_secondary = 0
    
    for batch in tqdm(dataloader, desc="Validating"):
        image = batch['image'].to(device)
        points = batch['points'].to(device)
        
        loss_dict = train_step(
            model, scheduler, points, image, device,
            sinkhorn_weight=phase_config['sinkhorn_weight'],
            chamfer_weight=phase_config['chamfer_weight'],
            repulsion_weight=phase_config['repulsion_weight'],
        )
        total_loss += loss_dict['loss'].item()
        total_chamfer += loss_dict['chamfer']
        
        if phase_config['sinkhorn_weight'] > 0:
            total_secondary += loss_dict.get('sinkhorn', 0.0)
        else:
            total_secondary += loss_dict.get('repulsion', 0.0)
    
    n_steps = len(dataloader)
    return {
        'loss': total_loss / n_steps,
        'chamfer': total_chamfer / n_steps,
        'secondary': total_secondary / n_steps,
        'phase': phase_config['phase'],
    }


def main():
    # Load config
    config = Config.default()
    
    # Override num_epochs to use hybrid total
    config.training.num_epochs = TOTAL_EPOCHS
    
    # Initialize wandb
    if config.training.use_wandb and HAS_WANDB:
        wandb.init(
            project=config.training.wandb_project,
            name=config.training.wandb_run_name or "v5_hybrid",
            config={
                'model_n_points': config.model.n_points,
                'model_dim': config.model.dim,
                'model_n_layers': config.model.n_layers,
                'model_n_heads': config.model.n_heads,
                'batch_size': config.training.batch_size,
                'learning_rate': config.training.learning_rate,
                'num_epochs': config.training.num_epochs,
                'data_num_points': config.data.num_points,
                'loss_strategy': 'hybrid',
                'phase1_epochs': PHASE1_EPOCHS,
                'phase2_epochs': PHASE2_EPOCHS,
                'phase1_loss': 'chamfer_repulsion',
                'phase2_loss': 'sinkhorn_chamfer',
            },
            resume='allow',  # Allow resuming wandb run
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
    
    # LR Scheduler - adjusted for total epochs
    if config.training.lr_scheduler == 'cosine':
        lr_scheduler = CosineAnnealingLR(optimizer, T_max=config.training.num_epochs)
    elif config.training.lr_scheduler == 'warmup_cosine':
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
    
    # ==========================================================================
    # CHECKPOINT LOADING - Robust crash recovery
    # ==========================================================================
    start_epoch = 0
    best_val_loss = float('inf')
    
    # First check for auto-resume from latest checkpoint
    latest_ckpt_path = os.path.join(config.training.output_dir, 'checkpoint_latest.pth')
    
    if os.path.exists(latest_ckpt_path):
        print(f"\n{'='*60}")
        print(f"RESUMING FROM CHECKPOINT: {latest_ckpt_path}")
        print(f"{'='*60}")
        
        checkpoint = torch.load(latest_ckpt_path, map_location=device, weights_only=False)
        model.load_state_dict(checkpoint['model_state_dict'])
        optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        
        start_epoch = checkpoint['epoch'] + 1
        best_val_loss = checkpoint.get('best_val_loss', float('inf'))
        
        # Restore LR scheduler state if available
        if lr_scheduler is not None and 'lr_scheduler_state_dict' in checkpoint:
            lr_scheduler.load_state_dict(checkpoint['lr_scheduler_state_dict'])
        elif lr_scheduler is not None:
            # Step scheduler to correct position
            for _ in range(start_epoch):
                lr_scheduler.step()
        
        saved_phase = checkpoint.get('training_phase', 1)
        current_phase = get_training_phase(start_epoch)['phase']
        
        print(f"  Resuming from epoch {start_epoch}")
        print(f"  Best val loss so far: {best_val_loss:.6f}")
        print(f"  Saved at phase {saved_phase}, continuing in phase {current_phase}")
        
        if current_phase != saved_phase:
            print(f"  ⚠ PHASE TRANSITION: Switching from Phase {saved_phase} to Phase {current_phase}")
        print(f"{'='*60}\n")
        
    elif config.training.checkpoint_path and os.path.exists(config.training.checkpoint_path):
        # Fallback to explicitly specified checkpoint
        print(f"Loading specified checkpoint: {config.training.checkpoint_path}")
        checkpoint = torch.load(config.training.checkpoint_path, map_location=device, weights_only=False)
        model.load_state_dict(checkpoint['model_state_dict'])
        optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        start_epoch = checkpoint['epoch'] + 1
        best_val_loss = checkpoint.get('best_val_loss', float('inf'))
        
        if lr_scheduler is not None:
            for _ in range(start_epoch):
                lr_scheduler.step()
        
        print(f"Resuming from epoch {start_epoch}")
    
    # ==========================================================================
    # TRAINING LOOP with Hybrid Strategy
    # ==========================================================================
    print(f"\n{'='*60}")
    print(f"HYBRID TRAINING STRATEGY")
    print(f"{'='*60}")
    print(f"Phase 1 (epochs 0-{PHASE1_EPOCHS-1}): Chamfer + Repulsion (~46 min/epoch)")
    print(f"Phase 2 (epochs {PHASE1_EPOCHS}-{TOTAL_EPOCHS-1}): Sinkhorn + Chamfer (~103 min/epoch)")
    print(f"Total epochs: {TOTAL_EPOCHS}")
    print(f"Starting from epoch: {start_epoch}")
    print(f"{'='*60}\n")
    
    for epoch in range(start_epoch, config.training.num_epochs):
        phase_config = get_training_phase(epoch)
        
        # Log phase transition
        if epoch == PHASE1_EPOCHS:
            print(f"\n{'='*60}")
            print(f"🔄 PHASE TRANSITION: Phase 1 → Phase 2")
            print(f"   Switching from Chamfer+Repulsion to Sinkhorn+Chamfer")
            print(f"   Expect slower epochs but better blue noise quality")
            print(f"{'='*60}\n")
        
        # Train
        train_metrics = train_epoch(
            model, scheduler, train_loader, optimizer, device, epoch, config
        )
        
        # Validate
        val_metrics = validate(model, scheduler, val_loader, device, epoch)
        
        # Get secondary loss name based on phase (weight-based)
        if phase_config['sinkhorn_weight'] > 0:
            secondary_name = 'sinkhorn'
        else:
            secondary_name = 'repulsion'
        
        print(f"\nEpoch {epoch} [Phase {phase_config['phase']}: {phase_config['name']}]:")
        print(f"  Train Loss = {train_metrics['loss']:.4f} (Chamfer: {train_metrics['chamfer']:.6f}, {secondary_name}: {train_metrics['secondary']:.4f})")
        print(f"  Val Loss = {val_metrics['loss']:.4f} (Chamfer: {val_metrics['chamfer']:.6f}, {secondary_name}: {val_metrics['secondary']:.4f})")
        
        # Log to wandb
        if config.training.use_wandb and HAS_WANDB:
            log_dict = {
                'epoch': epoch,
                'training_phase': phase_config['phase'],
                'train_loss': train_metrics['loss'],
                'train_chamfer': train_metrics['chamfer'],
                'val_loss': val_metrics['loss'],
                'val_chamfer': val_metrics['chamfer'],
                'learning_rate': optimizer.param_groups[0]['lr'],
            }
            # Log phase-specific metrics (weight-based)
            if phase_config['sinkhorn_weight'] > 0:
                log_dict['train_sinkhorn'] = train_metrics['secondary']
                log_dict['val_sinkhorn'] = val_metrics['secondary']
            else:
                log_dict['train_repulsion'] = train_metrics['secondary']
                log_dict['val_repulsion'] = val_metrics['secondary']
            
            wandb.log(log_dict)
        
        # Update LR
        if lr_scheduler is not None:
            lr_scheduler.step()
            print(f"Learning rate: {optimizer.param_groups[0]['lr']:.6e}")
        
        # =======================================================================
        # CHECKPOINT SAVING - Include all state for robust recovery
        # =======================================================================
        checkpoint = {
            'epoch': epoch,
            'model_state_dict': model.state_dict(),
            'optimizer_state_dict': optimizer.state_dict(),
            'lr_scheduler_state_dict': lr_scheduler.state_dict() if lr_scheduler else None,
            'train_loss': train_metrics['loss'],
            'val_loss': val_metrics['loss'],
            'best_val_loss': best_val_loss,
            'training_phase': phase_config['phase'],
            'phase_config': phase_config,
            'config': config,
            'hybrid_settings': {
                'phase1_epochs': PHASE1_EPOCHS,
                'phase2_epochs': PHASE2_EPOCHS,
                'total_epochs': TOTAL_EPOCHS,
            },
        }
        
        # Save latest (always - for crash recovery)
        torch.save(checkpoint, os.path.join(config.training.output_dir, 'checkpoint_latest.pth'))
        
        # Save best (within same phase for fair comparison)
        if val_metrics['loss'] < best_val_loss:
            best_val_loss = val_metrics['loss']
            checkpoint['best_val_loss'] = best_val_loss
            torch.save(checkpoint, os.path.join(config.training.output_dir, 'checkpoint_best.pth'))
            print(f"✓ New best model saved (val_loss: {val_metrics['loss']:.6f})")
        
        # Save at phase transitions
        if epoch == PHASE1_EPOCHS - 1:
            torch.save(checkpoint, os.path.join(config.training.output_dir, 'checkpoint_phase1_complete.pth'))
            print(f"✓ Phase 1 complete checkpoint saved")
        
        # Save periodic
        if (epoch + 1) % 10 == 0:
            torch.save(checkpoint, os.path.join(config.training.output_dir, f'checkpoint_epoch_{epoch+1}.pth'))
    
    # Save final model
    final_checkpoint = {
        'epoch': config.training.num_epochs - 1,
        'model_state_dict': model.state_dict(),
        'best_val_loss': best_val_loss,
        'training_complete': True,
        'hybrid_settings': {
            'phase1_epochs': PHASE1_EPOCHS,
            'phase2_epochs': PHASE2_EPOCHS,
            'total_epochs': TOTAL_EPOCHS,
        },
    }
    torch.save(final_checkpoint, os.path.join(config.training.output_dir, 'checkpoint_final.pth'))
    
    print(f"\n{'='*60}")
    print("✓ TRAINING COMPLETE!")
    print(f"  Total epochs: {TOTAL_EPOCHS}")
    print(f"  Best val loss: {best_val_loss:.6f}")
    print(f"  Checkpoints saved to: {config.training.output_dir}")
    print(f"{'='*60}")
    
    # Finish wandb run
    if config.training.use_wandb and HAS_WANDB:
        wandb.finish()


if __name__ == '__main__':
    main()
