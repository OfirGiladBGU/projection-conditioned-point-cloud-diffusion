"""
Short Training Script for Point-RT v7 - Quick Prototype Testing

This script runs a quick training on a small subset to verify the architecture
and see results before committing to a full training run.

Much faster than v5 since it's single-pass (no diffusion loop).

Usage:
    python short_train.py
    
    # On GPU node:
    bash /home/ofirgila/scripts/pycharm_rtx_6000_32h.sh
    cd experiments_point_detr_v7 && python short_train.py
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
from model.point_rt import PointRT
from fast_init import fast_density_initialization, create_point_input_vectors
from losses import PointRTLoss


def custom_collate_fn(batch):
    """Custom collate that drops optional GT point fields."""
    return {
        'image': torch.stack([item['image'] for item in batch]),
        'image_path': [item['image_path'] for item in batch],
    }


def load_target_binary(image_path: str, target_dir: str, image_size: int, device: str) -> torch.Tensor:
    """Load target binary image and return (1, 1, H, W) float tensor in [0, 1]."""
    target_path = Path(target_dir) / Path(image_path).name
    if not target_path.exists():
        raise FileNotFoundError(f"Target image not found: {target_path}")

    img = Image.open(target_path).convert('L').resize(
        (image_size, image_size), Image.BILINEAR
    )
    img_np = np.asarray(img, dtype=np.float32) / 255.0
    return torch.from_numpy(img_np).unsqueeze(0).unsqueeze(0).to(device)


def extract_coords_from_binary(
    binary_image: torch.Tensor,
    max_points: int,
    dots_are_white: bool = True,
    threshold: float = 0.5,
) -> torch.Tensor:
    """Convert binary image (B, 1, H, W) to coords (B, N, 2) in [-1, 1]."""
    B, _, H, W = binary_image.shape
    device = binary_image.device
    coords_list = []

    for b in range(B):
        if dots_are_white:
            y_idx, x_idx = torch.where(binary_image[b, 0] > threshold)
        else:
            y_idx, x_idx = torch.where(binary_image[b, 0] < threshold)

        if y_idx.numel() == 0:
            coords = torch.zeros(max_points, 2, device=device)
        else:
            y_norm = (y_idx.float() / (H - 1)) * 2 - 1
            x_norm = (x_idx.float() / (W - 1)) * 2 - 1
            coords = torch.stack([x_norm, y_norm], dim=-1)

            if coords.shape[0] > max_points:
                perm = torch.randperm(coords.shape[0], device=device)[:max_points]
                coords = coords[perm]
            elif coords.shape[0] < max_points:
                repeat_factor = (max_points // coords.shape[0]) + 1
                coords = coords.repeat(repeat_factor, 1)[:max_points]

        coords_list.append(coords)

    return torch.stack(coords_list, dim=0)


def build_gt_points_for_batch(
    image_paths: list,
    target_dir: str,
    image_size: int,
    device: str,
    max_points: int,
    dots_are_white: bool,
) -> torch.Tensor:
    """Load GT target images and convert them to point coordinates."""
    target_imgs = [
        load_target_binary(p, target_dir, image_size, device)
        for p in image_paths
    ]
    target_tensor = torch.cat(target_imgs, dim=0)
    return extract_coords_from_binary(
        target_tensor,
        max_points=max_points,
        dots_are_white=dots_are_white,
    )


def compute_metrics(pred_points: torch.Tensor, gt_points: torch.Tensor):
    """Compute evaluation metrics."""
    B, N, _ = pred_points.shape
    
    # Nearest neighbor distances for blue noise quality
    dists = torch.cdist(pred_points, pred_points)
    mask = torch.eye(N, device=pred_points.device).bool().unsqueeze(0).expand(B, -1, -1)
    dists = dists.masked_fill(mask, float('inf'))
    nn_dists, _ = dists.min(dim=2)
    
    mean_nn = nn_dists.mean().item()
    cv = nn_dists.std().item() / (mean_nn + 1e-8)
    
    # Chamfer distance
    x = pred_points.unsqueeze(2)
    y = gt_points.unsqueeze(1)
    dist_sq = torch.pow(x - y, 2).sum(-1)
    
    min_dist_pred_to_gt, _ = torch.min(dist_sq, dim=2)
    min_dist_gt_to_pred, _ = torch.min(dist_sq, dim=1)
    
    chamfer = (torch.mean(min_dist_pred_to_gt) + torch.mean(min_dist_gt_to_pred)).item() / 2
    
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
    
    # Ground truth
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
    
    # Overlay
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
    """Quick training loop for Point-RT."""
    
    # Configuration - small batch for quick testing
    BATCH_SIZE = 4
    NUM_EPOCHS = 20
    NUM_TRAIN_SAMPLES = 200  # Small subset
    EVAL_EVERY = 2
    
    # Loss weights (default)
    CHAMFER_WEIGHT = 1.0
    REPULSION_WEIGHT = 0.1
    DIVERSITY_WEIGHT = 0.05
    DOTS_ARE_WHITE = False
    
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    print(f"Device: {device}")
    print(f"Loss weights - Chamfer: {CHAMFER_WEIGHT}, Repulsion: {REPULSION_WEIGHT}, Diversity: {DIVERSITY_WEIGHT}")
    print(f"GT dots are white: {DOTS_ARE_WHITE}")
    
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
        target_dir=config.data.target_dir,
        lloyd_dir=None,
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
    
    # Precompute GT points for the training subset (fast training)
    with torch.no_grad():
        for batch in train_data:
            batch['gt_points'] = build_gt_points_for_batch(
                batch['image_path'],
                config.data.target_dir,
                config.data.image_size,
                device,
                max_points=config.data.num_points,
                dots_are_white=DOTS_ARE_WHITE,
            )

    # Get validation sample for consistent evaluation
    val_batch = next(iter(val_loader))
    val_image = val_batch['image'].to(device)
    val_image_paths = val_batch['image_path']

    with torch.no_grad():
        val_gt_points = build_gt_points_for_batch(
            val_image_paths,
            config.data.target_dir,
            config.data.image_size,
            device,
            max_points=config.data.num_points,
            dots_are_white=DOTS_ARE_WHITE,
        )
    
    # Model and optimizer
    model = PointRT(
        n_points=config.data.num_points,
        dim=config.model.dim,
        n_layers=config.model.n_layers,
        n_heads=config.model.n_heads,
        image_size=config.data.image_size,
        use_resnet_backbone=config.model.use_resnet_backbone,
        resnet_pretrained=False,  # False for quick test
        dropout=config.model.dropout,
    ).to(device)
    
    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=config.training.learning_rate,
        weight_decay=config.training.weight_decay,
    )
    
    loss_fn = PointRTLoss(
        mse_weight=CHAMFER_WEIGHT,
        repulsion_weight=REPULSION_WEIGHT,
        diversity_weight=DIVERSITY_WEIGHT,
    ).to(device)
    
    print(f"Model parameters: {model.get_num_params():,}")
    
    # Training loop
    history = {
        'train_loss': [],
        'train_chamfer': [],
        'train_repulsion': [],
        'val_mean_nn': [],
        'val_cv': [],
        'val_chamfer': [],
    }
    
    gt_metrics = compute_metrics(val_gt_points, val_gt_points)
    print(f"\nGT Metrics: Mean NN={gt_metrics['mean_nn']:.4f}, CV={gt_metrics['cv']:.3f}")
    
    start_time = time.time()
    
    for epoch in range(1, NUM_EPOCHS + 1):
        model.train()
        epoch_losses = {'loss': [], 'chamfer': [], 'repulsion': []}
        
        pbar = tqdm(train_data, desc=f"Epoch {epoch}/{NUM_EPOCHS}")
        for batch in pbar:
            image = batch['image'].to(device)
            
            # Fast initialization
            with torch.no_grad():
                init_points = fast_density_initialization(image, model.n_points)
                input_vecs = create_point_input_vectors(init_points, image)
                gt_points = batch['gt_points']
            
            optimizer.zero_grad()
            
            # Forward pass (SINGLE PASS - the key difference!)
            pred_points = model(input_vecs, image)
            
            # Loss computation
            loss, loss_dict = loss_fn(pred_points, gt_points)
            
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), config.training.max_grad_norm)
            optimizer.step()
            
            epoch_losses['loss'].append(loss.item())
            epoch_losses['chamfer'].append(loss_dict['chamfer_loss'])
            epoch_losses['repulsion'].append(loss_dict['repulsion_loss'])
            
            pbar.set_postfix({
                'loss': f"{loss.item():.4f}",
                'chamfer': f"{loss_dict['chamfer_loss']:.6f}",
                'rep': f"{loss_dict['repulsion_loss']:.6f}",
            })
        
        # Record training metrics
        avg_loss = np.mean(epoch_losses['loss'])
        avg_chamfer = np.mean(epoch_losses['chamfer'])
        avg_repulsion = np.mean(epoch_losses['repulsion'])
        
        history['train_loss'].append(avg_loss)
        history['train_chamfer'].append(avg_chamfer)
        history['train_repulsion'].append(avg_repulsion)
        
        print(f"Epoch {epoch}: Loss={avg_loss:.4f}, Chamfer={avg_chamfer:.6f}, Repulsion={avg_repulsion:.6f}")
        
        # Evaluation
        if epoch % EVAL_EVERY == 0 or epoch == 1:
            model.eval()
            with torch.no_grad():
                # Initialize and refine (single pass)
                init_points = fast_density_initialization(val_image, model.n_points)
                input_vecs = create_point_input_vectors(init_points, val_image)
                pred_points = model(input_vecs, val_image)  # SINGLE FORWARD PASS!
                
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
    
    elapsed = time.time() - start_time
    print(f"\nTraining completed in {elapsed/60:.1f} minutes")
    
    # Plot training curves
    fig, axes = plt.subplots(2, 2, figsize=(12, 10))
    
    # Training loss
    axes[0, 0].plot(history['train_loss'], label='Total Loss')
    axes[0, 0].set_xlabel('Epoch')
    axes[0, 0].set_ylabel('Loss')
    axes[0, 0].set_title('Training Loss')
    axes[0, 0].legend()
    axes[0, 0].grid(True, alpha=0.3)
    
    # Individual losses
    axes[0, 1].plot(history['train_chamfer'], label='Chamfer', linewidth=2)
    axes[0, 1].plot([x*10 for x in history['train_repulsion']], label='Repulsion (×10)', linewidth=2)
    axes[0, 1].set_xlabel('Epoch')
    axes[0, 1].set_ylabel('Loss')
    axes[0, 1].set_title('Loss Components')
    axes[0, 1].legend()
    axes[0, 1].grid(True, alpha=0.3)
    
    # Blue noise quality
    eval_epochs = list(range(EVAL_EVERY, NUM_EPOCHS + 1, EVAL_EVERY))
    if 1 not in eval_epochs:
        eval_epochs = [1] + eval_epochs
    
    axes[1, 0].plot(eval_epochs[:len(history['val_mean_nn'])], history['val_mean_nn'], 'b-o', label='Pred Mean NN', linewidth=2)
    axes[1, 0].axhline(y=gt_metrics['mean_nn'], color='r', linestyle='--', label=f"GT Mean NN: {gt_metrics['mean_nn']:.4f}")
    axes[1, 0].set_xlabel('Epoch')
    axes[1, 0].set_ylabel('Mean NN Distance')
    axes[1, 0].set_title('Blue Noise Quality (higher = better spaced)')
    axes[1, 0].legend()
    axes[1, 0].grid(True, alpha=0.3)
    
    # Chamfer distance
    axes[1, 1].plot(eval_epochs[:len(history['val_chamfer'])], history['val_chamfer'], 'g-o', linewidth=2)
    axes[1, 1].set_xlabel('Epoch')
    axes[1, 1].set_ylabel('Chamfer Distance')
    axes[1, 1].set_title('Position Accuracy (lower = better)')
    axes[1, 1].grid(True, alpha=0.3)
    
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
    print(f"Training time: {elapsed/60:.1f} minutes")
    print(f"GT Metrics:")
    print(f"  Mean NN: {gt_metrics['mean_nn']:.4f}")
    print(f"  CV:      {gt_metrics['cv']:.3f}")
    
    if history['val_mean_nn']:
        final_nn = history['val_mean_nn'][-1]
        final_chamfer = history['val_chamfer'][-1]
        final_cv = history['val_cv'][-1]
        print(f"\nFinal Predicted Metrics (Epoch {NUM_EPOCHS}):")
        print(f"  Mean NN: {final_nn:.4f} (ratio={final_nn/gt_metrics['mean_nn']:.3f})")
        print(f"  CV:      {final_cv:.3f}")
        print(f"  Chamfer: {final_chamfer:.6f}")
    
    print(f"\nKey observations:")
    print(f"  - Single forward pass per image (no diffusion loop)")
    print(f"  - Chamfer loss is main component")
    print(f"  - Repulsion loss enforces spacing")
    print(f"  - Expected CV target: 0.60-0.65")
    print(f"  - Expected Chamfer target: <0.001")
    
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
            'repulsion_weight': REPULSION_WEIGHT,
            'diversity_weight': DIVERSITY_WEIGHT,
        },
        'gt_metrics': gt_metrics,
    }, weights_path)
    print(f"  - Model weights: {weights_path}")
    
    print("\n✓ Short training complete!")


if __name__ == "__main__":
    main()
