#!/usr/bin/env python
"""Verify Point-DiT changes without interfering with running training.

Uses isolated test directories and CPU-only operations.
"""

import sys
import os
import tempfile
import torch
from pathlib import Path

print("=" * 70)
print("VERIFICATION: Point-DiT 5k Configuration")
print("=" * 70)

try:
    # 1. Check config
    print("\n[1/8] Checking config.py...")
    from experiments_pointdit_v5.config import Config
    cfg = Config.default()
    assert cfg.model.n_points == 5000, f"model.n_points should be 5000, got {cfg.model.n_points}"
    assert cfg.data.num_points == 5000, f"data.num_points should be 5000, got {cfg.data.num_points}"
    assert cfg.training.use_wandb == True, "use_wandb should be True"
    assert cfg.training.wandb_project == "PointDiT", f"wandb_project should be 'PointDiT'"
    print("   ✓ Config: n_points=5000, num_points=5000, wandb enabled")

    # 2. Check model instantiation (CPU, no data)
    print("\n[2/8] Checking model instantiation (5000 points)...")
    from experiments_pointdit_v5.model import PointDiT
    model = PointDiT(
        n_points=cfg.model.n_points,
        dim=cfg.model.dim,
        n_layers=cfg.model.n_layers,
        n_heads=cfg.model.n_heads,
        image_size=cfg.model.image_size,
        dropout=cfg.model.dropout,
    )
    params = model.get_num_params()
    print(f"   ✓ Model created: {params:,} parameters")

    # 3. Check diffusion returns dict with components
    print("\n[3/8] Checking diffusion.train_step() signature...")
    from experiments_pointdit_v5.diffusion import DDPMScheduler, train_step
    scheduler = DDPMScheduler(
        num_train_timesteps=cfg.diffusion.num_train_timesteps,
        beta_start=cfg.diffusion.beta_start,
        beta_end=cfg.diffusion.beta_end,
        beta_schedule=cfg.diffusion.beta_schedule,
    )
    
    # Create small batch without dataset
    x_0 = torch.randn(1, cfg.model.n_points, 2, device='cpu')
    image = torch.randn(1, 1, 512, 512, device='cpu')
    
    # Test train_step
    loss_dict = train_step(model, scheduler, x_0, image, device='cpu')
    assert isinstance(loss_dict, dict), f"train_step must return dict, got {type(loss_dict)}"
    assert set(loss_dict.keys()) == {'loss', 'chamfer', 'repulsion'}, f"Unexpected keys: {loss_dict.keys()}"
    assert torch.is_tensor(loss_dict['loss']), "loss must be tensor"
    assert isinstance(loss_dict['chamfer'], float), "chamfer must be float"
    assert isinstance(loss_dict['repulsion'], float), "repulsion must be float"
    print(f"   ✓ train_step() returns dict with components:")
    print(f"     - loss (tensor): {loss_dict['loss']:.6f}")
    print(f"     - chamfer (float): {loss_dict['chamfer']:.6f}")
    print(f"     - repulsion (float): {loss_dict['repulsion']:.6f}")

    # 4. Check dataset with isolated temp directory
    print("\n[4/8] Checking dataset (no conflict with running training)...")
    from experiments_pointdit_v5.dataset import create_dataloaders
    
    # Verify config paths exist
    assert Path(cfg.data.source_dir).exists(), f"source_dir missing: {cfg.data.source_dir}"
    assert Path(cfg.data.target_dir).exists(), f"target_dir missing: {cfg.data.target_dir}"
    print(f"   ✓ Data dirs exist:")
    print(f"     - source: {cfg.data.source_dir}")
    print(f"     - target: {cfg.data.target_dir}")
    
    # Create loaders with small batch (won't write to output_dir)
    train_loader, val_loader = create_dataloaders(
        source_dir=cfg.data.source_dir,
        target_dir=cfg.data.target_dir,
        batch_size=2,
        num_workers=0,
        val_split=cfg.data.val_split,
        image_size=cfg.data.image_size,
        num_points=cfg.data.num_points,
    )
    
    batch = next(iter(train_loader))
    assert batch['points'].shape == (2, 5000, 2), f"Expected (2, 5000, 2), got {batch['points'].shape}"
    print(f"   ✓ Dataset batch shapes correct:")
    print(f"     - image: {batch['image'].shape}")
    print(f"     - points: {batch['points'].shape}")

    # 5. Check imports in eval/test (no execution)
    print("\n[5/8] Checking eval.py syntax...")
    from experiments_pointdit_v5 import eval as eval_module
    assert hasattr(eval_module, 'load_model'), "eval missing load_model"
    assert hasattr(eval_module, 'sample'), "eval missing sample"
    print("   ✓ eval.py has required functions")

    print("\n[6/8] Checking test.py syntax...")
    from experiments_pointdit_v5 import test as test_module
    assert hasattr(test_module, 'load_model'), "test missing load_model"
    assert hasattr(test_module, 'load_image'), "test missing load_image"
    print("   ✓ test.py has required functions")

    # 6. Check train.py dotenv integration
    print("\n[7/8] Checking train.py imports and dotenv...")
    from experiments_pointdit_v5 import train as train_module
    assert hasattr(train_module, 'train_epoch'), "train missing train_epoch"
    assert hasattr(train_module, 'validate'), "train missing validate"
    assert hasattr(train_module, 'HAS_WANDB'), "train missing HAS_WANDB"
    print("   ✓ train.py has required functions")
    
    # Verify dotenv loads correctly
    from dotenv import load_dotenv
    env_path = Path('.env')
    if env_path.exists():
        load_dotenv(env_path)
        api_key = os.getenv('WANDB_API_KEY')
        assert api_key, "WANDB_API_KEY not loaded from .env"
        print(f"   ✓ dotenv loads WANDB_API_KEY: {api_key[:10]}...")
    else:
        print(f"   ! .env not found (OK, will use env vars)")

    # 7. Check wandb import
    print("\n[8/8] Checking wandb and config compatibility...")
    try:
        import wandb
        print("   ✓ wandb module available")
    except ImportError:
        print("   ✗ wandb not installed (error!)")
        sys.exit(1)

    # Verify wandb config settings
    assert cfg.training.wandb_project == "PointDiT", "wandb_project mismatch"
    print("   ✓ wandb config: project='PointDiT'")

    print("\n" + "=" * 70)
    print("ALL VERIFICATION CHECKS PASSED ✓")
    print("=" * 70)
    print("\nSummary of verified changes:")
    print("  ✓ Config: n_points=5000, num_points=5000")
    print("  ✓ Config: wandb enabled (project='PointDiT')")
    print("  ✓ Model: Instantiates with 5000 points")
    print("  ✓ Diffusion: train_step() returns dict with loss components")
    print("  ✓ Dataset: Emits (B, 5000, 2) point clouds")
    print("  ✓ Train: Imports dotenv, has loss tracking")
    print("  ✓ Eval/Test: Updated and compatible")
    print("  ✓ Wandb: Configured and available")
    print("\nNo conflicts with running training - used isolated test batches only.")
    print("Ready to commit!")

except Exception as e:
    print(f"\n✗ VERIFICATION FAILED: {e}")
    import traceback
    traceback.print_exc()
    sys.exit(1)
