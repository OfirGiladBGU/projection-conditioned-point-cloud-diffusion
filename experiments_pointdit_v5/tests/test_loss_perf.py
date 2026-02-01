"""Test loss performance for v5 and v5_5 experiments."""
import torch
import time
import sys

def test_experiment(exp_path, exp_name):
    print(f'\n=== Testing {exp_name} ===')
    
    # Clear previous imports
    modules_to_remove = [m for m in sys.modules if 'diffusion' in m or 'model' in m or 'config' in m]
    for m in modules_to_remove:
        del sys.modules[m]
    
    sys.path.insert(0, exp_path)
    
    from diffusion import train_step, DDPMScheduler
    from model import PointDiT
    from config import Config
    
    config = Config.default()
    device = 'cuda'
    
    model = PointDiT(
        n_points=config.model.n_points,
        dim=config.model.dim,
        n_layers=config.model.n_layers,
        n_heads=config.model.n_heads,
        image_size=config.model.image_size,
        dropout=config.model.dropout,
    ).to(device)
    
    scheduler = DDPMScheduler()
    
    B, N = 8, 2048
    image = torch.rand(B, 1, 256, 256, device=device)
    points = torch.rand(B, N, 2, device=device) * 2 - 1
    
    # Warmup
    for _ in range(3):
        r = train_step(model, scheduler, points, image, device,
                       chamfer_weight=1.0, repulsion_weight=1.0)
        r['loss'].backward()
    torch.cuda.synchronize()
    
    # Test 1: Phase 1 (Chamfer+Repulsion, grid_density=0)
    times = []
    for _ in range(10):
        start = time.time()
        r = train_step(model, scheduler, points, image, device,
                       chamfer_weight=1.0, repulsion_weight=1.0)
        r['loss'].backward()
        torch.cuda.synchronize()
        times.append(time.time() - start)
    print(f'Phase1 (Chamfer+Repulsion): {sum(times)/len(times)*1000:.1f} ms')
    print(f'  grid_density={r["grid_density"]:.4f} (should be 0)')
    print(f'  repulsion={r["repulsion"]:.4f} (should be >0)')
    
    # Test 2: Chamfer only (all others disabled)
    times = []
    for _ in range(10):
        start = time.time()
        r = train_step(model, scheduler, points, image, device,
                       chamfer_weight=1.0)
        r['loss'].backward()
        torch.cuda.synchronize()
        times.append(time.time() - start)
    print(f'Chamfer Only: {sum(times)/len(times)*1000:.1f} ms')
    print(f'  grid_density={r["grid_density"]:.4f} (should be 0)')
    print(f'  repulsion={r["repulsion"]:.4f} (should be 0)')
    
    # Test 3: Phase 2 (Sinkhorn+Chamfer)
    times = []
    for _ in range(10):
        start = time.time()
        r = train_step(model, scheduler, points, image, device,
                       chamfer_weight=10.0, sinkhorn_weight=1.0)
        r['loss'].backward()
        torch.cuda.synchronize()
        times.append(time.time() - start)
    print(f'Phase2 (Sinkhorn+Chamfer): {sum(times)/len(times)*1000:.1f} ms')
    print(f'  sinkhorn={r["sinkhorn"]:.4f} (should be >0)')
    print(f'  chamfer={r["chamfer"]:.4f} (should be >0)')
    
    sys.path.remove(exp_path)
    return True


if __name__ == '__main__':
    print('Testing loss configurations and performance...')
    print('GPU:', torch.cuda.get_device_name(0))
    
    # Test v5
    test_experiment(
        '/groups/asharf_group/ofirgila/projection-conditioned-point-cloud-diffusion/experiments_pointdit_v5',
        'v5'
    )
    
    # Test v5_5
    test_experiment(
        '/groups/asharf_group/ofirgila/projection-conditioned-point-cloud-diffusion/experiments_pointdit_v5_5',
        'v5_5'
    )
    
    print('\n=== Summary ===')
    print('If grid_density and repulsion show 0.0 when their weights are 0,')
    print('then losses are correctly disabled and no unnecessary computation is done.')
