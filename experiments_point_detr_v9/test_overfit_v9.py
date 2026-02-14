"""Overfitting test for Point-RT V9 on a single image."""

import argparse
import os

import numpy as np
import torch
from PIL import Image

try:
    import matplotlib.pyplot as plt

    MATPLOTLIB_AVAILABLE = True
except Exception:
    plt = None
    MATPLOTLIB_AVAILABLE = False

from config import Config
from dataset import PointRTDataset
from fast_init import fast_density_initialization
from losses import UnsupervisedStipplingLoss
from model import StippleRefiner


def visualize_sample(
    img_tensor: torch.Tensor,
    gt_img: np.ndarray,
    pred_pts: torch.Tensor,
    save_path: str,
):
    """Save side-by-side visualization: input | GT | predicted points."""
    img = img_tensor[0, 0].detach().cpu().numpy()
    H, W = img.shape
    pred = pred_pts[0].detach().cpu().numpy()

    pred_pix = np.stack([pred[:, 0] * (W - 1), pred[:, 1] * (H - 1)], axis=1)

    if MATPLOTLIB_AVAILABLE:
        fig, axes = plt.subplots(1, 3, figsize=(18, 6))

        axes[0].imshow(img, cmap="gray")
        axes[0].set_title("Input Image")
        axes[0].axis("off")

        axes[1].imshow(gt_img, cmap="gray", vmin=0, vmax=255)
        axes[1].set_title("GT Stippling")
        axes[1].axis("off")

        axes[2].scatter(pred_pix[:, 0], pred_pix[:, 1], c="red", s=1, alpha=0.7)
        axes[2].set_xlim(0, W)
        axes[2].set_ylim(H, 0)
        axes[2].set_aspect("equal")
        axes[2].set_facecolor("white")
        axes[2].set_title("Predicted Stippling")
        axes[2].axis("off")

        plt.tight_layout()
        plt.savefig(save_path, dpi=150, bbox_inches="tight")
        plt.close()
        print(f"Saved visualization to {save_path}")
    else:
        np.save(save_path.replace(".png", "_image.npy"), img)
        np.save(save_path.replace(".png", "_gt.npy"), gt_img)
        np.save(save_path.replace(".png", "_pred.npy"), pred_pix)
        print("Matplotlib not available, saved arrays instead")


def extract_gt_points_from_image(gt_image: np.ndarray, threshold: int = 127) -> np.ndarray:
    """Extract GT points from a binary stippling image as normalized coords in [0, 1]."""
    ys, xs = np.where(gt_image > threshold)
    if xs.size == 0:
        return np.zeros((0, 2), dtype=np.float32)
    H, W = gt_image.shape
    x_norm = xs.astype(np.float32) / (W - 1)
    y_norm = ys.astype(np.float32) / (H - 1)
    return np.stack([x_norm, y_norm], axis=1)


def chamfer_distance(p: torch.Tensor, q: torch.Tensor) -> torch.Tensor:
    """Chamfer distance between two point sets."""
    if p.numel() == 0 or q.numel() == 0:
        return torch.tensor(float("inf"), device=p.device)
    dists = torch.cdist(p, q, p=2)
    min_pq = dists.min(dim=2).values
    min_qp = dists.min(dim=1).values
    loss = (min_pq.pow(2).mean(dim=1) + min_qp.pow(2).mean(dim=1)).mean()
    return loss


def compute_nn_metrics(points: torch.Tensor) -> dict:
    """Compute nearest-neighbor statistics for a point set."""
    if points.numel() == 0:
        return {"mean_nn": float("inf"), "std_nn": float("inf"), "cv": float("inf")}
    B, N, _ = points.shape
    dists = torch.cdist(points, points)
    mask = torch.eye(N, device=points.device, dtype=torch.bool).unsqueeze(0).expand(B, -1, -1)
    dists = dists.masked_fill(mask, float("inf"))
    nn_dists, _ = dists.min(dim=2)
    mean_nn = nn_dists.mean().item()
    std_nn = nn_dists.std().item()
    cv = std_nn / (mean_nn + 1e-8)
    return {"mean_nn": mean_nn, "std_nn": std_nn, "cv": cv}


parser = argparse.ArgumentParser(description="Overfit Point-RT V9 on a single sample")
parser.add_argument("--steps", type=int, default=600, help="Training steps")
parser.add_argument("--sample-index", type=int, default=0, help="Dataset sample index")
parser.add_argument("--lr", type=float, default=2e-4, help="Learning rate")
parser.add_argument("--seed", type=int, default=42, help="Random seed")
parser.add_argument("--sinkhorn-weight", type=float, default=None, help="Override sinkhorn weight")
parser.add_argument("--repulsion-weight", type=float, default=None, help="Override repulsion weight")
parser.add_argument("--render-weight", type=float, default=None, help="Override render weight")
args = parser.parse_args()

np.random.seed(args.seed)
torch.manual_seed(args.seed)
torch.cuda.manual_seed_all(args.seed)
torch.backends.cudnn.deterministic = True
torch.backends.cudnn.benchmark = False

config = Config.default()
device = "cuda" if torch.cuda.is_available() else "cpu"

model = StippleRefiner(
    n_points=config.model.n_points,
    dim=config.model.dim,
    n_layers=config.model.n_layers,
    n_heads=config.model.n_heads,
    image_size=config.model.image_size,
    use_resnet_backbone=config.model.use_resnet_backbone,
    resnet_pretrained=config.model.resnet_pretrained,
    fourier_freqs=config.model.fourier_freqs,
    fourier_temperature=config.model.fourier_temperature,
    dropout=config.model.dropout,
    n_refine_steps=config.model.n_refine_steps,
    share_weights=config.model.share_weights,
    delta_scale=config.model.delta_scale,
).to(device)

print("Point-RT V9 model created")
print(f"Model parameters: {model.get_num_params():,}")

if not os.path.isdir(config.data.source_dir):
    raise FileNotFoundError(f"Source dir not found: {config.data.source_dir}")

dataset = PointRTDataset(
    source_dir=config.data.source_dir,
    image_size=config.data.image_size,
    num_points=config.data.num_points,
    target_dir=config.data.target_dir,
    lloyd_dir=config.data.lloyd_dir,
)

if args.sample_index < 0 or args.sample_index >= len(dataset):
    raise IndexError(f"sample_index {args.sample_index} out of range for dataset size {len(dataset)}")

sample_data = dataset[args.sample_index]
image = sample_data["image"].unsqueeze(0).to(device)
source_path = sample_data["image_path"]

if not config.data.target_dir:
    raise ValueError("target_dir is required for GT visualization")

target_path = os.path.join(config.data.target_dir, os.path.basename(source_path))
if not os.path.exists(target_path):
    raise FileNotFoundError(f"Target image not found: {target_path}")

gt_img = Image.open(target_path).convert("L").resize(
    (config.data.image_size, config.data.image_size), Image.BILINEAR
)
gt_np = np.asarray(gt_img, dtype=np.uint8)
gt_np_norm = gt_np.astype(np.float32)
gt_np_norm = gt_np_norm - gt_np_norm.min()
gt_np_norm = gt_np_norm / (gt_np_norm.max() + 1e-6)
gt_points_np = extract_gt_points_from_image(gt_np)
gt_points_tensor = torch.from_numpy(gt_points_np).unsqueeze(0).to(device)

# Fixed initialization for repeatability
with torch.no_grad():
    init_points = torch.rand(1, model.n_points, 2, device=device)
    gt_tensor = torch.from_numpy(gt_np_norm).unsqueeze(0).unsqueeze(0).to(device)
    B, _, H, W = gt_tensor.shape
    density = (1.0 - gt_tensor).clamp(min=0.0, max=1.0).pow(config.training.sinkhorn_gamma)
    flat = density.view(B, -1)
    probs = flat / (flat.sum(dim=1, keepdim=True) + 1e-6)
    indices = torch.multinomial(probs, model.n_points, replacement=True)
    y_idx = torch.div(indices, W, rounding_mode="floor").float()
    x_idx = (indices % W).float()
    debug_target_points = torch.stack([x_idx / (W - 1), y_idx / (H - 1)], dim=-1)

output_dir = os.path.join(os.path.dirname(__file__), "outputs_point_detr_v9_overfit")
os.makedirs(output_dir, exist_ok=True)

print("\n" + "=" * 60)
print("OVERFITTING TEST: Point-RT V9 (Iterative Refinement)")
print("=" * 60)
print(f"Image shape: {image.shape}")
print(f"Init points shape: {init_points.shape}")
print(f"Init points range: [{init_points.min():.3f}, {init_points.max():.3f}]")
print(f"Sample index: {args.sample_index} from dataset (total {len(dataset)})")
print(f"GT image range (uint8): [{gt_np.min()}, {gt_np.max()}]")
print(f"GT image range (norm): [{gt_np_norm.min():.3f}, {gt_np_norm.max():.3f}]")

sample_dir = os.path.join(output_dir, f"sample_{args.sample_index}")
os.makedirs(sample_dir, exist_ok=True)

img_np = image[0, 0].detach().cpu().numpy()
np.save(os.path.join(sample_dir, "input_image.npy"), img_np)
np.save(os.path.join(sample_dir, "gt_image.npy"), gt_np)
np.save(os.path.join(sample_dir, "gt_image_norm.npy"), gt_np_norm)
np.save(os.path.join(sample_dir, "gt_points_normalized.npy"), gt_points_np)

optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=1e-5)
sinkhorn_weight = config.training.sinkhorn_weight if args.sinkhorn_weight is None else args.sinkhorn_weight
repulsion_weight = config.training.repulsion_weight if args.repulsion_weight is None else args.repulsion_weight
render_weight = config.training.render_weight if args.render_weight is None else args.render_weight

loss_fn = UnsupervisedStipplingLoss(
    sinkhorn_weight=sinkhorn_weight,
    repulsion_weight=repulsion_weight,
    render_weight=render_weight,
    repulsion_base_radius=config.training.base_radius,
    repulsion_decay=config.training.repulsion_decay,
    repulsion_epsilon=config.training.repulsion_epsilon,
    repulsion_max_loss=config.training.repulsion_max_loss,
    sinkhorn_blur=config.training.sinkhorn_blur,
    render_grid_size=config.training.render_grid_size,
    render_sigma=config.training.render_sigma,
    sinkhorn_mode=config.training.sinkhorn_mode,
    sinkhorn_gamma=config.training.sinkhorn_gamma,
).to(device)

print("\nTraining Point-RT V9 on single sample:")
print(
    f"Using weights - Sinkhorn: {sinkhorn_weight}, "
    f"Repulsion: {repulsion_weight}, Render: {render_weight}"
)
print(f"{'Step':>5} {'Loss':>12} {'Sinkhorn':>12} {'Repulse':>12} {'Render':>12} {'|Delta|':>10}")
print("-" * 60)

if MATPLOTLIB_AVAILABLE:
    debug_path = os.path.join(sample_dir, "debug_target_points.png")
    dbg = debug_target_points[0].detach().cpu().numpy()
    plt.figure(figsize=(5, 5))
    plt.scatter(dbg[:, 0], dbg[:, 1], s=1, c="blue", alpha=0.7)
    plt.title("Debug Target Points (from GT density)")
    plt.xlim(0, 1)
    plt.ylim(0, 1)
    plt.gca().set_aspect("equal", "box")
    plt.savefig(debug_path, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"Saved debug target points to {debug_path}")

for step in range(args.steps):
    model.train()
    pred_points = model(init_points, image)
    loss, loss_dict = loss_fn(pred_points, image)

    optimizer.zero_grad()
    loss.backward()
    torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
    optimizer.step()

    if step % 25 == 0 or step == args.steps - 1:
        delta_mean = (pred_points - init_points).abs().mean().item()
        pred_min = pred_points.min().item()
        pred_max = pred_points.max().item()
        tgt_min = debug_target_points.min().item()
        tgt_max = debug_target_points.max().item()
        print(
            f"{step:5d} {loss.item():12.6f} "
            f"{loss_dict['sinkhorn_loss']:12.6f} "
            f"{loss_dict['repulsion_loss']:12.6f} "
            f"{loss_dict['render_loss']:12.6f} "
            f"{delta_mean:10.6f}"
        )
        # print(
        #     f"DEBUG ALIGNMENT: Pred[{pred_min:.2f}-{pred_max:.2f}] "
        #     f"Target[{tgt_min:.2f}-{tgt_max:.2f}]"
        # )

print("\nNow sampling from the trained model:")
model.eval()

with torch.no_grad():
    sampled_points = model(init_points, image)

    # Evaluate final losses and metrics
    final_loss, final_loss_dict = loss_fn(sampled_points, image)
    chamfer = chamfer_distance(sampled_points, gt_points_tensor).item()
    nn_metrics = compute_nn_metrics(sampled_points)

np.save(os.path.join(sample_dir, "pred_points_normalized.npy"), sampled_points.detach().cpu().numpy())

H, W = img_np.shape
pred = sampled_points[0].detach().cpu().numpy()
pred_pix = np.stack([pred[:, 0] * (W - 1), pred[:, 1] * (H - 1)], axis=1)
np.save(os.path.join(sample_dir, "pred_points_pixels.npy"), pred_pix)

vis_path = os.path.join(sample_dir, "comparison.png")
visualize_sample(image, gt_np, sampled_points, vis_path)

print("\nFinal Metrics:")
print(f"  Total Loss: {final_loss.item():.6f}")
print(f"  Sinkhorn:   {final_loss_dict['sinkhorn_loss']:.6f}")
print(f"  Repulsion:  {final_loss_dict['repulsion_loss']:.6f}")
print(f"  Render:     {final_loss_dict['render_loss']:.6f}")
print(f"  Chamfer:    {chamfer:.6f}")
print(f"  NN Mean:    {nn_metrics['mean_nn']:.6f}")
print(f"  NN Std:     {nn_metrics['std_nn']:.6f}")
print(f"  NN CV:      {nn_metrics['cv']:.6f}")

metrics_path = os.path.join(sample_dir, "metrics.json")
with open(metrics_path, "w", encoding="utf-8") as f:
    f.write(
        "{\n"
        f"  \"total_loss\": {final_loss.item():.6f},\n"
        f"  \"sinkhorn_loss\": {final_loss_dict['sinkhorn_loss']:.6f},\n"
        f"  \"repulsion_loss\": {final_loss_dict['repulsion_loss']:.6f},\n"
        f"  \"render_loss\": {final_loss_dict['render_loss']:.6f},\n"
        f"  \"chamfer\": {chamfer:.6f},\n"
        f"  \"mean_nn\": {nn_metrics['mean_nn']:.6f},\n"
        f"  \"std_nn\": {nn_metrics['std_nn']:.6f},\n"
        f"  \"cv_nn\": {nn_metrics['cv']:.6f}\n"
        "}\n"
    )
print(f"Saved metrics to {metrics_path}")
print(f"\n✓ Results saved to: {sample_dir}/")
