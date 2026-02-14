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

    pred_pix = np.stack(
        [(pred[:, 0] + 1.0) * (W / 2.0), (pred[:, 1] + 1.0) * (H / 2.0)], axis=1
    )

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


parser = argparse.ArgumentParser(description="Overfit Point-RT V9 on a single sample")
parser.add_argument("--steps", type=int, default=600, help="Training steps")
parser.add_argument("--sample-index", type=int, default=0, help="Dataset sample index")
parser.add_argument("--lr", type=float, default=2e-4, help="Learning rate")
parser.add_argument("--seed", type=int, default=42, help="Random seed")
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

# Fixed initialization for repeatability
with torch.no_grad():
    init_points = fast_density_initialization(image, model.n_points)

output_dir = os.path.join(os.path.dirname(__file__), "outputs_point_detr_v9_overfit")
os.makedirs(output_dir, exist_ok=True)

print("\n" + "=" * 60)
print("OVERFITTING TEST: Point-RT V9 (Iterative Refinement)")
print("=" * 60)
print(f"Image shape: {image.shape}")
print(f"Init points shape: {init_points.shape}")
print(f"Init points range: [{init_points.min():.3f}, {init_points.max():.3f}]")
print(f"Sample index: {args.sample_index} from dataset (total {len(dataset)})")

sample_dir = os.path.join(output_dir, f"sample_{args.sample_index}")
os.makedirs(sample_dir, exist_ok=True)

img_np = image[0, 0].detach().cpu().numpy()
np.save(os.path.join(sample_dir, "input_image.npy"), img_np)
np.save(os.path.join(sample_dir, "gt_image.npy"), gt_np)

optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=1e-5)
loss_fn = UnsupervisedStipplingLoss(
    sinkhorn_weight=config.training.sinkhorn_weight,
    repulsion_weight=config.training.repulsion_weight,
    render_weight=config.training.render_weight,
    repulsion_base_radius=config.training.base_radius,
    sinkhorn_blur=config.training.sinkhorn_blur,
    render_grid_size=config.training.render_grid_size,
    render_sigma=config.training.render_sigma,
).to(device)

print("\nTraining Point-RT V9 on single sample:")
print(f"{'Step':>5} {'Loss':>12} {'Sinkhorn':>12} {'Repulse':>12} {'Render':>12}")
print("-" * 60)

for step in range(args.steps):
    model.train()
    pred_points = model(init_points, image)
    loss, loss_dict = loss_fn(pred_points, image)

    optimizer.zero_grad()
    loss.backward()
    torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
    optimizer.step()

    if step % 25 == 0 or step == args.steps - 1:
        print(
            f"{step:5d} {loss.item():12.6f} "
            f"{loss_dict['sinkhorn_loss']:12.6f} "
            f"{loss_dict['repulsion_loss']:12.6f} "
            f"{loss_dict['render_loss']:12.6f}"
        )

print("\nNow sampling from the trained model:")
model.eval()

with torch.no_grad():
    sampled_points = model(init_points, image)

np.save(os.path.join(sample_dir, "pred_points_normalized.npy"), sampled_points.detach().cpu().numpy())

H, W = img_np.shape
pred = sampled_points[0].detach().cpu().numpy()
pred_pix = np.stack([(pred[:, 0] + 1.0) * (W / 2.0), (pred[:, 1] + 1.0) * (H / 2.0)], axis=1)
np.save(os.path.join(sample_dir, "pred_points_pixels.npy"), pred_pix)

vis_path = os.path.join(sample_dir, "comparison.png")
visualize_sample(image, gt_np, sampled_points, vis_path)
print(f"\n✓ Results saved to: {sample_dir}/")
