"""Overfit Point-DiT V5 on a single (source, target) pair.

Trains on one example for many steps, periodically samples from the
diffusion model, and saves comparison visualizations + metrics + weights.

Usage (from experiments_pointdit_v5/):
    python test_overfit.py --steps 2000
    python test_overfit.py --steps 5000 --sample-index 42 --vis-every 200
    python test_overfit.py --steps 3000 --checkpoint /path/to/checkpoint_best.pth
"""

import argparse
import json
import os
import sys
import time

import numpy as np
import torch

try:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    HAS_MPL = True
except Exception:
    plt = None
    HAS_MPL = False

try:
    import wandb
    HAS_WANDB = True
except ImportError:
    wandb = None
    HAS_WANDB = False

from config import Config
from dataset import SimpleImageDataset
from diffusion import DDPMScheduler, train_step, sample
from model import PointDiT
from overfit_metrics import visualize_overfit_metrics, collect_metrics_dict


# ── global config (edit these) ───────────────────────────────────────
DATA_ROOT = "/groups/asharf_group/ofirgila/ControlNet/training/data_grads_v3_wave_1024"
SOURCE_DIR = os.path.join(DATA_ROOT, "source")
TARGET_DIR = os.path.join(DATA_ROOT, "target")
N_POINTS = 1024
IMAGE_SIZE = 512
WANDB_ENV = os.path.join(
    os.path.dirname(os.path.dirname(os.path.abspath(__file__))), ".env"
)


# ── helpers ──────────────────────────────────────────────────────────

def load_wandb_key():
    if os.path.exists(WANDB_ENV):
        with open(WANDB_ENV) as f:
            for line in f:
                line = line.strip()
                if line.startswith("WANDB_API_KEY"):
                    key = line.split("=", 1)[1].strip().strip('"').strip("'")
                    os.environ["WANDB_API_KEY"] = key
                    return True
    return False


def sample_from_model(model, scheduler, image, n_points, device,
                      n_samples=4, inference_steps=50):
    """Run the reverse diffusion loop and return (n_samples, N, 2) numpy."""
    model.eval()
    image_batch = image.repeat(n_samples, 1, 1, 1)

    with torch.no_grad():
        pts = sample(
            model, scheduler, image_batch, n_points,
            num_inference_steps=inference_steps,
            device=device,
            show_progress=False,
            init_from_density=True,
            init_std=0.05,
            eta=0.0,
        )
    return pts.cpu().numpy()


# ── main ─────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(
        description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    parser.add_argument("--steps", type=int, default=2000,
                        help="Number of training steps")
    parser.add_argument("--sample-index", type=int, default=0,
                        help="Dataset sample index to overfit on")
    parser.add_argument("--lr", type=float, default=1e-4,
                        help="Learning rate")
    parser.add_argument("--vis-every", type=int, default=500,
                        help="Visualise & sample every N steps")
    parser.add_argument("--sample-timesteps", type=int, default=50,
                        help="Diffusion inference steps when sampling")
    parser.add_argument("--n-samples", type=int, default=2,
                        help="Number of samples to generate at each vis step")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--no-wandb", action="store_true")

    # Optional pretrained checkpoint
    parser.add_argument("--checkpoint", type=str, default=None,
                        help="Path to pretrained checkpoint to fine-tune")

    # Data override
    parser.add_argument("--source-dir", type=str, default=None,
                        help="Override source image directory")
    parser.add_argument("--target-dir", type=str, default=None,
                        help="Override target image directory")

    # Loss weights (Phase-1 defaults: Chamfer + Repulsion)
    parser.add_argument("--chamfer-weight", type=float, default=1.0)
    parser.add_argument("--repulsion-weight", type=float, default=1.0)
    parser.add_argument("--sinkhorn-weight", type=float, default=0.0)
    parser.add_argument("--grid-density-weight", type=float, default=0.0)
    parser.add_argument("--adaptive-repulsion", action="store_true", default=False)

    args = parser.parse_args()

    # ── seeds ─────────────────────────────────────────────────────────
    torch.manual_seed(args.seed)
    np.random.seed(args.seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(args.seed)
    device = torch.device(args.device if torch.cuda.is_available() else "cpu")

    config = Config.default()

    # ── wandb ─────────────────────────────────────────────────────────
    use_wandb = HAS_WANDB and not args.no_wandb
    if use_wandb:
        load_wandb_key()
        wandb.init(
            project="pointdit-overfit",
            config={**vars(args), "n_points": N_POINTS, "data_root": DATA_ROOT},
            name=f"v5-overfit-idx{args.sample_index}-{args.steps}steps",
        )

    # ── dataset ───────────────────────────────────────────────────────
    source_dir = args.source_dir or SOURCE_DIR
    target_dir = args.target_dir or TARGET_DIR

    dataset = SimpleImageDataset(
        source_dir=source_dir,
        target_dir=target_dir,
        image_size=IMAGE_SIZE,
        num_points=N_POINTS,
    )

    if args.sample_index >= len(dataset):
        sys.exit(f"sample-index {args.sample_index} out of range "
                 f"(dataset has {len(dataset)} files)")

    sample_data = dataset[args.sample_index]
    image = sample_data["image"].unsqueeze(0).to(device)      # (1, 1, H, W)
    gt_points = sample_data["points"].unsqueeze(0).to(device)  # (1, N, 2)

    # Derive file stem for output directory
    source_files = sorted(os.listdir(source_dir))
    fname = source_files[args.sample_index]
    stem = os.path.splitext(fname)[0]

    out_dir = os.path.join(os.path.dirname(__file__), "overfit_outputs", stem)
    os.makedirs(out_dir, exist_ok=True)

    # Save source image for reference
    source_np = (image[0, 0].cpu().numpy() * 255).astype(np.uint8)
    from PIL import Image as PILImage
    PILImage.fromarray(source_np).save(os.path.join(out_dir, "source.png"))

    gt_points_np = gt_points[0].cpu().numpy()
    np.save(os.path.join(out_dir, "gt_points.npy"), gt_points_np)

    print(f"Example : {fname}")
    print(f"  data root : {DATA_ROOT}")
    print(f"  source dir: {source_dir}")
    print(f"  target dir: {target_dir}")
    print(f"  n_points  : {N_POINTS}")
    print(f"  image size: {IMAGE_SIZE}")
    print(f"  image shape: {image.shape}")
    print(f"  GT points : {gt_points.shape}  range [{gt_points.min():.3f}, {gt_points.max():.3f}]")
    print(f"  output dir: {out_dir}")

    if use_wandb:
        wandb.log({
            "source": wandb.Image(source_np, caption="Source (condition)"),
        }, step=0)

    # ── model ─────────────────────────────────────────────────────────
    model = PointDiT(
        n_points=N_POINTS,
        dim=config.model.dim,
        n_layers=config.model.n_layers,
        n_heads=config.model.n_heads,
        image_size=IMAGE_SIZE,
        dropout=0.0,
    ).to(device)

    if args.checkpoint:
        print(f"  Loading pretrained checkpoint: {args.checkpoint}")
        ckpt = torch.load(args.checkpoint, map_location=device, weights_only=False)
        model.load_state_dict(ckpt["model_state_dict"])
        print(f"  Loaded from epoch {ckpt.get('epoch', '?')}")

    trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"  PointDiT trainable params: {trainable:,}")

    scheduler = DDPMScheduler(
        num_train_timesteps=config.diffusion.num_train_timesteps,
        beta_start=config.diffusion.beta_start,
        beta_end=config.diffusion.beta_end,
        beta_schedule=config.diffusion.beta_schedule,
    )

    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=1e-5)

    # ── training loop ─────────────────────────────────────────────────
    loss_weights_str = (
        f"chamfer={args.chamfer_weight} repulsion={args.repulsion_weight} "
        f"sinkhorn={args.sinkhorn_weight} grid_density={args.grid_density_weight}"
    )
    print(f"\nLoss weights: {loss_weights_str}")
    print(f"\n{'Step':>6}  {'Loss':>12}  {'Chamfer':>10}  {'Repuls':>10}  {'Sinkhorn':>10}")
    print("-" * 65)

    losses_history = []
    t_start = time.time()

    for step in range(1, args.steps + 1):
        model.train()

        loss_dict = train_step(
            model, scheduler, gt_points, image, str(device),
            chamfer_weight=args.chamfer_weight,
            repulsion_weight=args.repulsion_weight,
            sinkhorn_weight=args.sinkhorn_weight,
            grid_density_weight=args.grid_density_weight,
            use_adaptive_repulsion=args.adaptive_repulsion,
        )
        loss = loss_dict["loss"]

        optimizer.zero_grad()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        optimizer.step()

        loss_val = loss.item()
        losses_history.append(loss_val)

        if use_wandb:
            log = {"loss": loss_val}
            for k in ("chamfer", "repulsion", "sinkhorn", "grid_density"):
                if loss_dict.get(k, 0) != 0:
                    log[k] = loss_dict[k]
            wandb.log(log, step=step)

        if step % 50 == 0 or step == 1:
            print(f"{step:6d}  {loss_val:12.6f}  "
                  f"{loss_dict.get('chamfer', 0):10.6f}  "
                  f"{loss_dict.get('repulsion', 0):10.6f}  "
                  f"{loss_dict.get('sinkhorn', 0):10.6f}")

        # ── periodic visualisation ────────────────────────────────────
        if step % args.vis_every == 0 or step == args.steps:
            print(f"  [step {step}] Sampling {args.n_samples} predictions ...")
            pts_np = sample_from_model(
                model, scheduler, image, N_POINTS,
                device, n_samples=args.n_samples,
                inference_steps=args.sample_timesteps,
            )

            pred_list = [pts_np[i] for i in range(pts_np.shape[0])]

            vis_path = os.path.join(out_dir, f"vis_step{step:05d}.png")
            saved = visualize_overfit_metrics(
                source_np, gt_points_np, pred_list, vis_path, step=step,
            )
            np.save(os.path.join(out_dir, f"points_step{step:05d}.npy"), pts_np)
            print(f"  -> saved visualisation: {vis_path}")

            # Compute and print scalar metrics for the first prediction
            image_01 = source_np.astype(np.float64) / 255.0
            step_metrics = collect_metrics_dict(gt_points_np, pred_list[0], image_01)
            print(f"     Chamfer={step_metrics['chamfer']:.5f}  "
                  f"Cap={step_metrics['pred_capacity_score']:.3f}  "
                  f"SpaceCV={step_metrics['pred_spacing_cv']:.3f}  "
                  f"SpaceScore={step_metrics['pred_spacing_score']:.3f}")

            if use_wandb:
                wlog = {"step_metrics/" + k: v for k, v in step_metrics.items()}
                if saved:
                    wlog["comparison"] = wandb.Image(saved, caption=f"Step {step}")
                wandb.log(wlog, step=step)

            model.train()

    elapsed = time.time() - t_start
    print(f"\nTraining finished in {elapsed:.1f}s ({elapsed / 60:.1f}min)")

    # ── save final weights ────────────────────────────────────────────
    ckpt_path = os.path.join(out_dir, "pointdit_overfit.pt")
    torch.save({
        "model_state_dict": model.state_dict(),
        "step": args.steps,
        "loss_history": losses_history,
        "example": fname,
        "config": config,
        "args": vars(args),
    }, ckpt_path)
    print(f"  -> saved weights: {ckpt_path}")

    # ── save loss curve ───────────────────────────────────────────────
    if HAS_MPL:
        fig, ax = plt.subplots(figsize=(8, 4))
        ax.plot(range(1, len(losses_history) + 1), losses_history, linewidth=0.5)
        ax.set_xlabel("Step")
        ax.set_ylabel("Loss")
        ax.set_title(f"PointDiT V5 Overfit Loss ({fname})")
        ax.grid(True, alpha=0.3)
        loss_path = os.path.join(out_dir, "loss_curve.png")
        plt.tight_layout()
        plt.savefig(loss_path, dpi=150)
        plt.close()
        print(f"  -> saved loss curve: {loss_path}")
        if use_wandb:
            wandb.log({"loss_curve": wandb.Image(loss_path)})

    # ── save final metrics ────────────────────────────────────────────
    image_01 = source_np.astype(np.float64) / 255.0
    final_pts = sample_from_model(
        model, scheduler, image, N_POINTS,
        device, n_samples=1, inference_steps=args.sample_timesteps,
    )
    final_metrics = collect_metrics_dict(gt_points_np, final_pts[0], image_01)

    metrics = {
        "example": fname,
        "data_root": DATA_ROOT,
        "n_points": N_POINTS,
        "image_size": IMAGE_SIZE,
        "steps": args.steps,
        "lr": args.lr,
        "seed": args.seed,
        "final_loss": float(losses_history[-1]),
        "min_loss": float(min(losses_history)),
        "mean_loss_last100": float(np.mean(losses_history[-100:])),
        "elapsed_seconds": round(elapsed, 1),
        "loss_weights": {
            "chamfer": args.chamfer_weight,
            "repulsion": args.repulsion_weight,
            "sinkhorn": args.sinkhorn_weight,
            "grid_density": args.grid_density_weight,
        },
        **{f"final_{k}": v for k, v in final_metrics.items()},
    }
    with open(os.path.join(out_dir, "metrics.json"), "w") as f:
        json.dump(metrics, f, indent=2)
    print(f"  -> saved metrics.json")

    if use_wandb:
        wandb.log(metrics)
        wandb.finish()

    print(f"\nFinal loss : {losses_history[-1]:.6f}  (min: {min(losses_history):.6f})")
    print(f"Chamfer    : {final_metrics['chamfer']:.5f}")
    print(f"Capacity   : {final_metrics['pred_capacity_score']:.3f}")
    print(f"Spacing CV : {final_metrics['pred_spacing_cv']:.3f}")
    print(f"Spacing Scr: {final_metrics['pred_spacing_score']:.3f}")
    print(f"Results saved to: {out_dir}/")


if __name__ == "__main__":
    main()
