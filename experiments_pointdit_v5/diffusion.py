"""DDPM diffusion scheduler tailored for point stippling (x_start prediction)."""

import torch
import torch.nn as nn
import torch.nn.functional as F
from tqdm import tqdm


class ChamferLoss(nn.Module):
    """Chamfer distance loss for point clouds (set-aware matching)."""

    def __init__(self):
        super().__init__()

    def forward(self, preds: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
        """
        Args:
            preds: (B, N, 2) predicted points
            targets: (B, M, 2) target points
        Returns:
            Chamfer distance (scalar)
        """
        # Compute pairwise squared distances
        x = preds.unsqueeze(2)  # (B, N, 1, 2)
        y = targets.unsqueeze(1)  # (B, 1, M, 2)
        dist = torch.pow(x - y, 2).sum(-1)  # (B, N, M)

        # For each predicted point, find nearest target
        min_dist_pred_to_target, _ = torch.min(dist, dim=2)  # (B, N)

        # For each target point, find nearest prediction
        min_dist_target_to_pred, _ = torch.min(dist, dim=1)  # (B, M)

        # Average the distances
        loss = torch.mean(min_dist_pred_to_target) + torch.mean(min_dist_target_to_pred)

        return loss


class RepulsionLoss(nn.Module):
    """Repulsion loss to enforce blue-noise (even spacing) in point distributions.
    
    Penalizes points that are too close together, preventing clustering and
    encouraging uniform coverage (stippling quality).
    """

    def __init__(self, repulsion_radius: float = 0.02):
        """Initialize repulsion loss.
        
        Args:
            repulsion_radius: Minimum desired distance between points.
                             For 512x512 images in [-1,1] coords, 0.02 ≈ 5-10 pixels.
        """
        super().__init__()
        self.r = repulsion_radius

    def forward(self, points: torch.Tensor) -> torch.Tensor:
        """
        Args:
            points: (B, N, 2) predicted point coordinates
        Returns:
            Repulsion loss (scalar) - penalty for points closer than radius
        """
        # 1. Calculate pairwise distance matrix
        loc = points.unsqueeze(2)  # (B, N, 1, 2)
        ref = points.unsqueeze(1)  # (B, 1, N, 2)
        dist_sq = torch.sum((loc - ref) ** 2, dim=-1)  # (B, N, N)

        # 2. Get distances (add epsilon to avoid nan gradient at 0)
        dist = torch.sqrt(dist_sq + 1e-6)  # (B, N, N)

        # 3. Penalize distances < r (excluding self-distance which is 0)
        # Mask diagonal to ignore self-distance
        mask = torch.eye(points.shape[1], device=points.device).bool().unsqueeze(0)
        dist = dist.masked_fill(mask, float('inf'))  # Ignore self

        # Loss = sum of (r - dist) for all pairs where dist < r
        repulsion = torch.nn.functional.relu(self.r - dist)

        return torch.mean(repulsion)


class DDPMScheduler:
    """DDPM noise scheduler for diffusion models with x_start prediction."""

    def __init__(
        self,
        num_train_timesteps: int = 1000,
        beta_start: float = 1e-4,
        beta_end: float = 0.02,
        beta_schedule: str = "linear",
    ):
        self.num_train_timesteps = num_train_timesteps

        # Create beta schedule
        if beta_schedule == "linear":
            self.betas = torch.linspace(beta_start, beta_end, num_train_timesteps)
        elif beta_schedule == "cosine":
            steps = num_train_timesteps + 1
            s = 0.008
            t = torch.linspace(0, num_train_timesteps, steps) / num_train_timesteps
            alphas_cumprod = torch.cos((t + s) / (1 + s) * torch.pi / 2) ** 2
            alphas_cumprod = alphas_cumprod / alphas_cumprod[0]
            betas = 1 - (alphas_cumprod[1:] / alphas_cumprod[:-1])
            self.betas = torch.clip(betas, 0.0001, 0.9999)
        else:
            raise ValueError(f"Unknown beta schedule: {beta_schedule}")

        # Precompute useful values
        self.alphas = 1.0 - self.betas
        self.alphas_cumprod = torch.cumprod(self.alphas, dim=0)
        self.alphas_cumprod_prev = F.pad(self.alphas_cumprod[:-1], (1, 0), value=1.0)

        self.sqrt_alphas_cumprod = torch.sqrt(self.alphas_cumprod)
        self.sqrt_one_minus_alphas_cumprod = torch.sqrt(1.0 - self.alphas_cumprod)
        self.sqrt_recip_alphas = torch.sqrt(1.0 / self.alphas)

        # Posterior coefficients for x0 prediction sampling
        self.posterior_variance = (
            self.betas * (1.0 - self.alphas_cumprod_prev) / (1.0 - self.alphas_cumprod)
        )
        self.posterior_mean_coef1 = (
            self.betas * torch.sqrt(self.alphas_cumprod_prev) / (1.0 - self.alphas_cumprod)
        )
        self.posterior_mean_coef2 = (
            (1.0 - self.alphas_cumprod_prev) * torch.sqrt(self.alphas) / (1.0 - self.alphas_cumprod)
        )

    def add_noise(self, x_0: torch.Tensor, noise: torch.Tensor, timesteps: torch.Tensor) -> torch.Tensor:
        """Add noise to clean points x_0 to get x_t."""
        idx_device = timesteps.device
        sqrt_alpha_prod = self.sqrt_alphas_cumprod.to(idx_device)[timesteps]
        sqrt_one_minus_alpha_prod = self.sqrt_one_minus_alphas_cumprod.to(idx_device)[timesteps]

        sqrt_alpha_prod = sqrt_alpha_prod.view(-1, 1, 1).to(x_0.device)
        sqrt_one_minus_alpha_prod = sqrt_one_minus_alpha_prod.view(-1, 1, 1).to(x_0.device)

        x_t = sqrt_alpha_prod * x_0 + sqrt_one_minus_alpha_prod * noise
        return x_t

    def step_x0(self, model_output_x0: torch.Tensor, timestep: int, x_t: torch.Tensor, eta: float = 0.0) -> torch.Tensor:
        """Reverse diffusion step when the model predicts x_start."""
        t = timestep

        if t == 0:
            return model_output_x0

        coef1 = self.posterior_mean_coef1[t].to(x_t.device)
        coef2 = self.posterior_mean_coef2[t].to(x_t.device)
        model_mean = coef1 * model_output_x0 + coef2 * x_t

        var = self.posterior_variance[t].to(x_t.device)
        if eta > 0:
            noise = torch.randn_like(x_t)
            std = torch.sqrt(var) * eta
            return model_mean + std * noise
        else:
            noise = torch.randn_like(x_t)
            std = torch.sqrt(var)
            return model_mean + std * noise

    def set_timesteps(self, num_inference_steps: int):
        """Set timesteps for sampling."""
        self.num_inference_steps = num_inference_steps
        step_ratio = self.num_train_timesteps / num_inference_steps
        timesteps = torch.arange(0, num_inference_steps) * step_ratio
        self.timesteps = torch.flip(timesteps.long(), dims=[0])


# Instantiate losses globally
_chamfer_loss = ChamferLoss()
_repulsion_loss = RepulsionLoss(repulsion_radius=0.02)


def train_step(
    model: torch.nn.Module,
    scheduler: DDPMScheduler,
    x_0: torch.Tensor,
    image: torch.Tensor,
    device: str = "cuda",
    repulsion_weight: float = 0.5,
) -> dict:
    """Single training step predicting x_start with Chamfer + Repulsion loss.
    
    Args:
        model: Point-DiT model
        scheduler: DDPM scheduler
        x_0: (B, N, 2) clean target points
        image: (B, 1, H, W) conditioning image
        device: device string
        repulsion_weight: Weight for repulsion loss (default 0.5)
                         Higher = more even spacing, Lower = closer to GT shape
    
    Returns:
        Dict with 'loss' (combined), 'chamfer', and 'repulsion' components
    """
    batch_size = x_0.shape[0]

    noise = torch.randn_like(x_0)
    timesteps = torch.randint(0, scheduler.num_train_timesteps, (batch_size,), device=device, dtype=torch.long)

    x_t = scheduler.add_noise(x_0, noise, timesteps)
    pred_x0 = model(x_t, timesteps, image)

    # Chamfer: match shape coverage (primary objective)
    chamfer = _chamfer_loss(pred_x0, x_0)
    
    # Repulsion: enforce even spacing / blue noise (stippling quality)
    repulsion = _repulsion_loss(pred_x0)
    
    # Combined loss
    loss = chamfer + (repulsion_weight * repulsion)

    return {
        'loss': loss,
        'chamfer': chamfer.item(),
        'repulsion': repulsion.item(),
    }


@torch.no_grad()
def sample(
    model: torch.nn.Module,
    scheduler: DDPMScheduler,
    image: torch.Tensor,
    num_points: int,
    num_inference_steps: int = 50,
    device: str = "cuda",
    show_progress: bool = True,
    init_from_density: bool = True,
    init_std: float = 0.05,
    eta: float = 0.0,
) -> torch.Tensor:
    """Sample points from the diffusion model (x_start prediction)."""
    batch_size = image.shape[0]

    if init_from_density:
        x_t = density_guided_init(image, num_points, device, std=init_std)
    else:
        x_t = torch.randn(batch_size, num_points, 2, device=device) * init_std

    scheduler.set_timesteps(num_inference_steps)

    timesteps = scheduler.timesteps.to(device)
    iterator = tqdm(timesteps, desc="Sampling") if show_progress else timesteps

    for t in iterator:
        t_batch = t.unsqueeze(0).repeat(batch_size)
        pred_x0 = model(x_t, t_batch, image)
        pred_x0 = pred_x0.clamp(-1.0, 1.0)
        x_t = scheduler.step_x0(pred_x0, t.item(), x_t, eta=eta)

    x_t = torch.clamp(x_t, -1.0, 1.0)
    return x_t


@torch.no_grad()
def density_guided_init(
    image: torch.Tensor,
    num_points: int,
    device: str,
    std: float = 0.05,
) -> torch.Tensor:
    """Initialize points from image density distribution."""
    batch_size = image.shape[0]
    H, W = image.shape[2], image.shape[3]

    points_list = []

    for b in range(batch_size):
        img = image[b, 0].cpu().numpy()

        density = 1.0 - img
        density = density ** 2.0
        density = density / (density.sum() + 1e-6)

        flat_density = density.ravel()
        indices = torch.from_numpy(
            torch.multinomial(torch.from_numpy(flat_density).float(), num_points, replacement=True).numpy()
        )

        ys, xs = torch.div(indices, W, rounding_mode="floor"), indices % W

        xs_norm = (xs.float() / (W - 1)) * 2.0 - 1.0
        ys_norm = (ys.float() / (H - 1)) * 2.0 - 1.0

        pts = torch.stack([xs_norm, ys_norm], dim=-1)
        pts = pts + torch.randn_like(pts) * std
        pts = torch.clamp(pts, -1.0, 1.0)

        points_list.append(pts)

    return torch.stack(points_list, dim=0).to(device)
