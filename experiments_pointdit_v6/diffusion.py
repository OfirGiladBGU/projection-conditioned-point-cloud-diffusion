"""DDPM diffusion scheduler tailored for point stippling (x_start prediction)."""

import torch
import torch.nn as nn
import torch.nn.functional as F
from tqdm import tqdm

# Import Sinkhorn loss - the key addition for blue noise quality
try:
    from sinkhorn_lloyd_losses import SinkhornDensityLoss, HAS_GEOMLOSS
except ImportError:
    HAS_GEOMLOSS = False
    SinkhornDensityLoss = None


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


class GridDensityLoss(nn.Module):
    """Multi-scale Grid Density Loss for matching point distribution to image intensity.
    
    Forces the model to produce points whose local density matches the input image.
    This is the key missing piece that Chamfer loss cannot enforce.
    
    Uses differentiable bilinear splatting to create a soft histogram of points,
    then compares against the downsampled input image at multiple scales.
    """
    
    def __init__(self, grid_sizes: tuple = (32, 64)):
        """Initialize multi-scale grid density loss.
        
        Args:
            grid_sizes: Tuple of grid resolutions to check density at.
                       Using multiple scales prevents boundary gaming.
        """
        super().__init__()
        self.grid_sizes = grid_sizes
    
    def _compute_density_at_scale(self, pred_points: torch.Tensor, 
                                   input_images: torch.Tensor, 
                                   grid_size: int) -> torch.Tensor:
        """Compute density loss at a single scale using correlation-based loss.
        
        Args:
            pred_points: (B, N, 2) in [-1, 1]
            input_images: (B, 1, H, W) normalized [0, 1]
            grid_size: Resolution for density comparison
        
        Returns:
            Density matching loss (negative correlation + MSE for stability)
        """
        B, N, _ = pred_points.shape
        device = pred_points.device
        
        # 1. Downsample input image to grid_size - this is our target density
        target_density = F.interpolate(
            input_images, size=(grid_size, grid_size), mode='bilinear', align_corners=False
        )
        target_density = target_density.squeeze(1)  # (B, H, W)
        
        # IMPORTANT: Invert the image! In stippling:
        # - Dark pixels (intensity ~0) should have MORE points
        # - Light pixels (intensity ~1) should have FEWER points
        # So target density = 1 - image_intensity
        target_density = 1.0 - target_density
        
        # 2. Rasterize predicted points into a grid using differentiable bilinear splatting
        # Map [-1, 1] -> [0, grid_size-1]
        pts_grid = (pred_points + 1) / 2 * (grid_size - 1)
        
        x = pts_grid[..., 0]  # (B, N)
        y = pts_grid[..., 1]  # (B, N)
        
        # Bilinear interpolation corners
        x0 = torch.floor(x).long().clamp(0, grid_size - 1)
        y0 = torch.floor(y).long().clamp(0, grid_size - 1)
        x1 = (x0 + 1).clamp(0, grid_size - 1)
        y1 = (y0 + 1).clamp(0, grid_size - 1)
        
        # Bilinear weights
        wa = (x1.float() - x) * (y1.float() - y)
        wb = (x1.float() - x) * (y - y0.float())
        wc = (x - x0.float()) * (y1.float() - y)
        wd = (x - x0.float()) * (y - y0.float())
        
        # Accumulate weights into grid using scatter_add
        batch_indices = torch.arange(B, device=device).unsqueeze(1).expand(-1, N)
        
        def scatter_to_grid(idx_x, idx_y, weights):
            """Scatter weights to flattened grid and reshape."""
            flat_idx = batch_indices * (grid_size ** 2) + idx_y * grid_size + idx_x
            result = torch.zeros(B * grid_size ** 2, device=device, dtype=pred_points.dtype)
            result = result.scatter_add_(0, flat_idx.flatten(), weights.flatten())
            return result.view(B, grid_size, grid_size)
        
        pred_density = (
            scatter_to_grid(x0, y0, wa) + 
            scatter_to_grid(x0, y1, wb) + 
            scatter_to_grid(x1, y0, wc) + 
            scatter_to_grid(x1, y1, wd)
        )
        
        # 3. Normalize both to have zero mean and unit variance for correlation
        target_flat = target_density.view(B, -1)  # (B, H*W)
        pred_flat = pred_density.view(B, -1)
        
        target_mean = target_flat.mean(dim=1, keepdim=True)
        pred_mean = pred_flat.mean(dim=1, keepdim=True)
        
        target_centered = target_flat - target_mean
        pred_centered = pred_flat - pred_mean
        
        target_std = target_centered.std(dim=1, keepdim=True) + 1e-6
        pred_std = pred_centered.std(dim=1, keepdim=True) + 1e-6
        
        # Correlation coefficient (maximize, so negate for loss)
        correlation = (target_centered * pred_centered).sum(dim=1) / (
            target_std.squeeze() * pred_std.squeeze() * target_flat.shape[1]
        )
        
        # Loss = 1 - correlation (so perfect match = 0)
        return (1.0 - correlation.mean())
    
    def forward(self, pred_points: torch.Tensor, input_images: torch.Tensor) -> torch.Tensor:
        """Compute multi-scale grid density loss.
        
        Args:
            pred_points: (B, N, 2) predicted points in [-1, 1]
            input_images: (B, 1, H, W) conditioning image normalized [0, 1]
        
        Returns:
            Combined loss across all scales
        """
        total_loss = 0.0
        for grid_size in self.grid_sizes:
            total_loss = total_loss + self._compute_density_at_scale(
                pred_points, input_images, grid_size
            )
        return total_loss / len(self.grid_sizes)


class SpectralBluenoiseLoss(nn.Module):
    """Spectral loss enforcing blue-noise frequency characteristics.
    
    Uses Fourier analysis to ensure predicted point distributions match
    the characteristic "donut" shape in frequency domain that defines blue-noise.
    This forces global structure matching, not just local spacing.
    """
    
    def __init__(self, grid_size: int = 256, num_radial_bins: int = 32):
        """Initialize spectral loss.
        
        Args:
            grid_size: Resolution for rasterization (higher = more detail)
            num_radial_bins: Number of bins for radial power spectrum
        """
        super().__init__()
        self.grid_size = grid_size
        self.num_radial_bins = num_radial_bins
    
    def rasterize_points(self, points: torch.Tensor, grid_size: int) -> torch.Tensor:
        """Rasterize point cloud to a soft histogram using Gaussian splatting.
        
        Args:
            points: (B, N, 2) point coordinates in [-1, 1]
            grid_size: Grid resolution
        
        Returns:
            (B, grid_size, grid_size) soft histogram
        """
        B, N, _ = points.shape
        device = points.device
        
        # Initialize grid
        grid = torch.zeros(B, grid_size, grid_size, device=device)
        
        # Convert points from [-1, 1] to [0, grid_size-1]
        pts_norm = (points + 1) / 2 * (grid_size - 1)
        
        # Use soft splatting with Gaussian kernel for differentiability
        sigma = 1.0  # Gaussian width in pixels
        for b in range(B):
            for i in range(N):
                x, y = pts_norm[b, i].detach()  # Detach for indexing
                
                # Compute contributions in a local neighborhood
                x_int = int(x.item())
                y_int = int(y.item())
                x_min = max(0, x_int - 3)
                x_max = min(grid_size, x_int + 4)
                y_min = max(0, y_int - 3)
                y_max = min(grid_size, y_int + 4)
                
                if x_max > x_min and y_max > y_min:
                    # Create coordinate grids for the neighborhood
                    xx, yy = torch.meshgrid(
                        torch.arange(x_min, x_max, dtype=torch.float32, device=device),
                        torch.arange(y_min, y_max, dtype=torch.float32, device=device),
                        indexing='ij'
                    )
                    
                    # Gaussian kernel (keep differentiability with actual x, y)
                    x_tensor = pts_norm[b, i, 0]
                    y_tensor = pts_norm[b, i, 1]
                    kernel = torch.exp(-((xx - x_tensor)**2 + (yy - y_tensor)**2) / (2 * sigma**2))
                    grid[b, x_min:x_max, y_min:y_max] += kernel
        
        return grid
    
    def compute_radial_spectrum(self, fft_2d: torch.Tensor) -> torch.Tensor:
        """Compute radial (azimuthally averaged) power spectrum.
        
        Args:
            fft_2d: (B, H, W) 2D FFT output (complex or magnitude)
        
        Returns:
            (B, num_radial_bins) radial power spectrum
        """
        B, H, W = fft_2d.shape
        device = fft_2d.device
        
        # Compute power (magnitude squared)
        if torch.is_complex(fft_2d):
            power = torch.abs(fft_2d) ** 2
        else:
            power = fft_2d ** 2
        
        # Center coordinates
        cy, cx = H // 2, W // 2
        
        # Create coordinate grid
        yy, xx = torch.meshgrid(
            torch.arange(H, dtype=torch.float32, device=device),
            torch.arange(W, dtype=torch.float32, device=device),
            indexing='ij'
        )
        
        # Distance from center
        r = torch.sqrt((xx - cx)**2 + (yy - cy)**2)
        r_max = torch.sqrt(torch.tensor(cx**2 + cy**2, dtype=torch.float32, device=device))
        
        # Bin radii
        r_bins = torch.linspace(0, r_max.item(), self.num_radial_bins + 1, device=device)
        
        # Compute radial average
        radial_spectrum = torch.zeros(B, self.num_radial_bins, device=device)
        for bin_idx in range(self.num_radial_bins):
            mask = (r >= r_bins[bin_idx]) & (r < r_bins[bin_idx + 1])
            if mask.sum() > 0:
                for b in range(B):
                    radial_spectrum[b, bin_idx] = power[b, mask].mean()
        
        # Normalize
        radial_spectrum = radial_spectrum / (radial_spectrum.max(dim=1, keepdim=True)[0] + 1e-8)
        
        return radial_spectrum
    
    def forward(self, pred_points: torch.Tensor, gt_points: torch.Tensor) -> torch.Tensor:
        """Compute spectral loss between predicted and GT point clouds.
        
        Args:
            pred_points: (B, N, 2) predicted points in [-1, 1]
            gt_points: (B, M, 2) ground truth points in [-1, 1]
        
        Returns:
            MSE loss between radial power spectra
        """
        # Rasterize both to grids
        grid_pred = self.rasterize_points(pred_points, self.grid_size)
        grid_gt = self.rasterize_points(gt_points, self.grid_size)
        
        # Compute FFT
        fft_pred = torch.fft.fft2(grid_pred)
        fft_gt = torch.fft.fft2(grid_gt)
        
        # Compute radial power spectra
        spectrum_pred = self.compute_radial_spectrum(fft_pred)
        spectrum_gt = self.compute_radial_spectrum(fft_gt)
        
        # MSE loss
        loss = F.mse_loss(spectrum_pred, spectrum_gt)
        
        return loss



class AdaptiveRepulsionLoss(nn.Module):
    """Adaptive Repulsion Loss with density-aware spacing.
    
    Unlike fixed-radius repulsion, this adapts the required spacing based on
    the local image intensity. Dark areas (high density) allow closer points,
    while light areas (low density) require larger spacing.
    
    This is the "physics" that makes stippling work - points naturally pack
    tighter where more density is needed.
    """
    
    def __init__(self, base_radius: float = 0.02, min_radius: float = 0.005, max_radius: float = 0.1):
        """Initialize adaptive repulsion loss.
        
        Args:
            base_radius: Base repulsion radius (scaled by intensity)
            min_radius: Minimum allowed radius (prevents infinite packing in black)
            max_radius: Maximum allowed radius (prevents explosion in white)
        """
        super().__init__()
        self.base_radius = base_radius
        self.min_radius = min_radius
        self.max_radius = max_radius
    
    def forward(self, points: torch.Tensor, images: torch.Tensor) -> torch.Tensor:
        """Compute adaptive repulsion loss.
        
        Args:
            points: (B, N, 2) predicted points in [-1, 1]
            images: (B, 1, H, W) conditioning image
        
        Returns:
            Adaptive repulsion loss (scalar)
        """
        B, N, _ = points.shape
        device = points.device
        
        # 1. Sample image intensity at each point location using grid_sample
        # grid_sample expects (B, N, 1, 2) for 2D sampling
        grid = points.unsqueeze(2)  # (B, N, 1, 2)
        # Output: (B, 1, N, 1) -> squeeze to (B, N)
        intensity = F.grid_sample(
            images, grid, mode='bilinear', padding_mode='border', align_corners=True
        ).squeeze(-1).squeeze(1)  # (B, N)
        
        # 2. Compute dynamic radius per point
        # IMPORTANT: Invert intensity! In stippling:
        # - Dark pixels (intensity ~0) = HIGH density = SMALLER spacing allowed
        # - Light pixels (intensity ~1) = LOW density = LARGER spacing required
        # So we use (1 - intensity) as our density proxy
        density = 1.0 - intensity  # Now: dark=1 (high density), light=0 (low density)
        
        # Formula: R = base_radius / sqrt(density + epsilon)
        # This gives R ~ 1/sqrt(density), which matches Lloyd's relaxation physics
        # Higher density -> smaller radius -> points can be closer
        dynamic_r = self.base_radius / torch.sqrt(density + 0.1)
        dynamic_r = dynamic_r.clamp(min=self.min_radius, max=self.max_radius)
        
        # 3. Compute pairwise distance matrix
        loc = points.unsqueeze(2)  # (B, N, 1, 2)
        ref = points.unsqueeze(1)  # (B, 1, N, 2)
        dist = torch.sqrt(torch.sum((loc - ref) ** 2, dim=-1) + 1e-6)  # (B, N, N)
        
        # 4. Mask self-distances (diagonal)
        mask = torch.eye(N, device=device, dtype=torch.bool).unsqueeze(0)
        dist = dist.masked_fill(mask, float('inf'))
        
        # 5. Compute required distance for each pair
        # For pair (i, j), required distance = average of their radii
        r_i = dynamic_r.unsqueeze(2)  # (B, N, 1)
        r_j = dynamic_r.unsqueeze(1)  # (B, 1, N)
        required_dist = (r_i + r_j) / 2.0  # (B, N, N)
        
        # 6. Penalty for pairs closer than required
        penalty = F.relu(required_dist - dist)
        
        return torch.mean(penalty)


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
_grid_density_loss = GridDensityLoss(grid_sizes=(32, 64))
_spectral_loss = SpectralBluenoiseLoss(grid_size=256, num_radial_bins=32)
_adaptive_repulsion_loss = AdaptiveRepulsionLoss(base_radius=0.02, min_radius=0.005, max_radius=0.1)

# Sinkhorn loss for Optimal Transport matching (key for blue noise quality)
# OPTIMIZED: blur=0.02, scaling=0.8, grid_size=32 -> 12x faster than original
# Original (blur=0.01, scaling=0.9, grid=64): 2479ms/batch
# Optimized: ~200ms/batch with 0.90 gradient similarity
_sinkhorn_loss = SinkhornDensityLoss() if SinkhornDensityLoss is not None else None  # Uses optimized defaults


def train_step(
    model: torch.nn.Module,
    scheduler: DDPMScheduler,
    x_0: torch.Tensor,
    image: torch.Tensor,
    device: str = "cuda",
    chamfer_weight: float = 10.0,
    sinkhorn_weight: float = 0.0,
    repulsion_weight: float = 0.0,
    grid_density_weight: float = 0.0,
    spectral_weight: float = 0.0,
    use_adaptive_repulsion: bool = False,
) -> dict:
    """Single training step predicting x_start with loss combination.
    
    All losses are computed ONLY if their weight > 0. This avoids unnecessary
    computation and makes the API simple: set weight > 0 to enable a loss.
    
    Available losses:
    - chamfer: Position accuracy (match GT point positions)
    - sinkhorn: Optimal Transport for blue noise distribution
    - repulsion: Point spacing (prevents clustering)
    - grid_density: Multi-scale density matching
    - spectral: Frequency domain blue-noise enforcement
    
    Args:
        model: Point-DiT model
        scheduler: DDPM scheduler
        x_0: (B, N, 2) clean target points
        image: (B, 1, H, W) conditioning image
        device: device string
        chamfer_weight: Weight for Chamfer loss (0 = disabled)
        sinkhorn_weight: Weight for Sinkhorn OT loss (0 = disabled)
        repulsion_weight: Weight for repulsion loss (0 = disabled)
        grid_density_weight: Weight for grid density loss (0 = disabled)
        spectral_weight: Weight for spectral loss (0 = disabled)
        use_adaptive_repulsion: If True and repulsion_weight > 0, use AdaptiveRepulsionLoss
    
    Returns:
        Dict with 'loss' (combined) and individual loss components
    """
    batch_size = x_0.shape[0]

    noise = torch.randn_like(x_0)
    timesteps = torch.randint(0, scheduler.num_train_timesteps, (batch_size,), device=device, dtype=torch.long)

    x_t = scheduler.add_noise(x_0, noise, timesteps)
    pred_x0 = model(x_t, timesteps, image)

    # Initialize loss and tracking dict
    loss = 0.0
    losses = {
        'chamfer': 0.0,
        'sinkhorn': 0.0,
        'repulsion': 0.0,
        'grid_density': 0.0,
        'spectral': 0.0,
    }
    
    # Chamfer: match point positions to GT
    if chamfer_weight > 0:
        chamfer = _chamfer_loss(pred_x0, x_0)
        loss = loss + chamfer_weight * chamfer
        losses['chamfer'] = chamfer.item()
    
    # Sinkhorn: Optimal Transport for blue noise distribution
    if sinkhorn_weight > 0 and _sinkhorn_loss is not None:
        sinkhorn = _sinkhorn_loss(pred_x0, image)
        loss = loss + sinkhorn_weight * sinkhorn
        losses['sinkhorn'] = sinkhorn.item()
    
    # Repulsion: point spacing
    if repulsion_weight > 0:
        if use_adaptive_repulsion:
            repulsion = _adaptive_repulsion_loss(pred_x0, image)
        else:
            repulsion = _repulsion_loss(pred_x0)
        loss = loss + repulsion_weight * repulsion
        losses['repulsion'] = repulsion.item()
    
    # Grid density: multi-scale density matching
    if grid_density_weight > 0:
        grid_density = _grid_density_loss(pred_x0, image)
        loss = loss + grid_density_weight * grid_density
        losses['grid_density'] = grid_density.item()
    
    # Spectral: Fourier-domain blue-noise enforcement
    if spectral_weight > 0:
        spectral = _spectral_loss(pred_x0, x_0)
        loss = loss + spectral_weight * spectral
        losses['spectral'] = spectral.item()
    
    losses['loss'] = loss
    return losses


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
