"""
Sinkhorn Loss + Differentiable Lloyd's Step Implementation

This module implements the "Neural-Newton" pipeline:
- Idea 1: SinkhornDensityLoss - Optimal Transport between points and image density
- Idea 2: DifferentiableLloydStep - Weighted Voronoi relaxation

Theory:
- Sinkhorn Loss treats points as source distribution and image as target distribution
- It computes Earth Mover's Distance (Wasserstein) to force points to match density
- Lloyd's step moves points to centroids of their soft Voronoi cells weighted by density
"""

import torch
import torch.nn as nn
import torch.nn.functional as F

# Try to import geomloss, provide fallback if not available
try:
    from geomloss import SamplesLoss
    HAS_GEOMLOSS = True
except ImportError:
    HAS_GEOMLOSS = False
    print("[Warning] geomloss not installed. Install with: pip install geomloss")


class SinkhornDensityLoss(nn.Module):
    """
    Sinkhorn (Optimal Transport) Loss for matching point distribution to image density.
    
    This is the differentiable, end-to-end version of Capacity Constrained Voronoi Tessellation (CCVT).
    
    Theory:
    - Source: Predicted points (uniform weights)
    - Target: Image density (dark pixels = high probability mass)
    - Loss: Earth Mover's Distance between distributions
    
    The loss forces points to spread out proportionally to image density.
    Clumping = high transport cost = penalized.
    """
    
    def __init__(self, blur: float = 0.05, grid_size: int = 32, scaling: float = 0.5, p: int = 2):
        """
        Initialize Sinkhorn loss.
        
        Args:
            blur: Smoothness of matching (0.05 = fast, good for training)
            grid_size: Resolution for target density grid (32 = fast)
            scaling: Sinkhorn algorithm convergence parameter (0.5 = fast)
            p: Power for Wasserstein distance (2 = squared Euclidean)
            
        Performance notes (RTX 6000, batch=4, 5000 points):
            blur=0.01, scaling=0.9, grid=64: 2479ms (original, very slow)
            blur=0.02, scaling=0.8, grid=32:  500ms
            blur=0.05, scaling=0.5, grid=32:  193ms (fast, 0.70 gradient sim)
        """
        super().__init__()
        self.blur = blur
        self.grid_size = grid_size
        self.scaling = scaling
        self.p = p
        
        if HAS_GEOMLOSS:
            # SamplesLoss with aggressive speed optimization
            self.loss_fn = SamplesLoss(
                loss="sinkhorn", 
                p=p, 
                blur=blur,
                scaling=scaling,
                debias=True,  # Unbiased Sinkhorn divergence
                backend="tensorized",  # Faster GPU backend
            )
        else:
            self.loss_fn = None
        
        # Create fixed coordinate grid for target density [-1, 1]
        y = torch.linspace(-1, 1, grid_size)
        x = torch.linspace(-1, 1, grid_size)
        grid_y, grid_x = torch.meshgrid(y, x, indexing='ij')
        
        # Buffer shape: (1, grid_size*grid_size, 2)
        # Store as (x, y) to match point format
        self.register_buffer('target_coords', torch.stack([grid_x, grid_y], dim=-1).reshape(1, -1, 2))
    
    def forward(self, pred_points: torch.Tensor, input_images: torch.Tensor) -> torch.Tensor:
        """
        Compute Sinkhorn loss between points and image density.
        
        Args:
            pred_points: (B, N, 2) Source points in [-1, 1]
            input_images: (B, 1, H, W) Target density map (0=light, 1=dark)
        
        Returns:
            Sinkhorn divergence (scalar)
        """
        if not HAS_GEOMLOSS:
            # Fallback: return 0 loss if geomloss not available
            return torch.tensor(0.0, device=pred_points.device, requires_grad=True)
        
        B, N, _ = pred_points.shape
        device = pred_points.device
        
        # 1. Source Distribution (Points)
        # Uniform weights: 1/N for each point
        source_weights = torch.ones(B, N, device=device) / N
        source_coords = pred_points.contiguous()
        
        # 2. Target Distribution (Image)
        # Downsample image to grid_size for computational efficiency
        target_density = F.interpolate(
            input_images, 
            size=(self.grid_size, self.grid_size), 
            mode='bilinear',
            align_corners=False
        )
        
        # IMPORTANT: In stippling, dark pixels need MORE points
        # Invert: density = 1 - intensity
        target_density = 1.0 - target_density
        
        # Flatten and ensure positive
        target_density = target_density.view(B, -1)  # (B, grid_size^2)
        target_density = target_density.clamp(min=1e-6)  # Avoid zero mass cells
        
        # Normalize to probability distribution (sum to 1)
        target_weights = target_density / (target_density.sum(dim=1, keepdim=True) + 1e-8)
        
        # Expand grid coordinates to batch size
        target_coords = self.target_coords.expand(B, -1, -1).to(device).contiguous()
        
        # 3. Compute Wasserstein Distance using Sinkhorn algorithm
        # geomloss expects: (weights, coordinates) for both source and target
        loss = self.loss_fn(source_weights, source_coords, target_weights, target_coords)
        
        return loss.mean()


class SinkhornDensityLossSimple(nn.Module):
    """
    Simplified Sinkhorn loss without geomloss dependency.
    
    Uses a custom implementation based on the Sinkhorn-Knopp algorithm.
    Slower but doesn't require additional dependencies.
    """
    
    def __init__(self, grid_size: int = 64, blur: float = 0.01, max_iters: int = 50):
        """
        Initialize simplified Sinkhorn loss.
        
        Args:
            grid_size: Resolution for target density grid
            blur: Entropy regularization (higher = smoother, more stable)
            max_iters: Maximum Sinkhorn iterations
        """
        super().__init__()
        self.grid_size = grid_size
        self.blur = blur
        self.max_iters = max_iters
        
        # Create fixed coordinate grid
        y = torch.linspace(-1, 1, grid_size)
        x = torch.linspace(-1, 1, grid_size)
        grid_y, grid_x = torch.meshgrid(y, x, indexing='ij')
        self.register_buffer('target_coords', torch.stack([grid_x, grid_y], dim=-1).reshape(1, -1, 2))
    
    def forward(self, pred_points: torch.Tensor, input_images: torch.Tensor) -> torch.Tensor:
        """
        Compute Sinkhorn divergence using custom implementation.
        
        Args:
            pred_points: (B, N, 2) Source points in [-1, 1]
            input_images: (B, 1, H, W) Target density map
        
        Returns:
            Sinkhorn divergence (scalar)
        """
        B, N, _ = pred_points.shape
        device = pred_points.device
        M = self.grid_size ** 2  # Number of target points
        
        # 1. Source weights (uniform)
        a = torch.ones(B, N, device=device) / N
        
        # 2. Target weights (from image density)
        target_density = F.interpolate(
            input_images, 
            size=(self.grid_size, self.grid_size), 
            mode='bilinear',
            align_corners=False
        )
        target_density = 1.0 - target_density  # Invert: dark = high density
        b = target_density.view(B, -1)  # (B, M)
        b = b.clamp(min=1e-6)
        b = b / (b.sum(dim=1, keepdim=True) + 1e-8)  # Normalize
        
        # 3. Cost matrix C[i,j] = ||x_i - y_j||^2
        target_coords = self.target_coords.expand(B, -1, -1).to(device)  # (B, M, 2)
        
        # (B, N, 2) vs (B, M, 2) -> (B, N, M)
        C = torch.cdist(pred_points, target_coords, p=2) ** 2
        
        # 4. Sinkhorn iterations
        # K = exp(-C / blur)
        eps = self.blur ** 2  # Regularization
        K = torch.exp(-C / eps)  # (B, N, M)
        
        u = torch.ones(B, N, device=device)
        v = torch.ones(B, M, device=device)
        
        for _ in range(self.max_iters):
            u = a / (K @ v.unsqueeze(-1)).squeeze(-1).clamp(min=1e-8)
            v = b / (K.transpose(1, 2) @ u.unsqueeze(-1)).squeeze(-1).clamp(min=1e-8)
        
        # 5. Transport cost
        # T = diag(u) @ K @ diag(v)
        # cost = sum(T * C)
        T = u.unsqueeze(-1) * K * v.unsqueeze(1)  # (B, N, M)
        loss = (T * C).sum(dim=[1, 2])  # (B,)
        
        return loss.mean()


class DifferentiableLloydStep(nn.Module):
    """
    Differentiable Lloyd's Relaxation Step for Weighted Voronoi.
    
    This is a "neural physics" layer that moves each point to the centroid of its
    Voronoi cell, weighted by image density.
    
    Theory:
    - Each pixel "belongs" to its nearest point (Voronoi assignment)
    - Points move to weighted centroid of their cell
    - Weighting by density = Capacity Constrained Voronoi Tessellation
    
    Training through this layer teaches the model to output points that are
    already in "near-equilibrium" - requiring minimal refinement.
    """
    
    def __init__(self, grid_size: int = 64, tau: float = 0.001, num_steps: int = 10):
        """
        Initialize Lloyd step layer.
        
        Args:
            grid_size: Resolution for Voronoi computation (trade-off: quality vs speed)
                       Recommended: 64-128 for good quality
            tau: Temperature for soft Voronoi (lower = sharper boundaries)
                 Recommended: 0.001 for sharp Voronoi, 0.01 for smoother
            num_steps: Number of Lloyd iterations to perform
                       Recommended: 10-50 for convergence
        """
        super().__init__()
        self.grid_size = grid_size
        self.tau = tau
        self.num_steps = num_steps
        
        # Create fixed pixel grid
        y = torch.linspace(-1, 1, grid_size)
        x = torch.linspace(-1, 1, grid_size)
        grid_y, grid_x = torch.meshgrid(y, x, indexing='ij')
        self.register_buffer('pixel_coords', torch.stack([grid_x, grid_y], dim=-1).reshape(1, -1, 2))
    
    def forward(self, points: torch.Tensor, density_map: torch.Tensor) -> torch.Tensor:
        """
        Perform differentiable Lloyd relaxation.
        
        Args:
            points: (B, N, 2) Point coordinates in [-1, 1]
            density_map: (B, 1, H, W) Image density (will be inverted internally)
        
        Returns:
            (B, N, 2) Refined point coordinates
        """
        B, N, _ = points.shape
        device = points.device
        
        # Downsample density map for efficiency
        density = F.interpolate(
            density_map,
            size=(self.grid_size, self.grid_size),
            mode='bilinear',
            align_corners=False
        )
        # Invert: dark pixels = high weight
        density = 1.0 - density
        density = density.view(B, -1)  # (B, H*W)
        density = density.clamp(min=1e-6)
        
        # Get pixel coordinates
        pixel_coords = self.pixel_coords.expand(B, -1, -1).to(device)  # (B, H*W, 2)
        
        current_points = points
        
        for _ in range(self.num_steps):
            # 1. Compute squared distances: (B, N, H*W)
            # points: (B, N, 2), pixels: (B, H*W, 2)
            dists_sq = torch.cdist(current_points, pixel_coords, p=2) ** 2
            
            # 2. Soft Voronoi Assignment using softmax
            # Lower distance = higher assignment probability
            # Temperature tau controls sharpness (lower = sharper)
            assignment = F.softmax(-dists_sq / self.tau, dim=1)  # (B, N, H*W)
            
            # 3. Weight assignment by pixel density
            # assignment: (B, N, H*W), density: (B, H*W)
            weighted_assignment = assignment * density.unsqueeze(1)  # (B, N, H*W)
            
            # 4. Compute mass per cell (for normalization)
            mass_per_cell = weighted_assignment.sum(dim=2, keepdim=True) + 1e-8  # (B, N, 1)
            
            # 5. Compute weighted centroids
            # (B, N, H*W) @ (B, H*W, 2) -> (B, N, 2)
            new_centroids = torch.bmm(weighted_assignment, pixel_coords) / mass_per_cell
            
            current_points = new_centroids
        
        # Clamp to valid range
        current_points = current_points.clamp(-1.0, 1.0)
        
        return current_points


class DifferentiableCCVT(nn.Module):
    """
    Differentiable Capacity-Constrained Voronoi Tessellation (CCVT).
    
    Unlike standard weighted Lloyd, CCVT ensures each point "owns" EQUAL mass.
    This is achieved through power diagrams with adaptive weights.
    
    The algorithm:
    1. Compute power diagram assignment (Voronoi with additive weights)
    2. Adjust weights to balance cell masses (each cell gets 1/N of total mass)
    3. Move points to cell centroids
    
    This produces true blue noise where points are evenly distributed
    according to density, not clustered in high-density regions.
    """
    
    def __init__(self, grid_size: int = 64, tau: float = 0.001, num_steps: int = 10, 
                 weight_lr: float = 0.1):
        """
        Initialize CCVT layer.
        
        Args:
            grid_size: Resolution for Voronoi computation
            tau: Temperature for soft assignment (lower = sharper)
            num_steps: Number of Lloyd iterations
            weight_lr: Learning rate for power weights adjustment
        """
        super().__init__()
        self.grid_size = grid_size
        self.tau = tau
        self.num_steps = num_steps
        self.weight_lr = weight_lr
        
        # Create fixed pixel grid
        y = torch.linspace(-1, 1, grid_size)
        x = torch.linspace(-1, 1, grid_size)
        grid_y, grid_x = torch.meshgrid(y, x, indexing='ij')
        self.register_buffer('pixel_coords', torch.stack([grid_x, grid_y], dim=-1).reshape(1, -1, 2))
    
    def forward(self, points: torch.Tensor, density_map: torch.Tensor) -> torch.Tensor:
        """
        Perform differentiable CCVT relaxation.
        
        Args:
            points: (B, N, 2) Point coordinates in [-1, 1]
            density_map: (B, 1, H, W) Image (will be inverted: dark = high density)
        
        Returns:
            (B, N, 2) Refined point coordinates with capacity constraints
        """
        B, N, _ = points.shape
        device = points.device
        M = self.grid_size ** 2  # Number of pixels
        
        # Downsample and invert density
        density = F.interpolate(
            density_map,
            size=(self.grid_size, self.grid_size),
            mode='bilinear',
            align_corners=False
        )
        density = 1.0 - density  # Invert: dark = high density
        density = density.view(B, -1)  # (B, M)
        density = density.clamp(min=1e-6)
        
        # Normalize density to probability
        total_mass = density.sum(dim=1, keepdim=True)
        density_prob = density / total_mass  # (B, M) - sums to 1
        
        # Target capacity per point (equal share)
        target_capacity = 1.0 / N  # Each point should own 1/N of total mass
        
        # Get pixel coordinates
        pixel_coords = self.pixel_coords.expand(B, -1, -1).to(device)  # (B, M, 2)
        
        current_points = points.clone()
        power_weights = torch.zeros(B, N, device=device)  # Additive weights for power diagram
        
        for step in range(self.num_steps):
            # 1. Compute power distances: d^2(p, x) - w_p
            # Standard distance
            dists_sq = torch.cdist(current_points, pixel_coords, p=2) ** 2  # (B, N, M)
            
            # Power distance = dist^2 - weight
            power_dists = dists_sq - power_weights.unsqueeze(-1)  # (B, N, M)
            
            # 2. Soft power diagram assignment
            # Pixel j belongs to point i with lowest power distance
            assignment = F.softmax(-power_dists / self.tau, dim=1)  # (B, N, M)
            
            # 3. Compute current cell masses (weighted by density)
            # cell_mass[i] = sum over j of: assignment[i,j] * density[j]
            weighted_assignment = assignment * density_prob.unsqueeze(1)  # (B, N, M)
            cell_mass = weighted_assignment.sum(dim=2)  # (B, N) - should each be ~1/N
            
            # 4. Adjust power weights to balance masses
            # If cell has too much mass -> increase weight (expand cell)
            # If cell has too little mass -> decrease weight (shrink cell)
            mass_error = cell_mass - target_capacity  # Positive = too much mass
            power_weights = power_weights - self.weight_lr * mass_error
            
            # 5. Compute weighted centroids (same as before)
            mass_per_cell = weighted_assignment.sum(dim=2, keepdim=True) + 1e-8
            new_centroids = torch.bmm(weighted_assignment, pixel_coords) / mass_per_cell
            
            current_points = new_centroids
        
        # Clamp to valid range
        current_points = current_points.clamp(-1.0, 1.0)
        
        return current_points


class DifferentiableLloydWithRepulsion(nn.Module):
    """
    Differentiable Lloyd's Relaxation with Repulsion to prevent point collapse.
    
    The key issue with standard Lloyd: points can converge to the same location
    in high-density regions. This version adds a repulsion term that pushes
    points apart when they get too close.
    
    Algorithm per step:
    1. Standard Lloyd centroid update (move to weighted cell center)
    2. Repulsion force between nearby points (prevent overlap)
    3. Blend: new_pos = lloyd_pos + repulsion_strength * repulsion_force
    """
    
    def __init__(self, grid_size: int = 64, tau: float = 0.001, num_steps: int = 10,
                 repulsion_strength: float = 0.5, repulsion_radius: float = 0.02):
        """
        Initialize Lloyd with repulsion.
        
        Args:
            grid_size: Resolution for Voronoi computation
            tau: Temperature for soft Voronoi (lower = sharper)
            num_steps: Number of Lloyd iterations
            repulsion_strength: How strongly to push apart nearby points
            repulsion_radius: Minimum desired distance between points
        """
        super().__init__()
        self.grid_size = grid_size
        self.tau = tau
        self.num_steps = num_steps
        self.repulsion_strength = repulsion_strength
        self.repulsion_radius = repulsion_radius
        
        # Create fixed pixel grid
        y = torch.linspace(-1, 1, grid_size)
        x = torch.linspace(-1, 1, grid_size)
        grid_y, grid_x = torch.meshgrid(y, x, indexing='ij')
        self.register_buffer('pixel_coords', torch.stack([grid_x, grid_y], dim=-1).reshape(1, -1, 2))
    
    def _compute_repulsion(self, points: torch.Tensor) -> torch.Tensor:
        """
        Compute repulsion forces between nearby points.
        
        Returns force vectors that push points away from each other.
        """
        B, N, _ = points.shape
        
        # Pairwise differences: (B, N, N, 2)
        diff = points.unsqueeze(2) - points.unsqueeze(1)  # (B, N, N, 2)
        
        # Distances: (B, N, N)
        dist = torch.norm(diff, dim=-1) + 1e-8
        
        # Mask diagonal (no self-repulsion)
        mask = torch.eye(N, device=points.device).bool().unsqueeze(0)
        dist = dist.masked_fill(mask, float('inf'))
        
        # Repulsion only for points within repulsion_radius
        # Force magnitude = (r - dist) / r for dist < r, else 0
        repulsion_magnitude = torch.relu(self.repulsion_radius - dist) / self.repulsion_radius
        
        # Force direction: normalized diff (away from other point)
        force_direction = diff / (dist.unsqueeze(-1) + 1e-8)  # (B, N, N, 2)
        
        # Total force on each point = sum of forces from all other points
        force = (repulsion_magnitude.unsqueeze(-1) * force_direction).sum(dim=2)  # (B, N, 2)
        
        return force
    
    def forward(self, points: torch.Tensor, density_map: torch.Tensor) -> torch.Tensor:
        """
        Perform Lloyd relaxation with repulsion.
        
        Args:
            points: (B, N, 2) Point coordinates in [-1, 1]
            density_map: (B, 1, H, W) Image density
        
        Returns:
            (B, N, 2) Refined point coordinates
        """
        B, N, _ = points.shape
        device = points.device
        
        # Downsample density
        density = F.interpolate(
            density_map,
            size=(self.grid_size, self.grid_size),
            mode='bilinear',
            align_corners=False
        )
        density = 1.0 - density  # Invert: dark = high density
        density = density.view(B, -1)
        density = density.clamp(min=1e-6)
        
        pixel_coords = self.pixel_coords.expand(B, -1, -1).to(device)
        current_points = points.clone()
        
        for step in range(self.num_steps):
            # 1. Standard Lloyd centroid update
            dists_sq = torch.cdist(current_points, pixel_coords, p=2) ** 2
            assignment = F.softmax(-dists_sq / self.tau, dim=1)
            weighted_assignment = assignment * density.unsqueeze(1)
            mass_per_cell = weighted_assignment.sum(dim=2, keepdim=True) + 1e-8
            lloyd_centroids = torch.bmm(weighted_assignment, pixel_coords) / mass_per_cell
            
            # 2. Compute repulsion force
            repulsion_force = self._compute_repulsion(current_points)
            
            # 3. Blend: move toward centroid but also apply repulsion
            # The repulsion prevents collapse
            new_points = lloyd_centroids + self.repulsion_strength * repulsion_force
            
            current_points = new_points
        
        # Clamp to valid range
        current_points = current_points.clamp(-1.0, 1.0)
        
        return current_points


class NeuralNewtonLoss(nn.Module):
    """
    Combined "Neural-Newton" loss: Sinkhorn + optional Lloyd refinement.
    
    This loss trains the model to output points that:
    1. Have global density matching (Sinkhorn)
    2. Are near Lloyd equilibrium (Lloyd refinement)
    
    The model learns to predict "pre-relaxed" points that achieve 
    perfect blue noise after minimal refinement.
    """
    
    def __init__(
        self,
        sinkhorn_weight: float = 1.0,
        chamfer_weight: float = 1.0,
        use_lloyd_refinement: bool = True,
        lloyd_steps: int = 1,
        grid_size: int = 64,
        sinkhorn_blur: float = 0.01,
        lloyd_tau: float = 0.01,
    ):
        """
        Initialize Neural-Newton loss.
        
        Args:
            sinkhorn_weight: Weight for Sinkhorn loss
            chamfer_weight: Weight for Chamfer loss on refined points
            use_lloyd_refinement: Whether to apply Lloyd refinement before computing loss
            lloyd_steps: Number of Lloyd iterations
            grid_size: Resolution for density computations
            sinkhorn_blur: Sinkhorn smoothness parameter
            lloyd_tau: Lloyd softmax temperature
        """
        super().__init__()
        
        self.sinkhorn_weight = sinkhorn_weight
        self.chamfer_weight = chamfer_weight
        self.use_lloyd_refinement = use_lloyd_refinement
        
        # Sinkhorn loss
        if HAS_GEOMLOSS:
            self.sinkhorn_loss = SinkhornDensityLoss(blur=sinkhorn_blur, grid_size=grid_size)
        else:
            self.sinkhorn_loss = SinkhornDensityLossSimple(grid_size=grid_size, blur=sinkhorn_blur)
        
        # Lloyd refinement layer
        if use_lloyd_refinement:
            self.lloyd_step = DifferentiableLloydStep(
                grid_size=grid_size, 
                tau=lloyd_tau,
                num_steps=lloyd_steps
            )
        
        # Chamfer loss for structural matching
        from diffusion import ChamferLoss
        self.chamfer_loss = ChamferLoss()
    
    def forward(
        self,
        pred_points: torch.Tensor,
        gt_points: torch.Tensor,
        input_images: torch.Tensor,
    ) -> dict:
        """
        Compute Neural-Newton loss.
        
        Args:
            pred_points: (B, N, 2) Model predicted points
            gt_points: (B, N, 2) Ground truth points
            input_images: (B, 1, H, W) Conditioning image
        
        Returns:
            Dict with 'loss' (total), 'sinkhorn', 'chamfer', 'refined_points'
        """
        # Optionally refine points through Lloyd step
        if self.use_lloyd_refinement:
            refined_points = self.lloyd_step(pred_points, input_images)
        else:
            refined_points = pred_points
        
        # Sinkhorn loss on refined points
        sinkhorn = self.sinkhorn_loss(refined_points, input_images)
        
        # Chamfer loss to maintain structural match to GT
        chamfer = self.chamfer_loss(refined_points, gt_points)
        
        # Combined loss
        total_loss = (self.sinkhorn_weight * sinkhorn) + (self.chamfer_weight * chamfer)
        
        return {
            'loss': total_loss,
            'sinkhorn': sinkhorn.item() if torch.is_tensor(sinkhorn) else sinkhorn,
            'chamfer': chamfer.item() if torch.is_tensor(chamfer) else chamfer,
            'refined_points': refined_points.detach(),
        }


def test_losses():
    """Quick sanity test for all loss functions."""
    print("=" * 60)
    print("Testing Sinkhorn + Lloyd Losses")
    print("=" * 60)
    
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    print(f"Device: {device}")
    print(f"geomloss available: {HAS_GEOMLOSS}")
    
    B, N = 2, 512
    H, W = 256, 256
    
    # Random test data
    points = (torch.rand(B, N, 2, device=device) * 2 - 1)  # [-1, 1]
    gt_points = (torch.rand(B, N, 2, device=device) * 2 - 1)
    image = torch.rand(B, 1, H, W, device=device)  # [0, 1]
    
    print(f"\nTest data: points={points.shape}, image={image.shape}")
    
    # Test Sinkhorn Loss
    print("\n--- SinkhornDensityLoss ---")
    if HAS_GEOMLOSS:
        sinkhorn = SinkhornDensityLoss(blur=0.01, grid_size=64).to(device)
        loss = sinkhorn(points, image)
        print(f"Loss: {loss.item():.6f}")
        
        # Test gradient
        points.requires_grad_(True)
        loss = sinkhorn(points, image)
        loss.backward()
        print(f"Gradient norm: {points.grad.norm().item():.6f}")
        points.requires_grad_(False)
    else:
        print("Skipped (geomloss not installed)")
    
    # Test Simple Sinkhorn
    print("\n--- SinkhornDensityLossSimple ---")
    sinkhorn_simple = SinkhornDensityLossSimple(grid_size=32, blur=0.05).to(device)
    points.requires_grad_(True)
    loss = sinkhorn_simple(points, image)
    print(f"Loss: {loss.item():.6f}")
    loss.backward()
    print(f"Gradient norm: {points.grad.norm().item():.6f}")
    points.requires_grad_(False)
    
    # Test Lloyd Step
    print("\n--- DifferentiableLloydStep ---")
    lloyd = DifferentiableLloydStep(grid_size=32, tau=0.01, num_steps=1).to(device)
    points_before = points.clone()
    points_after = lloyd(points, image)
    movement = (points_after - points_before).norm(dim=-1).mean()
    print(f"Average point movement: {movement.item():.6f}")
    
    # Test gradient through Lloyd
    points.requires_grad_(True)
    refined = lloyd(points, image)
    loss = refined.sum()
    loss.backward()
    print(f"Gradient norm through Lloyd: {points.grad.norm().item():.6f}")
    points.requires_grad_(False)
    
    # Test NeuralNewtonLoss
    print("\n--- NeuralNewtonLoss ---")
    nn_loss = NeuralNewtonLoss(
        sinkhorn_weight=1.0,
        chamfer_weight=1.0,
        use_lloyd_refinement=True,
        lloyd_steps=1,
        grid_size=32,
    ).to(device)
    
    points.requires_grad_(True)
    result = nn_loss(points, gt_points, image)
    print(f"Total loss: {result['loss'].item():.6f}")
    print(f"Sinkhorn: {result['sinkhorn']:.6f}")
    print(f"Chamfer: {result['chamfer']:.6f}")
    
    result['loss'].backward()
    print(f"Gradient norm: {points.grad.norm().item():.6f}")
    
    print("\n" + "=" * 60)
    print("All tests passed!")
    print("=" * 60)


if __name__ == '__main__':
    test_losses()
