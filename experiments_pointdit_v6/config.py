"""Configuration for Point-DiT experiments (V6.1 with Exact OT + Density Input)."""

from dataclasses import dataclass
from typing import Literal, Union


@dataclass
class ModelConfig:
    """Point-DiT model configuration.
    
    V6.1 Enhancements:
    - Tactile Density Sensors: Points receive local pixel intensity directly
    - Input becomes (x, y, intensity) instead of just (x, y)
    - Points immediately know their local density requirement
    
    V6 Scaling (base):
    - Scaled from 1M params (dim=128, n_layers=4) to 5M params (dim=256, n_layers=6)
    - This provides ~10x more computational capacity for learning complex N-body interactions
    - Training speed ~1.5 hours/epoch (vs 0.75 hours for v5)
    """

    # num_points: int = 4096
    n_points: int = 5000
    dim: int = 256  # Increased from 128 for better capacity
    n_layers: int = 6  # Increased from 4 for more self-attention rounds
    n_heads: int = 8  # Increased from 4 to match larger dim
    image_size: int = 512
    dropout: float = 0.0
    use_density_input: bool = True  # NEW V6.1: Tactile density sensors


@dataclass
class DiffusionConfig:
    """Diffusion process configuration.
    
    V6.1 Enhancement: Exact Optimal Transport Matching
    - use_exact_ot: Use Hungarian algorithm for exact OT matching
    - exact_ot_subsample: Subsample size for tractable computation
    """

    num_train_timesteps: int = 1000
    num_inference_steps: int = 50
    beta_start: float = 1e-4
    beta_end: float = 0.02
    beta_schedule: Literal["linear", "cosine"] = "linear"
    
    # V6.1: Optimal Transport configuration
    # Note: Exact Hungarian matching is O(N³) = extremely slow!
    # For 5000 points: ~45 seconds per batch. Use only for research.
    use_exact_ot: bool = False  # DISABLED - Hilbert sort is fast approximation
    exact_ot_subsample: int = 500  # If enabled, subsample for speed


@dataclass
class TrainingConfig:
    """Training configuration with spectral loss support."""

    batch_size: int = 8
    num_epochs: int = 50  # Phase 1 (40 epochs) + Phase 2 (10 epochs)
    learning_rate: float = 1e-4
    weight_decay: float = 1e-5

    lr_scheduler: Literal["constant", "cosine", "warmup_cosine"] = "warmup_cosine"
    warmup_epochs: int = 5

    max_grad_norm: float = 1.0

    log_every: int = 50
    save_every: int = 1000
    sample_every: int = 500

    use_wandb: bool = True
    wandb_project: str = "PointDiT"
    wandb_run_name: Union[str, None] = "v6_scaled"

    output_dir: str = "./outputs_pointdit_v6_hybrid"
    checkpoint_path: Union[str, None] = None
    
    # Loss weights for Hybrid training
    use_spectral_loss: bool = False  # Disabled - V5 testing showed spectral loss degraded performance
    spectral_weight: float = 0.0  # Not used


@dataclass
class DataConfig:
    """Dataset configuration."""

    # source_dir: str = "/groups/asharf_group/ofirgila/ControlNet/training/fill50k-gs/source"
    # target_dir: str = "/groups/asharf_group/ofirgila/ControlNet/training/fill50k-gs/target"
    source_dir: str = "/groups/asharf_group/ofirgila/ControlNet/training/data_grads_v3/source"
    target_dir: str = "/groups/asharf_group/ofirgila/ControlNet/training/data_grads_v3/target"
    image_size: int = 512
    # num_points: int = 2048
    num_points: int = 5000
    val_split: float = 0.1


@dataclass
class Config:
    """Complete configuration."""

    model: ModelConfig
    diffusion: DiffusionConfig
    training: TrainingConfig
    data: DataConfig

    device: str = "cuda"
    seed: int = 42
    num_workers: int = 4

    @classmethod
    def default(cls):
        """Create default configuration."""
        return cls(
            model=ModelConfig(),
            diffusion=DiffusionConfig(),
            training=TrainingConfig(),
            data=DataConfig(),
        )
