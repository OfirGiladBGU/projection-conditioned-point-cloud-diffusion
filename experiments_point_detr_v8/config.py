"""Configuration for Point-RT V8 (Unsupervised Single-Pass Refinement)."""

from dataclasses import dataclass
from typing import Literal, Optional


@dataclass
class ModelConfig:
    """Point-RT model configuration (same as V7)."""
    
    n_points: int = 5000
    dim: int = 256
    n_layers: int = 6
    n_heads: int = 8
    image_size: int = 512
    
    # ResNet backbone
    use_resnet_backbone: bool = True
    resnet_pretrained: bool = True
    
    # Output configuration
    use_displacement: bool = True
    fourier_freqs: int = 32
    fourier_temperature: float = 100.0
    
    dropout: float = 0.0


@dataclass
class TrainingConfig:
    """Training configuration for unsupervised learning."""
    
    batch_size: int = 8  # Smaller batch due to Sinkhorn memory
    num_epochs: int = 50
    learning_rate: float = 1e-4  # Lower LR for physics-based learning
    weight_decay: float = 1e-5
    
    # LR schedule
    lr_scheduler: Literal["constant", "cosine", "warmup_cosine"] = "warmup_cosine"
    warmup_epochs: int = 3
    
    # === UNSUPERVISED LOSS WEIGHTS ===
    # These are the only two losses needed!
    sinkhorn_weight: float = 1.0      # Capacity constraint (density matching)
    repulsion_weight: float = 0.5     # Spacing constraint (blue noise)
    
    # Optional: GT supervision for hybrid training (set 0 for pure unsupervised)
    gt_weight: float = 0.0
    
    # Sinkhorn parameters
    sinkhorn_blur: float = 0.05
    sinkhorn_grid_size: int = 32
    
    # Repulsion parameters
    base_radius: float = 0.02
    
    # Training details
    max_grad_norm: float = 1.0
    log_every: int = 50
    save_every: int = 500
    sample_every: int = 500
    
    # Output
    output_dir: str = "./outputs_point_detr_v8"
    checkpoint_path: Optional[str] = None
    
    # Logging
    use_wandb: bool = True
    wandb_project: str = "PointRT-Unsupervised"
    wandb_run_name: Optional[str] = "v8_unsupervised"
    
    # Mixed precision
    use_amp: bool = False


@dataclass
class DataConfig:
    """Dataset configuration.
    
    NOTE: For unsupervised training, we only need source images!
    GT points (lloyd_dir) are OPTIONAL - only for hybrid training.
    """
    
    source_dir: str = "/groups/asharf_group/ofirgila/ControlNet/training/data_grads_v3/source"
    target_dir: Optional[str] = "/groups/asharf_group/ofirgila/ControlNet/training/data_grads_v3/target"  # Optional for eval
    lloyd_dir: Optional[str] = None  # Path to Lloyd GT points (optional for unsupervised)
    
    image_size: int = 512
    num_points: int = 5000
    val_split: float = 0.1
    
    # Data loading
    num_workers: int = 4
    pin_memory: bool = True


@dataclass
class Config:
    """Complete configuration."""
    
    model: ModelConfig
    training: TrainingConfig
    data: DataConfig
    
    device: str = "cuda"
    seed: int = 42
    
    @classmethod
    def default(cls):
        """Create default configuration."""
        return cls(
            model=ModelConfig(),
            training=TrainingConfig(),
            data=DataConfig(),
        )
    
    @classmethod
    def pure_unsupervised(cls):
        """Configuration for pure unsupervised training (no GT)."""
        config = cls.default()
        config.training.gt_weight = 0.0
        config.training.sinkhorn_weight = 1.0
        config.training.repulsion_weight = 0.5
        return config
    
    @classmethod 
    def hybrid(cls):
        """Configuration for hybrid training (physics + weak GT)."""
        config = cls.default()
        config.training.gt_weight = 0.1  # Small GT weight
        config.training.sinkhorn_weight = 1.0
        config.training.repulsion_weight = 0.5
        return config
    
    def to_dict(self) -> dict:
        """Convert to dictionary for logging."""
        return {
            "model": self.model.__dict__,
            "training": self.training.__dict__,
            "data": self.data.__dict__,
            "device": self.device,
            "seed": self.seed,
        }
