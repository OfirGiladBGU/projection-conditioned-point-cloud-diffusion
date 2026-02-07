"""Configuration for Point-RT V7 (Single-Pass Refinement)."""

from dataclasses import dataclass
from typing import Literal, Optional


@dataclass
class ModelConfig:
    """Point-RT model configuration."""
    
    n_points: int = 5000
    dim: int = 256              # Embedding dimension
    n_layers: int = 6           # Transformer layers
    n_heads: int = 8            # Attention heads
    image_size: int = 512       # Input image size
    
    # ResNet backbone configuration
    use_resnet_backbone: bool = True
    resnet_pretrained: bool = True
    resnet_freeze_early: bool = False  # Freeze first N layers
    
    # Output configuration
    use_displacement: bool = True  # Predict (dx, dy) if True, else absolute coords
    fourier_freqs: int = 32     # Fourier positional embedding frequencies
    fourier_temperature: float = 100.0
    
    dropout: float = 0.0


@dataclass
class TrainingConfig:
    """Training configuration."""
    
    batch_size: int = 16
    num_epochs: int = 30
    learning_rate: float = 5e-4
    weight_decay: float = 1e-5
    
    # Learning rate schedule
    lr_scheduler: Literal["constant", "cosine", "warmup_cosine"] = "warmup_cosine"
    warmup_epochs: int = 2
    
    # Loss weights
    mse_weight: float = 1.0
    repulsion_weight: float = 0.1
    diversity_weight: float = 0.05
    
    # Training details
    max_grad_norm: float = 1.0
    log_every: int = 50
    save_every: int = 500
    sample_every: int = 1000
    
    # Checkpoint and output
    output_dir: str = "./outputs_point_detr_v7"
    checkpoint_path: Optional[str] = None
    
    # Logging
    use_wandb: bool = True
    wandb_project: str = "PointRT"
    wandb_run_name: Optional[str] = "v7_single_pass"
    
    # Mixed precision
    use_amp: bool = False


@dataclass
class DataConfig:
    """Dataset configuration."""
    
    source_dir: str = "/groups/asharf_group/ofirgila/ControlNet/training/data_grads_v3/source"
    target_dir: str = "/groups/asharf_group/ofirgila/ControlNet/training/data_grads_v3/target"
    lloyd_dir: Optional[str] = None  # Directory with Lloyd's ground truth (if available)
    
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
    
    def to_dict(self) -> dict:
        """Convert to dictionary for logging."""
        return {
            "model": self.model.__dict__,
            "training": self.training.__dict__,
            "data": self.data.__dict__,
            "device": self.device,
            "seed": self.seed,
        }
