"""Configuration for Point-DiT experiments (V4 stable)."""

from dataclasses import dataclass
from typing import Literal, Union


@dataclass
class ModelConfig:
    """Point-DiT model configuration."""

    # num_points: int = 4096
    n_points: int = 5000
    dim: int = 128
    n_layers: int = 4
    n_heads: int = 4
    image_size: int = 512
    dropout: float = 0.0


@dataclass
class DiffusionConfig:
    """Diffusion process configuration."""

    num_train_timesteps: int = 1000
    num_inference_steps: int = 50
    beta_start: float = 1e-4
    beta_end: float = 0.02
    beta_schedule: Literal["linear", "cosine"] = "linear"


@dataclass
class TrainingConfig:
    """Training configuration."""

    batch_size: int = 8
    num_epochs: int = 100
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
    wandb_run_name: Union[str, None] = "v5_hybrid"

    output_dir: str = "./outputs_pointdit_v5_hybrid"
    checkpoint_path: Union[str, None] = None


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
