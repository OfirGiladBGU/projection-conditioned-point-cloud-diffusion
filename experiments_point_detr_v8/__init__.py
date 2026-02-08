"""
Point-RT V8: Unsupervised Stippling

Uses physics-based losses (Sinkhorn + AdaptiveRepulsion) instead of GT supervision.
"""

from .point_rt import PointRT
from .losses import UnsupervisedStipplingLoss, HybridStipplingLoss
from .config import Config
from .fast_init import fast_density_initialization, create_point_input_vectors

__all__ = [
    'PointRT',
    'UnsupervisedStipplingLoss',
    'HybridStipplingLoss',
    'Config',
    'fast_density_initialization',
    'create_point_input_vectors',
]
