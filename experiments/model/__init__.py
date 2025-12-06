import sys
from pathlib import Path

# Ensure parent directory is in path for absolute imports
_parent = Path(__file__).parent.parent
if str(_parent) not in sys.path:
    sys.path.insert(0, str(_parent))

from config.structured import ProjectConfig
from model.model import ConditionalPointCloudDiffusionModel
from model.model_coloring import PointCloudColoringModel
from model.model_utils import set_requires_grad


def get_model(cfg: ProjectConfig):
    model = ConditionalPointCloudDiffusionModel(**cfg.model)
    if cfg.run.freeze_feature_model:
        set_requires_grad(model.feature_model, False)
    return model


def get_coloring_model(cfg: ProjectConfig):
    model = PointCloudColoringModel(**cfg.model)
    if cfg.run.freeze_feature_model:
        set_requires_grad(model.feature_model, False)
    return model
