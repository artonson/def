import hydra
import torch.nn as nn
from omegaconf import DictConfig

from sharpf.utils.config import configurable
from .build import MODEL_REGISTRY
from ...utils.init import initialize_head


@MODEL_REGISTRY.register()
class PixelRegressor(nn.Module):
    @configurable
    def __init__(self, feature_extractor, regression_head):
        super().__init__()
        self.feature_extractor = feature_extractor
        self.regression_head = regression_head

    def initialize(self):
        initialize_head(self.regression_head)

    def forward(self, x):
        return self.regression_head(self.feature_extractor(x))

    @classmethod
    def from_config(cls, cfg: DictConfig):
        return {
            "feature_extractor": hydra.utils.instantiate(cfg.feature_extractor),
            "regression_head": nn.Sequential(*[hydra.utils.instantiate(node) for node in cfg.regression_head])
        }

# still believe that single class or abstract class for pixel-task model would be better
@MODEL_REGISTRY.register()
class PixelSegmentator(nn.Module):
    @configurable
    def __init__(self, feature_extractor, segmentation_head):
        super().__init__()
        self.feature_extractor = feature_extractor
        self.segmentation_head = segmentation_head

    def initialize(self):
        initialize_head(self.segmentation_head)

    def forward(self, x):
        return self.segmentation_head(self.feature_extractor(x))

    @classmethod
    def from_config(cls, cfg: DictConfig):
        return {
            "feature_extractor": hydra.utils.instantiate(cfg.feature_extractor),
            "segmentation_head": nn.Sequential(*[hydra.utils.instantiate(node) for node in cfg.segmentation_head])
        }
