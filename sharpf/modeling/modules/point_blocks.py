import hydra
import torch
import torch.nn as nn
from omegaconf import DictConfig
from sharpf.utils.config import configurable


class PointOpBlock(nn.Module):

    @configurable
    def __init__(self, neighbours, local_transform, feature_extractor, aggregation,
                 interpolation, in_features, **kwargs):
        super(PointOpBlock, self).__init__(**kwargs)
        in_features = list(in_features) if in_features is not None else []
        self.in_features = in_features
        self._op = torch.nn.Sequential(
            neighbours, local_transform, feature_extractor, aggregation, interpolation
        )

    def forward(self, points):
        block_features = self._op.forward(points)
        return block_features

    @classmethod
    def from_config(cls, cfg: DictConfig):
        return {
            'neighbours': hydra.utils.instantiate(cfg.neighbours),
            'local_transform': hydra.utils.instantiate(cfg.local_transform),
            'feature_extractor': hydra.utils.instantiate(cfg.features),
            'aggregation': hydra.utils.instantiate(cfg.aggregation),
            'interpolation': hydra.utils.instantiate(cfg.interpolation),
            'in_features': cfg.in_features
        }
