import torch
import torch.nn as nn
from omegaconf import DictConfig

from .build import MODEL_REGISTRY
from sharpf.utils.config import configurable
from ..modules.point_blocks import PointOpBlock


@MODEL_REGISTRY.register()
class DGCNN(nn.Module):

    @configurable
    def __init__(self, encoder_blocks, decoder_blocks, **kwargs):
        super().__init__(**kwargs)
        self.encoder_blocks = nn.ModuleList(encoder_blocks)
        self.decoder_blocks = nn.ModuleList(decoder_blocks)

    def forward(self, points):
        activations = {}
        features = points
        for idx, block in enumerate(self.encoder_blocks):
            features = block(features)
            activations[idx] = features
        features = []
        for idx, block in enumerate(self.decoder_blocks):
            concatenated_features = torch.cat(
                features + [activations[feat] for feat in block.in_features],
                dim=2
            )
            features = [block(concatenated_features)]

        features = features[0].squeeze()
        return features

    @classmethod
    def from_config(cls, cfg: DictConfig):
        return {
            "encoder_blocks": [PointOpBlock(block_cfg) for block_cfg in cfg.encoder_blocks],
            "decoder_blocks": [PointOpBlock(block_cfg) for block_cfg in cfg.decoder_blocks],
        }
