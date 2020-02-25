from sharpf.modules import module_by_kind
from sharpf.modules.base import ParameterizedModule, load_with_spec
import torch
from torch import nn


class GenericPointBasedNet(ParameterizedModule):
    def __init__(self, encoder_blocks):
        super(GenericPointBasedNet, self).__init__()
        self.encoder_blocks = encoder_blocks

    def forward(self, points):
        activations = [points]
        features = points
        for block in self.encoder_blocks:
            features = block(features)
            activations.append(features)
        return features

    @classmethod
    def from_spec(cls, spec):
        blocks = []
        for block_spec in spec['encoder_blocks']:
            block = load_with_spec(block_spec, module_by_kind)
            blocks.append(block)
        return cls(blocks)


class DGCNN(ParameterizedModule):
    def __init__(self, encoder_blocks, decoder_blocks, **kwargs):
        super().__init__(**kwargs)
        self.encoder_blocks = nn.ModuleList(encoder_blocks)
        self.decoder_blocks = nn.ModuleList(decoder_blocks)

    def old_forward(self, points):
        activations = [points]
        features = points
        for block in self.encoder_blocks:
            features = block(features)
            activations.append(features)
        concatenated_features = torch.cat(activations[1:], dim=2)
        features = self.decoder_blocks[0](concatenated_features)
        num_point = points.size(1)
        expand = torch.repeat_interleave(features, num_point, 2)
        features = self.decoder_blocks[1](torch.cat((expand, concatenated_features), dim=2))
        features = self.decoder_blocks[2](features)
        features = features.squeeze()
        return features

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
    def from_spec(cls, spec):
        blocks_enc = []
        for block_spec in spec['encoder_blocks']:
            block = load_with_spec(block_spec, module_by_kind)
            blocks_enc.append(block)
        blocks_dec = []
        for block_spec in spec['decoder_blocks']:
            block = load_with_spec(block_spec, module_by_kind)
            blocks_dec.append(block)
        return cls(blocks_enc, blocks_dec)


