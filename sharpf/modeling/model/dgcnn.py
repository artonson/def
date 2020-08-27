import torch
import torch.nn as nn


class DGCNN(nn.Module):

    def __init__(self, encoder_blocks, decoder_blocks):
        super().__init__()
        self.encoder_blocks = nn.ModuleList(encoder_blocks)
        self.decoder_blocks = nn.ModuleList(decoder_blocks)

    def forward(self, points):
        activations = []
        features = points
        for block in self.encoder_blocks:
            features = block(features)
            activations.append(features)
        features = []
        for idx, block in enumerate(self.decoder_blocks):
            concatenated_features = torch.cat(
                features + [activations[feat] for feat in block.in_features],
                dim=2
            )
            features = [block(concatenated_features)]
        features = features[0].squeeze(-1).squeeze(-1)
        return features
