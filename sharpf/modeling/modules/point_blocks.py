import torch
import torch.nn as nn


class PointOpBlock(nn.Module):

    def __init__(self,
                 neighbours=nn.Identity(),
                 local_transform=nn.Identity(),
                 feature_extractor=nn.Identity(),
                 aggregation=nn.Identity(),
                 interpolation=nn.Identity(),
                 in_features=None):
        super(PointOpBlock, self).__init__()
        in_features = list(in_features) if in_features is not None else []
        self.in_features = in_features
        self._op = torch.nn.Sequential(
            neighbours, local_transform, feature_extractor, aggregation, interpolation
        )

    def forward(self, points):
        block_features = self._op.forward(points)
        return block_features
