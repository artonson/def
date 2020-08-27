import torch.nn as nn

from .. import logits_to_scalar
from ...utils.init import initialize_head


class PixelRegressor(nn.Module):

    def __init__(self, feature_extractor, regression_head):
        super().__init__()
        self.feature_extractor = feature_extractor
        self.regression_head = regression_head

    def initialize(self):
        initialize_head(self.regression_head)

    def forward(self, x):
        return self.regression_head(self.feature_extractor(x))


class PixelRegressorHist(PixelRegressor):

    def forward(self, x):
        result = super().forward(x)
        if not self.training:
            result = result.permute(0, 2, 3, 1)  # (B, H, W, C)
            result = logits_to_scalar(result)  # (B, H, W, 1)
            result = result.permute(0, 3, 1, 2)  # (B, 1, H, W)
        return result
