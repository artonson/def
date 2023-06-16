import torch.nn as nn

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

    def __init__(self, a: float, b: float, discretization: int, margin: int, *args, **kwargs):
        super().__init__(*args, **kwargs)
        assert discretization > 0 and margin >= 0 and a < b
        self.a = a
        self.b = b
        self.discretization = discretization
        self.margin = margin

    def forward(self, x):
        result = super().forward(x)  # (B, C, H, W)
        return result