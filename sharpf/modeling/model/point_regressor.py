import torch.nn as nn

from ...utils.init import initialize_head


class PointRegressor(nn.Module):

    def __init__(self, feature_extractor, regression_head):
        super().__init__()
        self.feature_extractor = feature_extractor
        self.regression_head = regression_head

    def initialize(self):
        initialize_head(self.regression_head)

    def forward(self, x):
        """
        Args:
            x (Tensor): of shape (B, N, C)
        Returns (Tensor): of shape (B, N, C)
        """
        out = self.regression_head(self.feature_extractor(x.transpose(1, 2).contiguous())).transpose(1, 2).contiguous()
        return out


class PointRegressorHist(PointRegressor):

    def __init__(self, a: float, b: float, discretization: int, margin: int, *args, **kwargs):
        super().__init__(*args, **kwargs)
        assert discretization > 0 and margin >= 0 and a < b
        self.a = a
        self.b = b
        self.discretization = discretization
        self.margin = margin

    def forward(self, x):
        result = super().forward(x)  # (B, N, C)
        # if not self.training:
        #     result = logits_to_scalar(result, self.a, self.b, self.discretization, self.margin)  # (B, N, 1)
        return result
