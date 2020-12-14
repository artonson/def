import torch.nn as nn

from ...utils.init import initialize_head


class PixelSegmentator(nn.Module):

    def __init__(self, feature_extractor, segmentation_head):
        super().__init__()
        self.feature_extractor = feature_extractor
        self.segmentation_head = segmentation_head

    def initialize(self):
        initialize_head(self.segmentation_head)

    def forward(self, x):
        return self.segmentation_head(self.feature_extractor(x))
