import torch
import torch.nn as nn

from sharpf.modeling import logits_to_scalar


class DGCNN(nn.Module):

    def __init__(self, encoder_blocks, decoder_blocks):
        super().__init__()
        self.encoder_blocks = nn.ModuleList(encoder_blocks)
        self.decoder_blocks = nn.ModuleList(decoder_blocks)

    def forward(self, points):
        """
        Args:
            points (Tensor): of shape (B, N, C_in)
        Returns:
            Tensor: of shape (B, N, C_out)
        """
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
        features = features[0].squeeze(3)
        return features


class DGCNNHist(DGCNN):

    def __init__(self, a: float, b: float, discretization: int, margin: int, *args, **kwargs):
        super().__init__(*args, **kwargs)
        assert discretization > 0 and margin >= 0 and a < b
        self.a = a
        self.b = b
        self.discretization = discretization
        self.margin = margin

    def forward(self, points):
        """
        Args:
            points (Tensor): of shape (B, N, C_in)
        Returns:
            Tensor: of shape (B, N, C_out)
                if self.training=False then C_out=1 else C_out=number of logits
        """
        result = super().forward(points)
        if not self.training:
            result = logits_to_scalar(result, self.a, self.b, self.discretization, self.margin)
        return result
