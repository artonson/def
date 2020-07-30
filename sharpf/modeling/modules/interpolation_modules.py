# import torch
# import torch.nn as nn
#
# from contrib.pointweb.lib.pointops.functions import pointops
#
#
# class PointNet2Interpolation(nn.Module):
#     """
#     Implementation from PointWeb https://github.com/hszhao/PointWeb
#
#     Parameters
#     ----------
#         unknown : torch.Tensor
#             (B, n, 3) tensor of the xyz positions of the unknown features
#         known : torch.Tensor
#             (B, m, 3) tensor of the xyz positions of the known features
#         unknown_feats : torch.Tensor
#             (B, C1, n) tensor of the features to be propigated to
#         known_feats : torch.Tensor
#             (B, C2, m) tensor of features to be propigated
#     Returns
#     -------
#         new_features : torch.Tensor
#             (B, C2 + C1, n) tensor of the features of the unknown features
#     """
#
#     def __init__(self, **kwargs):
#         super().__init__(**kwargs)
#
#     def forward(self,
#                 unknown: torch.Tensor,
#                 known: torch.Tensor,
#                 unknown_feats: torch.Tensor,
#                 known_feats: torch.Tensor) -> torch.Tensor:
#         if known is not None:
#             dist, idx = pointops.nearestneighbor(unknown, known)
#             dist_recip = 1.0 / (dist + 1e-8)
#             norm = torch.sum(dist_recip, dim=2, keepdim=True)
#             weight = dist_recip / norm
#             interpolated_feats = pointops.interpolation(known_feats, idx, weight)
#         else:
#             interpolated_feats = known_feats.expand(*known_feats.size()[0:2], unknown.size(1))
#
#         if unknown_feats is not None:
#             new_features = torch.cat([interpolated_feats, unknown_feats], dim=1)  # (B, C2 + C1, n)
#         else:
#             new_features = interpolated_feats
#
#         return new_features.unsqueeze(-1)
