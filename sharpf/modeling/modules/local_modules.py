import torch
import torch.nn as nn
import torch.nn.functional as F

from .pt_utils import MaskedQueryAndGroup


class LocalDynamicGraph(nn.Module):
    """
    LocalDynamicGraph - class for Local transformation methods for ParameterizedPointNet.
    Current class deals with local dynamic graph construction for DGCNN.
    """

    def forward(self, x):
        """
        Args:
            x (tuple): of elements
                x[0] (Tensor): of shape (B, N, C). Batch of points
                x[1] (Tensor): of shape (B, N, k). Indices of nearest neighbour points for each point
        Returns:
            Tensor: of shape (B, N, C, k).
        """
        return local_dynamic_graph(x)


def local_dynamic_graph(x):
    """
    Args:
        x (tuple): of elements
            x[0] (Tensor): of shape (B, N, C). Batch of points
            x[1] (Tensor): of shape (B, N, k). Indices of nearest neighbour points for each point
    Returns:
        Tensor: of shape (B, N, C, k).
    """
    idx = x[1]
    x = x[0]

    batch_size, num_points, num_dims = x.shape

    k = idx.shape[2]

    idx_base = torch.arange(0, batch_size, device=idx.device).view(-1, 1, 1) * num_points  # (B, 1, 1)
    idx = idx + idx_base  # (B, N, k)
    idx = idx.view(-1)  # (B * N * k,)

    feature = x.view(batch_size * num_points, num_dims)[idx, :]  # (B * N * k, num_dims)
    feature = feature.view(batch_size, num_points, k, num_dims)  # (B, N, k, num_dims)

    x = x.unsqueeze(2).expand(batch_size, num_points, k, num_dims)  # (B, N, k, num_dims)
    feature = torch.cat((feature - x, x), dim=3).permute(0, 1, 3, 2)  # (B, N, 2 * num_dims, k)
    return feature


class LocalAggregation(nn.Module):
    def __init__(self, in_channels, out_channels, radius, nsample, local_aggregation_type='pospool', **kwargs):
        """LocalAggregation operators
        Args:
            in_channels: input channels.
            out_channels: output channels.
            radius: ball query radius
            nsample: neighborhood limit.
            config: config file
        """
        super(LocalAggregation, self).__init__()
        if local_aggregation_type == 'pospool':
            self.local_aggregation_operator = PosPool(in_channels, out_channels, radius, nsample, **kwargs)
        else:
            raise NotImplementedError(f'LocalAggregation {local_aggregation_type} not implemented')

    def forward(self, query_xyz, support_xyz, query_mask, support_mask, support_features):
        """
        Args:
            query_xyz: [B, N1, 3], query points.
            support_xyz: [B, N2, 3], support points.
            query_mask: [B, N1], mask for query points.
            support_mask: [B, N2], mask for support points.
            support_features: [B, C_in, N2], input features of support points.
        Returns:
           output features of query points: [B, C_out, 3]
        """
        return self.local_aggregation_operator(query_xyz, support_xyz, query_mask, support_mask, support_features)


class PosPool(nn.Module):
    def __init__(self, in_channels, out_channels, radius, nsample, position_embedding, agg_reduction, agg_output_conv,
                 agg_act_layer, agg_norm_layer, agg_standard_bn_params):
        """A PosPool operator for local aggregation
        Args:
            in_channels: input channels.
            out_channels: output channels.
            radius: ball query radius
            nsample: neighborhood limit.
            position_embedding:
            agg_reduction:
            agg_output_conv:
            agg_norm_layer:
            agg_standard_bn_params:
        """
        super(PosPool, self).__init__()
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.radius = radius
        self.nsample = nsample
        self.position_embedding = position_embedding
        self.reduction = agg_reduction
        self.output_conv = agg_output_conv or (self.in_channels != self.out_channels)

        self.grouper = MaskedQueryAndGroup(radius, nsample, use_xyz=False, ret_grouped_xyz=True, normalize_xyz=True)

        bn_args = dict()
        if not agg_standard_bn_params:
            bn_args = dict(eps=1e-3, momentum=0.01)

        if self.output_conv:
            self.out_conv = nn.Sequential(
                nn.Conv1d(in_channels, out_channels, kernel_size=1, bias=False),
                agg_norm_layer(out_channels, **bn_args),
                agg_act_layer(inplace=True))
        else:
            self.out_transform = nn.Sequential(
                agg_norm_layer(out_channels, **bn_args),
                agg_act_layer(inplace=True))

    def forward(self, query_xyz, support_xyz, query_mask, support_mask, support_features):
        """
        Args:
            query_xyz: [B, N1, 3], query points.
            support_xyz: [B, N2, 3], support points.
            query_mask: [B, N1], mask for query points.
            support_mask: [B, N2], mask for support points.
            support_features: [B, C_in, N2], input features of support points.
        Returns:
           output features of query points: [B, C_out, 3]
        """
        B = query_xyz.shape[0]
        C = support_features.shape[1]
        npoint = query_xyz.shape[1]
        neighborhood_features, relative_position, neighborhood_mask = self.grouper(query_xyz, support_xyz, query_mask,
                                                                                   support_mask, support_features)

        if self.position_embedding == 'xyz':
            position_embedding = torch.unsqueeze(relative_position, 1)
            aggregation_features = neighborhood_features.view(B, C // 3, 3, npoint, self.nsample)
            aggregation_features = position_embedding * aggregation_features  # (B, C//3, 3, npoint, nsample)
            aggregation_features = aggregation_features.view(B, C, npoint, self.nsample)  # (B, C, npoint, nsample)
        elif self.position_embedding == 'sin_cos':
            feat_dim = C // 6
            wave_length = 1000
            alpha = 100
            feat_range = torch.arange(feat_dim, dtype=torch.float32, device=query_xyz.device)  # (feat_dim, )
            dim_mat = torch.pow(1.0 * wave_length, (1.0 / feat_dim) * feat_range)  # (feat_dim, )
            position_mat = torch.unsqueeze(alpha * relative_position, -1)  # (B, 3, npoint, nsample, 1)
            div_mat = torch.div(position_mat, dim_mat)  # (B, 3, npoint, nsample, feat_dim)
            sin_mat = torch.sin(div_mat)  # (B, 3, npoint, nsample, feat_dim)
            cos_mat = torch.cos(div_mat)  # (B, 3, npoint, nsample, feat_dim)
            position_embedding = torch.cat([sin_mat, cos_mat], -1)  # (B, 3, npoint, nsample, 2*feat_dim)
            position_embedding = position_embedding.permute(0, 1, 4, 2, 3).contiguous()
            position_embedding = position_embedding.view(B, C, npoint, self.nsample)  # (B, C, npoint, nsample)
            aggregation_features = neighborhood_features * position_embedding  # (B, C, npoint, nsample)
        else:
            raise NotImplementedError(f'Position Embedding {self.position_embedding} not implemented in PosPool')

        if self.reduction == 'max':
            out_features = F.max_pool2d(
                aggregation_features, kernel_size=[1, self.nsample]
            )
            out_features = torch.squeeze(out_features, -1)
        elif self.reduction == 'avg' or self.reduction == 'mean':
            feature_mask = neighborhood_mask + (1 - query_mask[:, :, None])
            feature_mask = feature_mask[:, None, :, :]
            aggregation_features *= feature_mask
            out_features = aggregation_features.sum(-1)
            neighborhood_num = feature_mask.sum(-1)
            out_features /= neighborhood_num
        elif self.reduction == 'sum':
            feature_mask = neighborhood_mask + (1 - query_mask[:, :, None])
            feature_mask = feature_mask[:, None, :, :]
            aggregation_features *= feature_mask
            out_features = aggregation_features.sum(-1)
        else:
            raise NotImplementedError(f'Reduction {self.reduction} not implemented in PosPool ')

        if self.output_conv:
            out_features = self.out_conv(out_features)
        else:
            out_features = self.out_transform(out_features)

        return out_features
