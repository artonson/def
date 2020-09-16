import math

import torch.nn as nn

from .local_modules import LocalAggregation
from .pt_utils import MaskedAvgPoolStride, MaskedMaxPoolStride

__all__ = ['Bottleneck', 'PointResNet']


class MultiInputSequential(nn.Sequential):

    def forward(self, *input):
        for module in self._modules.values():
            input = module(*input)
        return input


class DownsampleAvg1d(nn.Module):

    def __init__(self, in_channels, out_channels, radius, nsample, sampleDl, stride=1, norm_layer=None):
        super(DownsampleAvg1d, self).__init__()
        norm_layer = norm_layer or nn.BatchNorm1d
        if stride > 1:
            self.pool = MaskedAvgPoolStride(stride, radius, nsample, sampleDl)
        else:
            self.pool = None
        self.conv = nn.Conv1d(in_channels, out_channels, 1, stride=1, padding=0, bias=False)
        self.bn = norm_layer(out_channels)

    def forward(self, xyz, mask, x):
        if self.pool is not None:
            xyz, mask, x = self.pool(xyz, mask, x)
        x = self.conv(x)
        x = self.bn(x)
        return xyz, mask, x


class DownsampleMax1d(nn.Module):

    def __init__(self, in_channels, out_channels, radius, nsample, sampleDl, stride=1, norm_layer=None):
        super(DownsampleMax1d, self).__init__()
        norm_layer = norm_layer or nn.BatchNorm1d
        if stride > 1:
            self.pool = MaskedMaxPoolStride(stride, radius, nsample, sampleDl)
        else:
            self.pool = None
        self.conv = nn.Conv1d(in_channels, out_channels, 1, stride=1, padding=0, bias=False)
        self.bn = norm_layer(out_channels)

    def forward(self, xyz, mask, x):
        if self.pool is not None:
            xyz, mask, x = self.pool(xyz, mask, x)
        x = self.conv(x)
        x = self.bn(x)
        return xyz, mask, x


class Projection1d(nn.Module):

    def __init__(self, in_channels, out_channels, norm_layer=None):
        super(Projection1d, self).__init__()
        norm_layer = norm_layer or nn.BatchNorm1d
        self.conv = nn.Conv1d(in_channels, out_channels, 1, stride=1, padding=0, bias=False)
        self.bn = norm_layer(out_channels)

    def forward(self, xyz, mask, x):
        x = self.conv(x)
        x = self.bn(x)
        return xyz, mask, x


class Bottleneck(nn.Module):
    expansion = 4

    def __init__(self, inplanes, planes, radius, nsample, downsample=None, local_aggregation_type='pospool',
                 cardinality=1, base_width=64, reduce_first=1, act_layer=nn.ReLU, norm_layer=nn.BatchNorm1d,
                 **aggregation_kwargs):
        super(Bottleneck, self).__init__()

        width = int(math.floor(planes * (base_width / 64)) * cardinality)
        first_planes = width // reduce_first
        outplanes = planes * self.expansion

        self.conv1 = nn.Conv1d(inplanes, first_planes, kernel_size=1, bias=False)
        self.bn1 = norm_layer(first_planes)
        self.act1 = act_layer(inplace=True)

        conv_kwargs = dict(groups=cardinality)
        # todo add conv kwargs to agg kwargs
        self.local_aggregation = LocalAggregation(first_planes, width, radius, nsample, local_aggregation_type,
                                                  **aggregation_kwargs)

        self.conv3 = nn.Conv1d(width, outplanes, kernel_size=1, bias=False)
        self.bn3 = norm_layer(outplanes)
        self.act3 = act_layer(inplace=True)
        self.downsample = downsample

    def zero_init_last_bn(self):
        nn.init.zeros_(self.bn3.weight)

    def forward(self, xyz, mask, x):
        if self.downsample is not None:
            query_xyz, query_mask, residual = self.downsample(xyz, mask, x)
        else:
            query_xyz, query_mask, residual = xyz, mask, x

        x = self.conv1(x)
        x = self.bn1(x)
        x = self.act1(x)

        x = self.local_aggregation(query_xyz, xyz, query_mask, mask, x)

        x = self.conv3(x)
        x = self.bn3(x)

        x += residual
        x = self.act3(x)

        return query_xyz, query_mask, x


class Bottleneck2(Bottleneck):
    expansion = 2


def make_blocks(block_fn, strides, radiuses_f, radiuses_s, sampleDls, nsamples_f, nsamples_s, channels, block_repeats,
                inplanes, first_pool_stride, norm_layer=nn.BatchNorm1d,
                reduce_first=1, output_stride=32, type_downsample='avg', local_aggregation_type='pospool',
                **aggregation_kwargs):
    stages = []
    feature_info = []
    net_stride = first_pool_stride
    for stage_idx, (stride, radius_f, radius_s, sampleDl, nsample_f, nsample_s, planes, num_blocks) in enumerate(
            zip(strides, radiuses_f, radiuses_s, sampleDls, nsamples_f, nsamples_s, channels, block_repeats)):
        stage_name = f'layer{stage_idx + 1}'
        if net_stride >= output_stride:
            raise ValueError
        else:
            net_stride *= stride

        downsample = None
        if stride != 1 or inplanes != planes * block_fn.expansion:
            down_kwargs = dict(
                in_channels=inplanes, out_channels=planes * block_fn.expansion, radius=radius_f,
                nsample=nsample_f,
                sampleDl=sampleDl, stride=stride, norm_layer=norm_layer)
            if type_downsample == 'avg':
                downsample = DownsampleAvg1d(**down_kwargs)
            elif type_downsample == 'max':
                downsample = DownsampleMax1d(**down_kwargs)
            else:
                raise ValueError

        block_kwargs = dict(reduce_first=reduce_first, local_aggregation_type=local_aggregation_type,
                            **aggregation_kwargs)
        blocks = []
        for block_idx in range(num_blocks):
            downsample = downsample if block_idx == 0 else None
            nsample = nsample_f if block_idx == 0 else nsample_s
            radius = radius_f if block_idx == 0 else radius_s
            blocks.append(block_fn(inplanes, planes, radius, nsample, downsample, **block_kwargs))
            inplanes = planes * block_fn.expansion

        stages.append((stage_name, MultiInputSequential(*blocks)))
        feature_info.append(dict(num_chs=inplanes, reduction=net_stride, module=stage_name))

    return stages, feature_info


class PointResNet(nn.Module):
    def __init__(self,
                 block,
                 layers,
                 in_chans,
                 inplanes,
                 first_radius,
                 first_nsample,
                 first_pool,
                 nsamples_f,
                 nsamples_s,
                 channels,
                 strides,
                 radiuses_f,
                 radiuses_s,
                 sampleDls,
                 first_pool_stride=1,
                 first_sampleDl=None,
                 cardinality=1,
                 base_width=64,
                 stem_type='',
                 block_reduce_first=1,
                 type_downsample='avg',
                 output_stride=32,
                 act_layer=nn.ReLU,
                 norm_layer=nn.BatchNorm1d,
                 zero_init_last_bn=True,
                 remove_first_bn_act=False,
                 local_aggregation_type='pospool',
                 **aggregation_kwargs
                 ):
        # print(aggregation_kwargs)
        super(PointResNet, self).__init__()
        assert stem_type in ['', 'btnk']
        assert len(nsamples_s) == len(nsamples_f) == len(channels) == len(layers) == len(strides) == len(
            radiuses_f) == len(radiuses_s) == len(sampleDls) == 4, 'general hooks are not added yet'

        # Stem
        if stem_type == '':
            self.conv1 = nn.Conv1d(in_chans, inplanes, kernel_size=1, bias=False)
            self.bn1 = norm_layer(inplanes) if not remove_first_bn_act else None
            self.act1 = act_layer(inplace=True) if not remove_first_bn_act else None
            self.la1 = LocalAggregation(inplanes, inplanes, first_radius, first_nsample,
                                        local_aggregation_type, **aggregation_kwargs)
            self.btnk1 = None
            self.feature_info = [dict(num_chs=inplanes, reduction=1, module='la1')]
        elif stem_type == 'btnk':
            self.conv1 = nn.Conv1d(in_chans, inplanes // 2, kernel_size=1, bias=False)
            self.bn1 = norm_layer(inplanes // 2) if not remove_first_bn_act else None
            self.act1 = act_layer(inplace=True) if not remove_first_bn_act else None
            self.la1 = LocalAggregation(inplanes // 2, inplanes // 2, first_radius, first_nsample,
                                        local_aggregation_type, **aggregation_kwargs)
            self.btnk1 = block(
                inplanes // 2, inplanes // 2, first_radius, first_nsample,
                downsample=Projection1d(inplanes // 2, inplanes // 2 * block.expansion, norm_layer),
                local_aggregation_type=local_aggregation_type, cardinality=cardinality,
                base_width=base_width, reduce_first=1, act_layer=act_layer, norm_layer=norm_layer,
                **aggregation_kwargs)
            self.feature_info = [dict(num_chs=inplanes, reduction=1, module='btnk1')]
        else:
            raise ValueError

        # Stem Pooling
        if first_pool is None:
            assert first_pool_stride == 1
            self.pool = None
        elif first_pool == 'max':
            assert first_pool_stride > 1 and first_sampleDl is not None
            self.pool = MaskedMaxPoolStride(first_pool_stride, first_radius, first_nsample, first_sampleDl)
        elif first_pool == 'avg':
            assert first_pool_stride > 1 and first_sampleDl is not None
            self.pool = MaskedAvgPoolStride(first_pool_stride, first_radius, first_nsample, first_sampleDl)
        else:
            raise ValueError

        # Feature Blocks
        stage_modules, stage_feature_info = make_blocks(
            block, strides, radiuses_f, radiuses_s, sampleDls, nsamples_f, nsamples_s, channels, layers, inplanes,
            first_pool_stride,
            cardinality=cardinality,
            base_width=base_width,
            output_stride=output_stride, reduce_first=block_reduce_first, type_downsample=type_downsample,
            act_layer=act_layer, norm_layer=norm_layer, **aggregation_kwargs)
        for stage in stage_modules:
            self.add_module(*stage)  # layer1, layer2, etc
        self.feature_info.extend(stage_feature_info)

        for n, m in self.named_modules():
            if isinstance(m, nn.Conv1d):
                if act_layer == nn.LeakyReLU:
                    nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='leaky_relu')
                elif act_layer == nn.ReLU:
                    nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
                else:
                    raise NotImplementedError
            elif isinstance(m, nn.BatchNorm1d):
                nn.init.constant_(m.weight, 1.)
                nn.init.constant_(m.bias, 0.)
        if zero_init_last_bn:
            for m in self.modules():
                if hasattr(m, 'zero_init_last_bn'):
                    m.zero_init_last_bn()

    def forward(self, xyz, mask, features):
        """
        Args:
            xyz: (B, N, 3), point coordinates
            mask: (B, N), 0/1 mask to distinguish padding points
            features: (B, 3, input_features_dim), input points features
            end_points: a dict

        Returns:
            end_points: a dict contains all outputs
        """
        end_points = {}

        # res1
        features = self.conv1(features)
        if self.bn1 is not None:
            features = self.bn1(features)
        if self.act1 is not None:
            features = self.act1(features)
        features = self.la1(xyz, xyz, mask, mask, features)
        if self.btnk1 is not None:
            xyz, mask, features = self.btnk1(xyz, mask, features)
        end_points['res1_xyz'] = xyz
        end_points['res1_mask'] = mask
        end_points['res1_features'] = features

        if self.pool is not None:
            xyz, mask, features = self.pool(xyz, mask, features)

        # res2
        xyz, mask, features = self.layer1(xyz, mask, features)
        end_points['res2_xyz'] = xyz
        end_points['res2_mask'] = mask
        end_points['res2_features'] = features

        # res3
        xyz, mask, features = self.layer2(xyz, mask, features)
        end_points['res3_xyz'] = xyz
        end_points['res3_mask'] = mask
        end_points['res3_features'] = features

        # res4
        xyz, mask, features = self.layer3(xyz, mask, features)
        end_points['res4_xyz'] = xyz
        end_points['res4_mask'] = mask
        end_points['res4_features'] = features

        # res5
        xyz, mask, features = self.layer4(xyz, mask, features)
        end_points['res5_xyz'] = xyz
        end_points['res5_mask'] = mask
        end_points['res5_features'] = features

        return end_points
