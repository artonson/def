#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
@Author: Yue Wang
@Contact: yuewangx@mit.edu
@File: model.py
@Time: 2018/10/13 6:35 PM
"""

import os
import sys
import copy
import math
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from modules.base import ParameterizedModule
from models.transformer import MultiHeadAttention, ScaledDotProductAttention


def knn(x, k):
    inner = -2 * torch.matmul(x.transpose(2, 1), x)
    xx = torch.sum(x ** 2, dim=1, keepdim=True)
    pairwise_distance = -xx - inner - xx.transpose(2, 1)

    idx = pairwise_distance.topk(k=k, dim=-1)[1]  # (batch_size, num_points, k)
#    dist = torch.softmax(pairwise_distance, dim=2)
    return idx, pairwise_distance


def get_graph_feature(x, k=20, idx=None, device='cuda'):
    batch_size = x.size(0)
    num_points = x.size(2)
    x = x.view(batch_size, -1, num_points)
    if idx is None:
        idx, dist = knn(x, k=k)  # (batch_size, num_points, k)
    else:
        dist = None

    idx_base = torch.arange(0, batch_size, device=device).view(-1, 1, 1) * num_points

    idx = idx + idx_base

    idx = idx.view(-1)

    _, num_dims, _ = x.size()

    x = x.transpose(2,1).contiguous()  # (batch_size, num_points, num_dims)  -> (batch_size*num_points, num_dims) #   batch_size * num_points * k + range(0, batch_size*num_points)
    feature = x.view(batch_size * num_points, -1)[idx, :]
    feature = feature.view(batch_size, num_points, k, num_dims)

    x = x.view(batch_size, num_points, 1, num_dims).repeat(1, 1, k, 1)

    feature = torch.cat((feature - x, x), dim=3).permute(0, 3, 1, 2)

    return feature, dist


class DGCNN(ParameterizedModule):
    def __init__(self, **args):
        super(DGCNN, self).__init__()
        self.args = args
        self.k = args['k']
        if 'device' in args:
            self.device = args['device']
        else:
            self.device = 'cuda'
        print(self.device)
#        self.bn01 = nn.BatchNorm2d(64)
#        self.bn02 = nn.BatchNorm2d(64)
#        self.bn03 = nn.BatchNorm2d(64)
#        self.bn04 = nn.BatchNorm2d(64)
#        self.bn05 = nn.BatchNorm2d(64)
#        self.bn06 = nn.BatchNorm2d(64)

        self.bn1 = nn.BatchNorm2d(64)
        self.bn2 = nn.BatchNorm2d(64)
        self.bn3 = nn.BatchNorm2d(64)
        self.bn4 = nn.BatchNorm2d(64)
        self.bn5 = nn.BatchNorm2d(64)
        self.bn6 = nn.BatchNorm2d(64)
        self.bn7 = nn.BatchNorm2d(1024)
        self.bn8 = nn.BatchNorm2d(512)
        self.bn9 = nn.BatchNorm2d(256)

        self.conv01 = nn.Sequential(nn.Conv2d(3, 512, kernel_size=1, bias=False),
#                                    self.bn01,
                                    nn.ReLU())
        self.conv02 = nn.Sequential(nn.Conv2d(512, 1024, kernel_size=1, bias=False),
#                                    self.bn02,
                                    nn.ReLU())
        self.conv03 = nn.Sequential(nn.Conv2d(64, 512, kernel_size=1, bias=False),
#                                    self.bn03,
                                    nn.ReLU())
        self.conv04 = nn.Sequential(nn.Conv2d(512, 1024, kernel_size=1, bias=False),
#                                    self.bn04,
                                    nn.ReLU())
        self.conv05 = nn.Sequential(nn.Conv2d(64, 512, kernel_size=1, bias=False),
#                                    self.bn05,
                                    nn.ReLU())
        self.conv06 = nn.Sequential(nn.Conv2d(512, 1024, kernel_size=1, bias=False),
#                                    self.bn06,
                                    nn.ReLU())

        self.conv1 = nn.Sequential(nn.Conv2d(6, 64, kernel_size=1, bias=False),
                                   self.bn1,
                                   nn.ReLU())
        self.conv2 = nn.Sequential(nn.Conv2d(64, 64, kernel_size=1, bias=False),
                                   self.bn2,
                                   nn.ReLU())
        self.conv3 = nn.Sequential(nn.Conv2d(64 * 2, 64, kernel_size=1, bias=False),
                                   self.bn3,
                                   nn.ReLU())
        self.conv4 = nn.Sequential(nn.Conv2d(64, 64, kernel_size=1, bias=False),
                                   self.bn4,
                                   nn.ReLU())
        self.conv5 = nn.Sequential(nn.Conv2d(64 * 2, 64, kernel_size=1, bias=False),
                                   self.bn5,
                                   nn.ReLU())
        self.conv6 = nn.Sequential(nn.Conv2d(64, 64, kernel_size=1, bias=False),
                                   self.bn6,
                                   nn.ReLU())
        self.conv7 = nn.Sequential(nn.Conv2d(64 * 3, 1024, kernel_size=1, bias=False),
                                   self.bn7,
                                   nn.ReLU())
        self.conv8 = nn.Sequential(nn.Conv2d(1024 + 64 * 3, 512, kernel_size=1, bias=False),
                                   self.bn8,
                                   nn.ReLU())
        self.conv9 = nn.Sequential(nn.Conv2d(512, 256, kernel_size=1, bias=False),
                                   self.bn9,
                                   nn.ReLU())
        self.conv10 = nn.Conv2d(256, 1, kernel_size=1, bias=False)

        self.max_pooling = nn.MaxPool2d([1024, 1])

        self.dp1 = nn.Dropout(p=args['dropout'])
        #        self.attention1 = MultiHeadAttention(1,3,use_residual=False)
        self.attention1 = ScaledDotProductAttention(1, 0)

    def forward(self, x):
        points = x[:, :, :, None]
        _, dist1 = get_graph_feature(x, self.k, device=self.device)

        f = self.conv01(points)
        f = self.conv02(f).squeeze()
        f1 = torch.softmax(f, dim=2)
#
       # f = f.transpose(2, 1)
       # norm = 1e-6 + torch.norm(f, 2, 2, keepdim=True).repeat_interleave(64, dim=2)
       # f = f / norm
       #
       # f = f.squeeze()
       # x = points.transpose(2, 1).squeeze()
       # _, _, att1, dist_att1 = self.attention1(f, f, x, attn_mask=dist1)
       # f = f.transpose(2, 1)[:, :, :, None]
#
        idx = f1.topk(k=self.k, dim=-1)[1]
        x, dist1 = get_graph_feature(points, k=self.k, idx=idx, device=self.device)

        x = self.conv1(x)
        x = self.conv2(x)
        x1 = x.max(dim=-1, keepdim=True)[0]
#        x11 = torch.cat((x1, f), dim=1)

#        _, dist2 = get_graph_feature(x1, self.k)
        f = self.conv03(x1)
        f = self.conv04(f).squeeze()
        f2 = torch.softmax(f, dim=2)

#        f = f.transpose(2, 1)
#        norm = 1e-6 + torch.norm(f, 2, 2, keepdim=True).repeat_interleave(64, dim=2)
#        f = f / norm

#        f = f.squeeze()
#        x = x1.transpose(2, 1).squeeze()
#        _, _, att2, dist_att2 = self.attention1(f, f, x, attn_mask=dist2)
#        f = f.transpose(2, 1)[:, :, :, None]

        idx = f2.topk(k=self.k, dim=-1)[1]
        x, _ = get_graph_feature(x1, k=self.k, idx=idx, device=self.device)

        x = self.conv3(x)
        x = self.conv4(x)
        x2 = x.max(dim=-1, keepdim=True)[0]
#        x22 = torch.cat((x2, f), dim=1)

#        _, dist3 = get_graph_feature(x2, self.k)
        f = self.conv05(x2)
        f = self.conv06(f).squeeze()
        f3 = torch.softmax(f, dim=2)

#        f = f.transpose(2, 1)
#        norm = 1e-6 + torch.norm(f, 2, 2, keepdim=True).repeat_interleave(64, dim=2)
#        f = f / norm
#
#        f = f.squeeze()
#        x = x2.transpose(2, 1).squeeze()
#        _, _, att3, dist_att3 = self.attention1(f, f, x, attn_mask=dist3)
#        f = f.transpose(2, 1)[:, :, :, None]

        idx = f3.topk(k=self.k, dim=-1)[1]
        x, dist3 = get_graph_feature(x2, k=self.k, idx=idx, device=self.device)

        x = self.conv5(x)
        x = self.conv6(x)
        x3 = x.max(dim=-1, keepdim=True)[0]
#        x33 = torch.cat((x3, f), dim=1)

        x = torch.cat((x1, x2, x3), dim=1)
        x = self.conv7(x)
        x = self.max_pooling(x)

        expand = torch.repeat_interleave(x, 1024, 2)

        x = torch.cat((expand, x1, x2, x3), dim=1)
        x = self.conv8(x)
        x = self.conv9(x)

        x = self.dp1(x)

        x = self.conv10(x)
        x = torch.squeeze(x)

        return x, (f1, f2, f3)
