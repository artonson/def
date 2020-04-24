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

from sharpf.modules.base import ParameterizedModule


def knn(x, k):
    inner = -2*torch.matmul(x.transpose(2, 1), x)
    xx = torch.sum(x**2, dim=1, keepdim=True)
    pairwise_distance = -xx - inner - xx.transpose(2, 1)
 
    idx = pairwise_distance.topk(k=k, dim=-1)[1]   # (batch_size, num_points, k)
    return idx, pairwise_distance


def get_graph_feature(x, k=20, idx=None):
    batch_size = x.size(0)
    num_points = x.size(2)
    x = x.view(batch_size, -1, num_points)
    if idx is None:
        idx, dist = knn(x, k=k)   # (batch_size, num_points, k)
    device = torch.device('cuda')

    idx_base = torch.arange(0, batch_size, device=device).view(-1, 1, 1)*num_points

    idx = idx + idx_base

    idx = idx.view(-1)
 
    _, num_dims, _ = x.size()

    x = x.transpose(2, 1).contiguous()   # (batch_size, num_points, num_dims)  -> (batch_size*num_points, num_dims) #   batch_size * num_points * k + range(0, batch_size*num_points)
    feature = x.view(batch_size*num_points, -1)[idx, :]
    feature = feature.view(batch_size, num_points, k, num_dims) 

    x = x.view(batch_size, num_points, 1, num_dims).repeat(1, 1, k, 1)
    
    feature = torch.cat((feature-x, x), dim=3).permute(0, 3, 1, 2)
  
    return feature


#class PointNet(nn.Module):
#    def __init__(self, args, output_channels=40):
#        super(PointNet, self).__init__()
#        self.args = args
#        self.conv1 = nn.Conv1d(3, 64, kernel_size=1, bias=False)
#        self.conv2 = nn.Conv1d(64, 64, kernel_size=1, bias=False)
#        self.conv3 = nn.Conv1d(64, 64, kernel_size=1, bias=False)
#        self.conv4 = nn.Conv1d(64, 128, kernel_size=1, bias=False)
#        self.conv5 = nn.Conv1d(128, args.emb_dims, kernel_size=1, bias=False)
#        self.bn1 = nn.BatchNorm1d(64)
#        self.bn2 = nn.BatchNorm1d(64)
#        self.bn3 = nn.BatchNorm1d(64)
#        self.bn4 = nn.BatchNorm1d(128)
#        self.bn5 = nn.BatchNorm1d(args.emb_dims)
#        self.linear1 = nn.Linear(args.emb_dims, 512, bias=False)
#        self.bn6 = nn.BatchNorm1d(512)
#        self.dp1 = nn.Dropout()
#        self.linear2 = nn.Linear(512, output_channels)
#
#    def forward(self, x):
#        x = F.relu(self.bn1(self.conv1(x)))
#        x = F.relu(self.bn2(self.conv2(x)))
#        x = F.relu(self.bn3(self.conv3(x)))
#        x = F.relu(self.bn4(self.conv4(x)))
#        x = F.relu(self.bn5(self.conv5(x)))
#        x = F.adaptive_max_pool1d(x, 1).squeeze()
#        x = F.relu(self.bn6(self.linear1(x)))
#        x = self.dp1(x)
#        x = self.linear2(x)
#        return x


class DGCNN_SEMSEG(ParameterizedModule):
    def __init__(self, args, output_channels=40):
        super(DGCNN_SEMSEG, self).__init__()
        self.args = args
        self.k = args.k
        
        self.bn1 = nn.BatchNorm2d(64)
        self.bn2 = nn.BatchNorm2d(64)
        self.bn3 = nn.BatchNorm2d(64)
        self.bn4 = nn.BatchNorm2d(64)
        self.bn5 = nn.BatchNorm2d(64)
        self.bn6 = nn.BatchNorm2d(64)
        self.bn7 = nn.BatchNorm2d(1024)
        self.bn8 = nn.BatchNorm2d(512)
        self.bn9 = nn.BatchNorm2d(256)

        self.conv1 = nn.Sequential(nn.Conv2d(6, 64, kernel_size=1, bias=False),
                                   self.bn1,
                                   nn.ReLU())
        self.conv2 = nn.Sequential(nn.Conv2d(64, 64, kernel_size=1, bias=False),
                                   self.bn2,
                                   nn.ReLU())
        self.conv3 = nn.Sequential(nn.Conv2d(64*2, 64, kernel_size=1, bias=False),
                                   self.bn3,
                                   nn.ReLU())
        self.conv4 = nn.Sequential(nn.Conv2d(64, 64, kernel_size=1, bias=False),
                                   self.bn4,
                                   nn.ReLU())
        self.conv5 = nn.Sequential(nn.Conv2d(64*2, 64, kernel_size=1, bias=False),
                                   self.bn5,
                                   nn.ReLU())
        self.conv6 = nn.Sequential(nn.Conv2d(64, 64, kernel_size=1, bias=False),
                                   self.bn6,
                                   nn.ReLU())
        self.conv7 = nn.Sequential(nn.Conv2d(64*3, 1024, kernel_size=1, bias=False),
                                   self.bn7,
                                   nn.ReLU())
        self.conv8 = nn.Sequential(nn.Conv2d(1024+64*3, 512, kernel_size=1, bias=False),
                                   self.bn8,
                                   nn.ReLU())
        self.conv9 = nn.Sequential(nn.Conv2d(512, 256, kernel_size=1, bias=False),
                                   self.bn9,
                                   nn.ReLU())
        self.conv10 = nn.Conv2d(256, 1, kernel_size=1, bias=False)
        
        self.max_pooling = nn.MaxPool2d([1024,1])

        self.dp1 = nn.Dropout(p=args.dropout)

    def forward(self, x):
        batch_size = x.size(0)
        x, dist1 = get_graph_feature(x, k=self.k)
        x = self.conv1(x)
        x = self.conv2(x)
        x1 = x.max(dim=-1, keepdim=True)[0]

        x, dist2 = get_graph_feature(x1, k=self.k)
        x = self.conv3(x)
        x = self.conv4(x)
        x2 = x.max(dim=-1, keepdim=True)[0]

        x, dist3 = get_graph_feature(x2, k=self.k)
        x = self.conv5(x)
        x = self.conv6(x)
        x3 = x.max(dim=-1, keepdim=True)[0]

#        x = get_graph_feature(x3, k=self.k)
#        x = self.conv4(x)
#        x4 = x.max(dim=-1, keepdim=False)[0]

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

        return x, (dist1, dist2, dist3)



class DGCNN_CLS(ParameterizedModule):
    def __init__(self, output_channels=40, k=30, dropout=0.1,
                 emb_dims=128, **kwargs):
        super(DGCNN_CLS, self).__init__()
        self.k = k
        self.dropout = dropout
        self.emb_dims = emb_dims

        self.conv1 = nn.Sequential(nn.Conv2d(6, 64, kernel_size=1, bias=False),
                                   nn.LeakyReLU(negative_slope=0.2))
        self.conv2 = nn.Sequential(nn.Conv2d(64 * 2, 64, kernel_size=1, bias=False),
                                   nn.LeakyReLU(negative_slope=0.2))
        self.conv3 = nn.Sequential(nn.Conv2d(64 * 2, 128, kernel_size=1, bias=False),
                                   nn.LeakyReLU(negative_slope=0.2))
        self.conv4 = nn.Sequential(nn.Conv2d(128 * 2, 256, kernel_size=1, bias=False),
                                   nn.LeakyReLU(negative_slope=0.2))
        self.conv5 = nn.Sequential(nn.Conv1d(512, self.emb_dims, kernel_size=1, bias=False),
                                   nn.LeakyReLU(negative_slope=0.2))
        self.linear1 = nn.Linear(self.emb_dims * 2, 512, bias=False)
        self.dp1 = nn.Dropout(p=self.dropout)
        self.linear2 = nn.Linear(512, 256)
        self.dp2 = nn.Dropout(p=self.dropout)
        self.linear3 = nn.Linear(256, output_channels)

    def forward(self, x):
        x = x.transpose(1, 2)
        batch_size = x.size(0)
        x = get_graph_feature(x, k=self.k)
        x = self.conv1(x)
        x1 = x.max(dim=-1, keepdim=False)[0]

        x = get_graph_feature(x1, k=self.k)
        x = self.conv2(x)
        x2 = x.max(dim=-1, keepdim=False)[0]

        x = get_graph_feature(x2, k=self.k)
        x = self.conv3(x)
        x3 = x.max(dim=-1, keepdim=False)[0]

        x = get_graph_feature(x3, k=self.k)
        x = self.conv4(x)
        x4 = x.max(dim=-1, keepdim=False)[0]

        x = torch.cat((x1, x2, x3, x4), dim=1)

        x = self.conv5(x)
        x1 = F.adaptive_max_pool1d(x, 1).view(batch_size, -1)
        x2 = F.adaptive_avg_pool1d(x, 1).view(batch_size, -1)
        x = torch.cat((x1, x2), 1)

        x = F.leaky_relu(self.linear1(x), negative_slope=0.2)
        x = self.dp1(x)
        x = F.leaky_relu(self.linear2(x), negative_slope=0.2)
        x = self.dp2(x)
        x = self.linear3(x)
        return x.squeeze()
