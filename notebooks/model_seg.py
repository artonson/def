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
    device = x.device

    idx_base = torch.arange(0, batch_size, device=device).view(-1, 1, 1)*num_points

    idx = idx + idx_base

    idx = idx.view(-1)
 
    _, num_dims, _ = x.size()

    x = x.transpose(2, 1).contiguous()   # (batch_size, num_points, num_dims)  -> (batch_size*num_points, num_dims) #   batch_size * num_points * k + range(0, batch_size*num_points)
    feature = x.view(batch_size*num_points, -1)[idx, :]
    feature = feature.view(batch_size, num_points, k, num_dims) 

    x = x.view(batch_size, num_points, 1, num_dims).repeat(1, 1, k, 1)
    
    feature = torch.cat((feature-x, x), dim=3).permute(0, 3, 1, 2)
  
    return feature, dist


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


class DGCNN(nn.Module):
    def __init__(self, args, output_channels=40):
        super(DGCNN, self).__init__()
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
        x, _ = get_graph_feature(x, k=self.k)
        x = self.conv1(x)
        x = self.conv2(x)
        
        x1 = x.max(dim=-1, keepdim=True)[0]

        x, _ = get_graph_feature(x1, k=self.k)
        x = self.conv3(x)
        x = self.conv4(x)
        x2 = x.max(dim=-1, keepdim=True)[0]

        x, _ = get_graph_feature(x2, k=self.k)
        x = self.conv5(x)
        x = self.conv6(x)
        x3 = x.max(dim=-1, keepdim=True)[0]

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

        return x