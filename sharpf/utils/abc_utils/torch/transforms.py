from abc import ABC, abstractmethod
from collections import Callable

import numpy as np
import torch

from .transformations import random_3d_rotation_matrix, random_scale_matrix


class AbstractTransform(ABC, Callable):
    @abstractmethod
    def __call__(self, data, target):
        pass


class Random3DRotation(AbstractTransform):
    def __call__(self, data, target):
        data = torch.cat((data, torch.ones(len(data), 1)), dim=1)
        transform = random_3d_rotation_matrix()
        data = torch.matmul(data, transform)
        return data[:, :-1], target


class RandomScale(AbstractTransform):
    def __init__(self, min_scale, max_scale):
        self.scale_range = min_scale, max_scale

    def __call__(self, data, target):
        data = torch.cat((data, torch.ones(len(data), 1)), dim=1)
        transform = random_scale_matrix(*self.scale_range)
        data = torch.matmul(data, transform)
        target *= transform[0, 0].item()
        return data[:, :-1], target


class Center(AbstractTransform):
    def __call__(self, data, target):
        # center: E p_i = 0
        data -= data.mean(axis=0, keepdim=True)
        return data, target


class NormalizeL2(AbstractTransform):
    def __call__(self, data, target):
        # scale: sup_i ||p_i|| = 1
        data_scale = data.norm(dim=1).max(dim=0).values
        data /= data_scale
        # target /= data_scale
        return data, target


class CompositeTransform(AbstractTransform):
    def __init__(self, transforms):
        self.transforms = transforms

    def __call__(self, data, target):
        for t in self.transforms:
            data, target = t(data, target)
        return data, target


class TypeCast(AbstractTransform):
    def __init__(self, data_type=torch.float32, target_type=torch.float32):
        self.data_type = data_type
        self.target_type = target_type

    def __call__(self, data, target):
        return data.type(self.data_type), \
               target.type(self.target_type)


class RandomSubsamplePoints(AbstractTransform):
    def __init__(self, n_points, theta=1.0):
        self.n_points = n_points
        self.theta = theta

    def __call__(self, data, target):
        n_points = len(data)
        p = torch.exp(-target * self.theta)
        p /= torch.sum(p)
        i_subset = np.random.choice(
            np.arange(n_points), size=self.n_points, replace=False, p=p.numpy())
        return data[i_subset], target[i_subset]


class QuantileNormalize(AbstractTransform):

    def __init__(self, quantile):
        self.quantile = quantile

    def __call__(self, data, target):
        # mask -> min shift -> quantile

        norm_data = np.copy(data)
        mask_obj = np.where(norm_data != 0)
        mask_back = np.where(norm_data == 0)
        norm_data[mask_back] = norm_data.max() + 1.0  # new line
        norm_data -= norm_data[mask_obj].min()

        norm_data /= self.quantile

        return norm_data, target


class HighResQuantileNormalize(QuantileNormalize):

    def __init__(self):
        super().__init__(7.4776)


class Standardize(AbstractTransform):

    def __call__(self, data, target):
        # zero mean, unit variance

        standart_data = np.copy(data)
        standart_data -= np.mean(standart_data)
        std = np.linalg.norm(standart_data, axis=1).max()
        if std > 0:
            standart_data /= std

        return standart_data, target
