from abc import ABC, abstractmethod
from collections import Callable

import torch

from .transformations import random_3d_rotation_matrix


class AbstractTransform(ABC, Callable):

    def __init__(self, keys):
        self.keys = keys

    @abstractmethod
    def __call__(self, item):
        pass


class Random3DRotation(AbstractTransform):
    def __call__(self, item):
        transform = random_3d_rotation_matrix()
        for key in self.keys:
            assert item[key].ndim == 2 and len(item[key][1]) == 3, item[key].size()
            item[key] = torch.cat((item[key], torch.ones(len(item[key]), 1)), dim=1)
            item[key] = torch.matmul(item[key], transform)[:, :-1]
        return item


class RandomScale(AbstractTransform):
    def __init__(self, keys, min_scale, max_scale):
        super().__init__(keys)
        self.min_scale = min_scale
        self.max_scale = max_scale

    def __call__(self, item):
        scale_value = self.min_scale + torch.rand(1).item() * (self.max_scale - self.min_scale)
        for key in self.keys:
            item[key] *= scale_value
        return item


class Center(AbstractTransform):

    def __init__(self, keys, dim):
        super().__init__(keys)
        self.dim = dim

    def __call__(self, item):
        for key in self.keys:
            item[key] -= item[key].mean(axis=self.dim, keepdim=True)
        return item


class NormalizeL2(AbstractTransform):

    def __init__(self, keys, dim):
        super().__init__(keys)
        self.dim = dim

    def __call__(self, item):
        for key in self.keys:
            scale = item[key].norm(dim=self.dim).max()
            assert scale.item() > 0
            item[key] /= scale
        return item


class CompositeTransform(AbstractTransform):
    def __init__(self, transforms):
        super().__init__(None)
        self.transforms = transforms

    def __call__(self, item):
        for t in self.transforms:
            item = t(item)
        return item


class ToTensor(AbstractTransform):
    def __init__(self, keys, type):
        super().__init__(keys)
        self.type = type

    def __call__(self, item):
        for key in self.keys:
            item[key] = torch.from_numpy(item[key]).type(self.type)
        return item


class PreprocessDepth(AbstractTransform):

    def __init__(self, quantile):
        super().__init__(None)
        self.quantile = quantile

    def __call__(self, item):
        item['background_mask'] = (item['image'] == 0)
        item['image'] = torch.where(item['background_mask'], item['image'].max() + 1.0, item['image'])
        item['image'] -= item['image'].min()
        item['image'] /= self.quantile
        item['image'].unsqueeze_(0)
        return item


class PreprocessDistances(AbstractTransform):

    def __init__(self):
        super().__init__(None)

    def __call__(self, item):
        item['distances'] = torch.where(item['background_mask'], torch.ones_like(item['distances']), item['distances'])
        return item


class ComputeCloseToSharpMask(AbstractTransform):

    def __init__(self):
        super().__init__(None)

    def __call__(self, item):
        item['close_to_sharp_mask'] = (item['distances'] < 1.0).float()
        item['close_to_sharp_mask'] = torch.where(item['background_mask'],
                                                  torch.zeros_like(item['close_to_sharp_mask']),
                                                  item['close_to_sharp_mask'])
        item['close_to_sharp_mask'].unsqueeze_(0)
        return item


class DeleteKeys(AbstractTransform):

    def __call__(self, item):
        for key in self.keys:
            del item[key]
        return item


class RenameKeys(AbstractTransform):

    def __init__(self, old_keys, new_keys):
        super().__init__(None)
        assert len(old_keys) == len(new_keys)
        self.old_keys = old_keys
        self.new_keys = new_keys

    def __call__(self, item):
        for old_key, new_key in zip(self.old_keys, self.new_keys):
            assert new_key not in item
            item[new_key] = item[old_key]
        return item

# class RandomSubsamplePoints(AbstractTransform):
#     def __init__(self, n_points, theta=1.0):
#         self.n_points = n_points
#         self.theta = theta
#
#     def __call__(self, data, target):
#         n_points = len(data)
#         p = torch.exp(-target * self.theta)
#         p /= torch.sum(p)
#         i_subset = np.random.choice(
#             np.arange(n_points), size=self.n_points, replace=False, p=p.numpy())
#         return data[i_subset], target[i_subset]
