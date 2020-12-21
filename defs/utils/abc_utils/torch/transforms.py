from abc import ABC, abstractmethod
from collections import Callable

import torch
import numpy as np
import torch.nn.functional as F

from .transformations import random_3d_rotation_matrix, image_to_points


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
            if key == 'points':
                item['original_points'] = item['points']
        return item


class NormalizeByMaxL2(AbstractTransform):

    def __init__(self, keys, dim):
        super().__init__(keys)
        self.dim = dim

    def __call__(self, item):
        for key in self.keys:
            try:
                scale = item[key].norm(dim=self.dim).max()
            except IndexError:
                print()
                print(key, item[key].shape)
                raise IndexError
            assert scale.item() > 0
            item[key] /= scale
        return item


class NormalizeL2(AbstractTransform):

    def __init__(self, keys, dim):
        super().__init__(keys)
        self.dim = dim

    def __call__(self, item):
        for key in self.keys:
            item[key] = F.normalize(item[key], dim=self.dim)
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
            if key in item:
                if key == 'points':
                    item['points'] = item['points'].reshape((-1, 3))
                # if key == 'voronoi':
                #     item['voronoi'] = item['voronoi'].reshape((-1, 1))
                try:
                    item[key] = torch.from_numpy(item[key]).type(self.type)
                except TypeError:
                    print(key, item[key].dtype, self.type)
                    print(item[key])
                    raise TypeError
        return item


class PreprocessDepth(AbstractTransform):

    def __init__(self, quantile):
        super().__init__(None)
        self.quantile = quantile

    def __call__(self, item):
        if 'voronoi' in item:
            item['voronoi'] = item['voronoi'].reshape(64, 64)
        if 'sharpness' in item:
            item['sharpness'] = item['sharpness'].reshape(64, 64)
        if 'sharpness_seg' in item:
            item['sharpness_seg'] = item['sharpness_seg'].reshape(64, 64)
        if 'ecnet' in item:
            item['ecnet'] = item['ecnet'].reshape(64, 64)
        if 'normals' in item:
            item['normals'] = item['normals'].permute(2, 0, 1).contiguous()
        if 'directions' in item:
            item['directions'] = item['directions'].permute(2, 0, 1).contiguous()
        item['image'] -= item['image'].min()
        item['original_points'] = item['image']
        item['image'] /= self.quantile
        item['image'].unsqueeze_(0)
        return item


class PreprocessArbitraryDepth(AbstractTransform):

    def __init__(self, quantile):
        super().__init__(None)
        self.quantile = quantile

    def __call__(self, item):
        if 'voronoi' in item:
            item['voronoi'] = item['voronoi'].reshape(64, 64)
        if 'normals' in item:
            item['normals'] = item['normals'].permute(2, 0, 1).contiguous()
        if 'directions' in item:
            item['directions'] = item['directions'].permute(2, 0, 1).contiguous()
        item['background_mask'] = (item['image'] == 0.0)
        item['image'] = torch.where(item['background_mask'], item['image'],
                                    item['image'] - torch.masked_select(item['image'], ~item['background_mask']).min())
        item['image'] /= self.quantile
        item['image'].unsqueeze_(0)
        return item


class PreprocessArbitrarySLDepth(AbstractTransform):

    def __init__(self):
        super().__init__(None)

    def __call__(self, item):
        if 'voronoi' in item:
            item['voronoi'] = item['voronoi'].reshape(64, 64)
        if 'normals' in item:
            item['normals'] = item['normals'].permute(2, 0, 1).contiguous()
        if 'directions' in item:
            item['directions'] = item['directions'].permute(2, 0, 1).contiguous()
        item['background_mask'] = (item['image'] == 0.0)

        # mark: take into account in which direction the z axis is directed
        # the model assumes that z axis is directed from us
        item['image'] = -item['image']

        item['image'] = torch.where(item['background_mask'], item['image'],
                                    item['image'] - torch.masked_select(item['image'], ~item['background_mask']).min())
        item['image'] /= np.quantile(item['image'].numpy(), 0.95)
        item['image'].unsqueeze_(0)
        return item


class PreprocessDistances(AbstractTransform):

    def __init__(self):
        super().__init__(None)

    def __call__(self, item):
        item['distances'] = torch.where(item['background_mask'], torch.ones_like(item['distances']), item['distances'])
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


class Concatenate(AbstractTransform):

    def __init__(self, in_keys, out_key, dim):
        super().__init__(None)
        self.in_keys = in_keys
        self.out_key = out_key
        self.dim = dim

    def __call__(self, item):
        if 'voronoi' in self.in_keys:
            if item['voronoi'].ndim == 2:
                item['voronoi'] = item['voronoi'].unsqueeze(0)
            elif item['voronoi'].ndim == 1:
                item['voronoi'] = item['voronoi'].unsqueeze(1)
        item[self.out_key] = torch.cat([item[in_key] for in_key in self.in_keys], dim=self.dim)
        return item


class DepthToPointCloud(AbstractTransform):

    def __init__(self, resolution):
        super().__init__(None)
        self.resolution = resolution

    def __call__(self, item):
        item['image'] = image_to_points(item['image'], self.resolution)
        item['distances'] = item['distances'].reshape(-1)
        if 'voronoi' in item:
            item['voronoi'] = item['voronoi'].reshape(-1)
        if 'normals' in item:
            item['normals'] = item['normals'].reshape(-1, 3)
        if 'directions' in item:
            item['directions'] = item['directions'].reshape(-1, 3)
        return item


class Flatten(AbstractTransform):

    def __init__(self, keys, start_dims):
        assert len(keys) == len(start_dims)
        super().__init__(keys)
        self.start_dims = start_dims

    def __call__(self, item):
        for key, start_dim in zip(self.keys, self.start_dims):
            item[key] = torch.flatten(item[key], start_dim=start_dim)
        return item


class ComputeIsFlatProperty(AbstractTransform):

    def __init__(self):
        super().__init__(None)

    def __call__(self, item):
        if 'distances' in item:
            item['is_flat'] = (item['distances'] >= 1.0).all()
        return item


class ComputeTargetSharp(AbstractTransform):

    def __init__(self, resolution):
        super().__init__(None)
        self.resolution = resolution

    def __call__(self, item):
        item['target_sharp'] = (item['distances'] < self.resolution).long()
        return item

# class ComputeBackgroundMask(AbstractTransform):
#
#     def __init__(self):
#         super().__init__(None)
#
#     def __call__(self, item):
#         item['background_mask'] = (item['image'] == 0.0)
#         return item
