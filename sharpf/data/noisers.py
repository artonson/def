from abc import ABC, abstractmethod

import numpy as np


class NoiserFunc(ABC):
    """Implements obtaining point samples from meshes.
    Given a mesh, extracts a point cloud located on the
    mesh surface, i.e. a set of 3d point locations."""
    def __init__(self, scale):
        self.scale = scale

    @classmethod
    def from_config(cls, config):
        return cls(config['scale'])

    @abstractmethod
    def make_noise(self, points, normals, **kwargs):
        """Noises a point cloud.

        :param points: an input point cloud
        :type points: np.ndarray
        :param normals: an input per-point normals
        :type normals: np.ndarray

        :returns: noisy_points: a noised point cloud
        :rtype: np.ndarray
        """
        pass


class IsotropicGaussianNoise(NoiserFunc):
    """Noise independent of viewing angle, mesh, etc."""
    def make_noise(self, points, normals, **kwargs):
        noise = np.random.normal(size=(len(points), 3), scale=self.scale)
        noisy_points = points + noise
        return noisy_points


class NormalsGaussianNoise(NoiserFunc):
    """Add gaussian noise in the direction of the normal."""
    def make_noise(self, points, normals, **kwargs):
        noise = np.random.normal(size=(len(points), 1), scale=self.scale) * normals
        noisy_points = points + noise * normals
        return noisy_points


class ZDirectionGaussianNoise(NoiserFunc):
    """Add gaussian noise in the direction of the normal."""
    def make_noise(self, points, normals, z_direction=None, **kwargs):
        assert z_direction is not None
        noise = np.random.normal(size=(len(points), 1), scale=self.scale) * z_direction
        noisy_points = points + noise * z_direction
        return noisy_points


class NoNoise(NoiserFunc):
    def make_noise(self, points, normals, **kwargs): return points

    @classmethod
    def from_config(cls, config): return cls()


NOISE_BY_TYPE = {
    'no_noise': NoNoise,
    'isotropic_gaussian': IsotropicGaussianNoise,
    'normals_gaussian': NormalsGaussianNoise,
    'z_direction': ZDirectionGaussianNoise,
}

