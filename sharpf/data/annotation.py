import itertools
from abc import ABC, abstractmethod

import numpy as np
from scipy.spatial import KDTree


class AnnotatorFunc(ABC):
    """Implements obtaining point samples from meshes.
    Given a mesh, extracts a point cloud located on the
    mesh surface, i.e. a set of 3d point locations."""
    @abstractmethod
    def annotate(self, mesh_patch, features, points, **kwargs):
        """Noises a point cloud.

        :param mesh_patch: an input mesh patch
        :type mesh_patch: MeshType (must be present attributes `vertices`, `faces`, and `edges`)
        :param features: an input feature annotation
        :type features: dict
        :param points: an input point cloud
        :type points: np.ndarray

        :returns: distances: euclidean distances to the closest sharp curve
        :rtype: np.ndarray
        :returns: directions: unit norm vectors in the direction to closest sharp curve
        :rtype: np.ndarray
        """
        pass


class SharpnessResamplingAnnotator(AnnotatorFunc):
    """Sample lots of points on sharp edges,
    compute distances from the input point clouds
    to the closest sharp points."""
    def __init__(self, distance_upper_bound, n_sharp_points):
        super(SharpnessResamplingAnnotator, self).__init__()
        self.distance_upper_bound = distance_upper_bound
        self.n_sharp_points = n_sharp_points

    def _resample_sharp_edges(self, mesh_patch, features):
        sharp_points = []
        for curve in features['curves']:
            if curve['sharp']:
                sharp_vert_indices = curve['vert_indices']
                sharp_edges = np.array([
                    pair_i_j for pair_i_j in itertools.combinations(np.unique(sharp_vert_indices), 2)
                    if (mesh_patch.edges == pair_i_j).all(axis=1).any(axis=0)
                ])
                first, second = mesh_patch.vertices[sharp_edges[:, 0]], mesh_patch.vertices[sharp_edges[:, 1]]
                sharp_points.append(first)
                sharp_points.append(second)

                # quite an ugly way to estimate the number of points to sample on edge
                n_points_per_edge = np.linalg.norm(first - second, axis=1) / self.sharp_discretization
                d_points_per_edge = 1. / n_points_per_edge
                for n, v1, v2 in zip(n_points_per_edge, first, second):
                    t = np.linspace(d_points_per_edge, 1 - d_points_per_edge, n)
                    sharp_points.append(np.outer(t, v1) + np.outer(1 - t, v2))

        sharp_points = np.concatenate(sharp_points)
        return sharp_points

    def annotate(self, mesh_patch, features_patch, points, **kwargs):
        # if patch is without sharp features
        if all(not curve['sharp'] for curve in features_patch['curves']):
            distances = np.ones_like(points[:, 0]) * self.distance_upper_bound
            directions = np.zeros_like(points)
            return distances, directions

        # model a dense sample of points lying on sharp edges
        sharp_points = self._resample_sharp_edges(mesh_patch, features_patch)

        # compute distances from each input point to the sharp points
        tree = KDTree(sharp_points)
        distances, vert_indices = tree.query(points, distance_upper_bound=self.distance_upper_bound)

        mask_point = distances == np.inf
        mask_sharp = vert_indices < len(sharp_samples)
        vert_indices = vert_indices[mask_sharp]

        directions = np.zeros_like(point_samples)
        directions[~mask_point] = sharp_samples[vert_indices] - point_samples[~mask_point]

        distances[mask_point] = distance_upper_bound

        # adding noise to regular boundary samples
        noise = (np.random.rand(*sharp_samples.shape) - 0.5) * 2 * noise_ampl
        edge_samples = sharp_samples + noise
        vert = np.concatenate([point_samples, sharp_samples])
        distances = np.concatenate([distances, np.linalg.norm(noise, axis=1)])
        directions = np.concatenate([directions, noise])
        return vert, distances, directions


    @classmethod
    def from_config(cls, config):
        return cls(config['distance_upper_bound'], config['n_sharp_points'])


ANNOTATOR_BY_TYPE = {
    'resampling': SharpnessResamplingAnnotator,
}

