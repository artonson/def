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
    def __init__(self, distance_upper_bound, sharp_discretization):
        super(SharpnessResamplingAnnotator, self).__init__()
        self.distance_upper_bound = distance_upper_bound
        self.sharp_discretization = sharp_discretization

    def _resample_sharp_edges(self, mesh_patch, features):
        sharp_points = []
        for curve in features['curves']:
            if curve['sharp']:
                sharp_vert_indices = curve['vert_indices']
                # there may be no sharp edges, but single vertices may be present -- add them
                sharp_points.append(mesh_patch.vertices[sharp_vert_indices])
                sharp_edges = np.array([
                    pair_i_j for pair_i_j in itertools.combinations(np.unique(sharp_vert_indices), 2)
                    if (mesh_patch.edges == pair_i_j).all(axis=1).any(axis=0)
                ])
                if len(sharp_edges) > 0:
                    first, second = mesh_patch.vertices[sharp_edges[:, 0]], mesh_patch.vertices[sharp_edges[:, 1]]
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
        tree = KDTree(sharp_points, leafsize=100)
        distances, vert_indices = tree.query(points, distance_upper_bound=self.distance_upper_bound)

        far_from_sharp = distances == np.inf  # boolean mask marking objects far away from sharp curves
        distances[far_from_sharp] = self.distance_upper_bound

        # compute directions for points close to sharp curves
        directions = np.zeros_like(points)
        directions[~far_from_sharp] = sharp_points[vert_indices[~far_from_sharp]] - points[~far_from_sharp]
        directions[~far_from_sharp] /= np.linalg.norm(directions[~far_from_sharp], axis=1, keepdims=True)

        return distances, directions

    @classmethod
    def from_config(cls, config):
        return cls(config['distance_upper_bound'], config['sharp_discretization'])


ANNOTATOR_BY_TYPE = {
    'resampling': SharpnessResamplingAnnotator,
}

