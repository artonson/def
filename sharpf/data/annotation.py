from abc import ABC, abstractmethod
from functools import partial
from operator import itemgetter

import numpy as np
from scipy.spatial import KDTree

from pyaabb import pyaabb

from sharpf.utils.geometry import dist_vector_proj


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
                sharp_edges = mesh_patch.edges_unique[
                    np.where(
                        np.all(np.isin(mesh_patch.edges_unique, curve['vert_indices']), axis=1)
                    )[0]
                ]
                if len(sharp_edges) > 0:
                    first, second = mesh_patch.vertices[sharp_edges[:, 0]], mesh_patch.vertices[sharp_edges[:, 1]]
                    n_points_per_edge = np.linalg.norm(first - second, axis=1) / self.sharp_discretization
                    d_points_per_edge = 1. / n_points_per_edge
                    for i1, i2, n, v1, v2 in zip(sharp_edges[:, 0], sharp_edges[:, 1], n_points_per_edge, first,
                                                 second):
                        t = np.linspace(d_points_per_edge, 1 - d_points_per_edge, int(n))
                        sharp_points.append(np.outer(t, v1) + np.outer(1 - t, v2))

        sharp_points = np.concatenate(sharp_points) if sharp_points else np.array(sharp_points)
        return sharp_points

    def annotate(self, mesh_patch, features_patch, points, distance_scaler=1, **kwargs):
        # if patch is without sharp features
        if all(not curve['sharp'] for curve in features_patch['curves']):
            distances = np.ones_like(points[:, 0]) * self.distance_upper_bound
            directions = np.zeros_like(points)
            return distances, directions

        # model a dense sample of points lying on sharp edges
        sharp_points = self._resample_sharp_edges(mesh_patch, features_patch)

        # compute distances from each input point to the sharp points
        tree = KDTree(sharp_points, leafsize=100)
        distances, vert_indices = tree.query(points, distance_upper_bound=self.distance_upper_bound * distance_scaler)

        distances = distances / distance_scaler

        far_from_sharp = distances == np.inf  # boolean mask marking objects far away from sharp curves
        distances[far_from_sharp] = self.distance_upper_bound

        # compute directions for points close to sharp curves
        directions = np.zeros_like(points)
        directions[~far_from_sharp] = sharp_points[vert_indices[~far_from_sharp]] - points[~far_from_sharp]
        eps = 1e-6
        directions[~far_from_sharp] /= (np.linalg.norm(directions[~far_from_sharp], axis=1, keepdims=True) + eps)

        return distances, directions

    @classmethod
    def from_config(cls, config):
        return cls(config['distance_upper_bound'], config['sharp_discretization'])


class AABBAnnotator(AnnotatorFunc):
    """Use axis-aligned bounding box representation sharp edges and compute
    distances from the input point cloud to the closest sharp edges."""
    def __init__(self, distance_upper_bound, closest_matching_distance_q, max_empty_envelope_radius):
        super(AABBAnnotator, self).__init__()
        self.distance_upper_bound = distance_upper_bound
        self.closest_matching_distance_q = closest_matching_distance_q
        self.max_empty_envelope_radius = max_empty_envelope_radius

    @classmethod
    def from_config(cls, config):
        return cls(config['distance_upper_bound'], config['closest_matching_distance_q'],
                   config['max_empty_envelope_radius'])

    def _prepare_aabb(self, mesh_patch, features):
        """Creates a set of axis-aligned bboxes """
        def create_aabboxes(lines):
            # create bounding boxes for sharp edges
            corners = []
            eps = 1e-8
            for l in lines:
                minc = np.array([
                    min(l[0][0], l[1][0]) - eps,
                    min(l[0][1], l[1][1]) - eps,
                    min(l[0][2], l[1][2]) - eps
                ])
                maxc = np.array([
                    max(l[0][0], l[1][0]) + eps,
                    max(l[0][1], l[1][1]) + eps,
                    max(l[0][2], l[1][2]) + eps
                ])
                corners.append([minc, maxc])
            return corners

        sharp_edge_indexes = np.concatenate([
            mesh_patch.edges_unique[
                np.where(
                    np.all(np.isin(mesh_patch.edges_unique, curve['vert_indices']), axis=1)
                )[0]
            ]
            for curve in features['curves'] if curve['sharp']])

        sharp_edges = mesh_patch.vertices[sharp_edge_indexes]
        aabboxes = create_aabboxes(sharp_edges)
        return aabboxes, sharp_edges

    def annotate(self, mesh_patch, features_patch, points, distance_scaler=1, **kwargs):
        # if patch is without sharp features
        if all(not curve['sharp'] for curve in features_patch['curves']):
            distances = np.ones_like(points[:, 0]) * self.distance_upper_bound
            directions = np.zeros_like(points)
            return distances, directions

        # model a dense sample of points lying on sharp edges
        aabboxes, sharp_edges = self._prepare_aabb(mesh_patch, features_patch)
        aabb_solver = pyaabb.AABB()
        aabb_solver.build(aabboxes)

        distance_func = partial(dist_vector_proj, lines=sharp_edges)

        query_results = [aabb_solver.nearest_point(p, distance_func) for p in points.astype('float32')]
        matching_edges, projections, distances = [np.array(map(itemgetter(i), query_results)) for i in [0, 1, 2]]

        # check whether most matching points live not too far: if they do, reset corresponding distances
        for edge_idx in range(len(sharp_edges)):
            matching_mask = matching_edges == edge_idx
            if np.any(matching_mask):
                closest_matching_distance = np.quantile(distances[matching_mask], self.closest_matching_distance_q)
                if closest_matching_distance > self.max_empty_envelope_radius:
                    distances[matching_mask] = self.distance_upper_bound

        distances = distances / distance_scaler
        far_from_sharp = distances > self.distance_upper_bound
        distances[far_from_sharp] = self.distance_upper_bound

        # compute directions for points close to sharp curves
        directions = np.zeros_like(points)
        directions[~far_from_sharp] = projections[~far_from_sharp] - points[~far_from_sharp]
        eps = 1e-6
        directions[~far_from_sharp] /= (np.linalg.norm(directions[~far_from_sharp], axis=1, keepdims=True) + eps)

        return distances, directions


ANNOTATOR_BY_TYPE = {
    'resampling': SharpnessResamplingAnnotator,
    'aabb': AABBAnnotator,
}

