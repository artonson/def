from abc import ABC, abstractmethod
from functools import partial
from itertools import chain
from operator import itemgetter
import os

from joblib import Parallel, delayed
import numpy as np
from scipy.spatial import cKDTree
import igl
from pyaabb import pyaabb

from sharpf.data import DataGenerationException
from sharpf.utils.abc_utils import get_adjacent_features_by_bfs_with_depth1, build_surface_patch_graph
from sharpf.utils.geometry import dist_vector_proj
from sharpf.utils.mesh_utils.indexing import in2d


def compute_bounded_labels(points, projections, distances=None, max_distance=np.inf, distance_scaler=1.0):
    if distances is None:
        distances = np.linalg.norm(projections - points, axis=1)

    distances = distances / distance_scaler
    # boolean mask marking objects far away from sharp curves
    far_from_sharp = distances > max_distance
    distances[far_from_sharp] = max_distance
    # compute directions for points close to sharp curves
    directions = np.zeros_like(points)
    directions[~far_from_sharp] = projections[~far_from_sharp] - points[~far_from_sharp]
    eps = 1e-6
    directions[~far_from_sharp] /= (np.linalg.norm(directions[~far_from_sharp], axis=1, keepdims=True) + eps)
    return distances, directions


class AnnotatorFunc(ABC):
    """Implements obtaining point samples from meshes.
    Given a mesh, extracts a point cloud located on the
    mesh surface, i.e. a set of 3d point locations."""
    def __init__(self, distance_upper_bound, validate_annotation):
        self.distance_upper_bound = distance_upper_bound
        self.validate_annotation = validate_annotation

    def flat_annotation(self, points):
        projections = np.zeros_like(points)
        distances = np.ones_like(points[:, 0]) * self.distance_upper_bound
        directions = np.zeros_like(points)
        return projections, distances, directions

    def annotate(self, mesh_patch, features_patch, points, **kwargs):
        """Noises a point cloud.

        :param mesh_patch: an input mesh patch
        :type mesh_patch: MeshType (must be present attributes `vertices`, `faces`, and `edges`)

        :param features_patch: an input feature annotation
        :type features_patch: dict

        :param points: an input point cloud
        :type points: np.ndarray

        :returns: distances: euclidean distances to the closest sharp curve
        :rtype: np.ndarray

        :returns: directions: unit norm vectors in the direction to closest sharp curve
        :rtype: np.ndarray
        """

        # if patch is without sharp features
        has_sharp = any(curve['sharp'] for curve in features_patch['curves'])
        if not has_sharp:
            _, distances, directions = self.flat_annotation(points)

        else:
            projections, distances = self.do_annotate(mesh_patch, features_patch, points)

            distances, directions = compute_bounded_labels(
                points, projections, distances=distances,
                max_distance=self.distance_upper_bound)

            if self.validate_annotation:
                # validate for Lipshitz condition:
                # if for two points x_i and x_j (nearest neighbours of each other)
                # corresponding values f(x_i) and f(x_j) differ by more than ||x_i - x_j||, discard the patch
                n_omp_threads = int(os.environ.get('OMP_NUM_THREADS', 1))
                nn_distances, nn_indexes = cKDTree(points, leafsize=16).query(points, k=2, n_jobs=n_omp_threads)
                values = np.abs(distances[nn_indexes[:, 0]] - distances[nn_indexes[:, 1]]) / nn_distances[:, 1]
                if np.any(values > 1.1):
                    raise DataGenerationException('Discontinuities found in SDF values, discarding patch')

        return distances, directions, has_sharp

    @abstractmethod
    def do_annotate(self, mesh_patch, features, points, **kwargs):
        raise NotImplementedError()


class SharpnessResamplingAnnotator(AnnotatorFunc):
    """Sample lots of points on sharp edges,
    compute distances from the input point clouds
    to the closest sharp points."""

    def __init__(self, distance_upper_bound, sharp_discretization):
        super(SharpnessResamplingAnnotator, self).__init__(distance_upper_bound)
        self.sharp_discretization = sharp_discretization

    @classmethod
    def from_config(cls, config):
        return cls(config['distance_upper_bound'], config['sharp_discretization'])

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

    def do_annotate(self, mesh_patch, features_patch, points, **kwargs):
        # model a dense sample of points lying on sharp edges
        sharp_points = self._resample_sharp_edges(mesh_patch, features_patch)
        # compute distances from each input point to the sharp points
        tree = KDTree(sharp_points, leafsize=100)
        distances, vert_indices = tree.query(points, distance_upper_bound=self.distance_upper_bound)
        return sharp_points[vert_indices], distances


class AABBAnnotator(AnnotatorFunc, ABC):
    """Use axis-aligned bounding box representation sharp edges and compute
    distances from the input point cloud to the closest sharp edges."""

    @classmethod
    def from_config(cls, config):
        return cls(config['distance_upper_bound'], config['validate_annotation'])

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

    def compute_aabb_nearest_points(self, mesh_patch, features_patch, points):
        aabboxes, sharp_edges = self._prepare_aabb(mesh_patch, features_patch)
        aabb_solver = pyaabb.AABB()
        aabb_solver.build(aabboxes)
        distance_func = partial(dist_vector_proj, lines=sharp_edges)

        def parallel_nearest_point(aabb_solver, points, distance_func):
            return [aabb_solver.nearest_point(p, distance_func)
                    for p in points.astype(np.float32)]

        n_omp_threads = int(os.environ.get('OMP_NUM_THREADS', 1))
        parallel = Parallel(n_jobs=n_omp_threads, backend='multiprocessing')
        delayed_iterable = (delayed(parallel_nearest_point)(aabb_solver, points_to_thread, distance_func)
                            for points_to_thread in np.split(points.astype(np.float32), n_omp_threads))
        query_results = list(chain(parallel(delayed_iterable)))

        # query_results = [aabb_solver.nearest_point(p, distance_func) for p in points.astype('float32')]
        matching_edges, projections, distances = [np.array(list(map(itemgetter(i), query_results))) for i in [0, 1, 2]]
        return matching_edges, projections, distances


class AABBGlobalAnnotator(AABBAnnotator):
    """Use axis-aligned bounding box representation sharp edges and compute
    distances from the input point cloud to the closest sharp edges."""

    def __init__(self, distance_upper_bound, closest_matching_distance_q, max_empty_envelope_radius):
        super(AABBGlobalAnnotator, self).__init__(distance_upper_bound)
        self.closest_matching_distance_q = closest_matching_distance_q
        self.max_empty_envelope_radius = max_empty_envelope_radius

    @classmethod
    def from_config(cls, config):
        return cls(config['distance_upper_bound'], config['closest_matching_distance_q'],
                   config['max_empty_envelope_radius'])

    def reset_distances(self, distances, matching_edges):
        # check whether most matching points live not too far: if they do, reset corresponding distances
        # TODO code not used
        for edge_idx in range(len(sharp_edges)):
            matching_mask = matching_edges == edge_idx
            close_matching_mask = matching_mask & ~(distances == self.distance_upper_bound)
            if np.any(close_matching_mask):
                closest_matching_distance = np.quantile(distances[close_matching_mask],
                                                        self.closest_matching_distance_q)
                if closest_matching_distance > self.max_empty_envelope_radius:
                    distances[close_matching_mask] = self.distance_upper_bound
        return distances

    def do_annotate(self, mesh_patch, features_patch, points, **kwargs):
        matching_edges, projections, distances = self.compute_aabb_nearest_points(mesh_patch, features_patch, points)
        return projections, distances


class AABBSurfacePatchAnnotator(AABBAnnotator):
    """
     * For each surface patch `SurfPatch` in the neighbourhood, find a set of vertices `V`
        belonging to it - `surface['vert_indices']` (reindexed)
     * For each point in point cloud, `p_i in PointCloud`, we know the closest
        vertex `v(p_i)`, and we know to which surface patch it belongs
     * For each surface patch, we know the set of adjacent sharp features `SharpFeat` (in the form of edges)

    ```
    for each surface patch SurfPatch:
      SharpFeat = build_sharp_features(SurfPatch)
      for each point p_i in PointCloud such that the closest v(p_i) is in SurfPatch:
        d = find_distance(p_i, SharpFeat)
    ```
    """

    def do_annotate(self, mesh_patch, features_patch, points, **kwargs):
        # index mesh vertices to search for closest sharp features
        # tree = KDTree(mesh_patch.vertices, leafsize=100)
        # _, closest_nbhood_vertex_idx = tree.query(points)
        _, point_face_indexes, _ = \
            igl.point_mesh_squared_distance(points, mesh_patch.vertices, mesh_patch.faces)

        # understand which surface patches are adjacent to which sharp features
        # and other surface patches
        adjacent_sharp_features, adjacent_surfaces = build_surface_patch_graph(features_patch)

        # compute distance, iterating over points sampled from corresponding surface patches
        projections, distances, _ = self.flat_annotation(points)
        for surface_idx, surface in enumerate(features_patch['surfaces']):
            # constrain distance computation to certain sharp features only
            adjacent_sharp_indexes = get_adjacent_features_by_bfs_with_depth1(
                surface_idx, adjacent_sharp_features, adjacent_surfaces)
            surface_adjacent_features = {
                'curves': [features_patch['curves'][idx]
                           for idx in np.unique(adjacent_sharp_indexes)]
            }
            if len(surface_adjacent_features['curves']) == 0:
                continue

            point_cloud_indexes = np.where(
                in2d(mesh_patch.faces[point_face_indexes], surface['face_indices'])
            )[0]
            # point_cloud_indexes = np.where(np.isin(closest_nbhood_vertex_idx, surface['vert_indices']))[0]
            if len(point_cloud_indexes) == 0:
                continue
            # compute distances using parent class AABB method
            surface_matching_edges, surface_projections, surface_distances = \
                self.compute_aabb_nearest_points(mesh_patch, surface_adjacent_features, points[point_cloud_indexes])
            distances[point_cloud_indexes], projections[point_cloud_indexes] = surface_distances, surface_projections

        return projections, distances


ANNOTATOR_BY_TYPE = {
    'resampling': SharpnessResamplingAnnotator,
    'global_aabb': AABBGlobalAnnotator,
    'surface_based_aabb': AABBSurfacePatchAnnotator,
}
