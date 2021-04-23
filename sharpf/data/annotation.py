from abc import ABC, abstractmethod

import numpy as np
import igl

import sharpf.utils.abc_utils.abc.feature_utils as features
from sharpf.utils.abc_utils.mesh.indexing import in2d
import sharpf.utils.geometry_utils.aabb as aabb
import sharpf.utils.geometry_utils.geometry as geom


def compute_bounded_labels(
        points,
        projections,
        distances=None,
        max_distance=np.inf,
        distance_scaler=1.0
):
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
    def __init__(
            self,
            distance_upper_bound: float,
            always_check_adjacent_surfaces: bool = False
    ):
        """Implements computing distance-to-feature annotations.
        Given a point set aligned with a mesh and the feature
        description of the mesh, computes distances to closest
        sharp curves."""
        self.distance_upper_bound = distance_upper_bound
        self.always_check_adjacent_surfaces = always_check_adjacent_surfaces

    def flat_annotation(self, points):
        projections = np.zeros_like(points)
        distances = np.ones_like(points[:, 0]) * self.distance_upper_bound
        directions = np.zeros_like(points)
        return projections, distances, directions

    def annotate(self, mesh_patch, features_patch, points, **kwargs):
        # if patch is without sharp features
        has_sharp = any(curve['sharp'] for curve in features_patch['curves'])
        if not has_sharp:
            _, distances, directions = self.flat_annotation(points)

        else:
            projections, distances = self.do_annotate(
                mesh_patch,
                features_patch,
                points)
            distances, directions = compute_bounded_labels(
                points,
                projections,
                distances=distances,
                max_distance=self.distance_upper_bound)

        return distances, directions, has_sharp

    @abstractmethod
    def do_annotate(self, mesh_patch, features, points, **kwargs):
        raise NotImplementedError()

    @classmethod
    def from_config(cls, config):
        return cls(
            config['distance_upper_bound'],
            always_check_adjacent_surfaces=config.get('always_check_adjacent_surfaces', False))


class AABBSurfacePatchAnnotator(AnnotatorFunc):
    """Use axis-aligned bounding box representation sharp edges and compute
    distances from the input point cloud to the closest sharp edges.

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
    def __init__(
            self,
            distance_upper_bound: float,
            always_check_adjacent_surfaces: bool = False,
            distance_computation_method: str = 'aabb'):
        """Implements computing distance-to-feature annotations.
        Given a point set aligned with a mesh and the feature
        description of the mesh, computes distances to closest
        sharp curves."""
        super().__init__(distance_upper_bound, always_check_adjacent_surfaces)
        self.distance_computation_method = distance_computation_method

    @classmethod
    def from_config(cls, config):
        return cls(
            config['distance_upper_bound'],
            always_check_adjacent_surfaces=config.get('always_check_adjacent_surfaces', False),
            distance_computation_method=config.get('distance_computation_method', 'aabb'))

    def do_annotate(self, mesh_patch, features_patch, points, **kwargs):
        # index mesh vertices to search for closest sharp features
        _, point_face_indexes, _ = igl.point_mesh_squared_distance(
            points,
            mesh_patch.vertices,
            mesh_patch.faces)

        # understand which surface patches are adjacent to which sharp features
        # and other surface patches
        adjacent_sharp_features, adjacent_surfaces = features.build_surface_patch_graph(
            mesh_patch, features_patch)

        # compute distance, iterating over points sampled from corresponding surface patches
        projections, distances, _ = self.flat_annotation(points)

        if self.distance_computation_method == 'aabb':
            pointset_edgeset_distances_projections = aabb.pointset_edgeset_distances_projections
        elif self.distance_computation_method == 'geom':
            pointset_edgeset_distances_projections = geom.parallel_pointset_edgeset_projections
        else:
            raise ValueError('distance_computation_method unknown')

        # iterate over points, computing stuff
        for surface_idx in range(len(features_patch['surfaces'])):
            surface = features_patch['surfaces'][surface_idx]

            # constrain distance computation to certain sharp features only
            adjacent_sharp_indexes = features.get_adjacent_features_by_bfs_with_depth1(
                surface_idx,
                adjacent_sharp_features,
                adjacent_surfaces,
                always_check_adjacent_surfaces=self.always_check_adjacent_surfaces)
            surface_adjacent_features = {
                'curves': [features_patch['curves'][idx]
                           for idx in np.unique(adjacent_sharp_indexes)]
            }
            if len(surface_adjacent_features['curves']) == 0:
                # no curves; no need to annotate from this surface patch
                continue

            # get all points the must be annotated
            # using curves adjacent to this surface patch
            points_indexes = np.where(in2d(
                mesh_patch.faces[point_face_indexes],
                mesh_patch.faces[surface['face_indices']]))[0]
            if len(points_indexes) == 0:
                continue

            sharp_edges = features.get_sharp_edge_endpoints(
                mesh_patch,
                surface_adjacent_features)
            surface_projections, surface_distances = pointset_edgeset_distances_projections(
                points[points_indexes],
                sharp_edges)
            distances[points_indexes], projections[points_indexes] = \
                surface_distances, surface_projections

        return projections, distances


ANNOTATOR_BY_TYPE = {
    'surface_based_aabb': AABBSurfacePatchAnnotator,
}
