from abc import ABC, abstractmethod
import os
import warnings

import numpy as np
from scipy.spatial import cKDTree

from sharpf.utils.abc_utils.abc.feature_utils import (
    get_intersecting_surfaces,
    get_boundary_curves,
    get_surface_as_mesh
)
from sharpf.utils.geometry_utils.geometry import mean_mmd


class DataSmell(ABC):
    @abstractmethod
    def run(self, *args): pass

    @classmethod
    def from_config(cls, config):
        return cls()


class SmellCoarseSurfacesByNumEdges(DataSmell):
    def __init__(self, num_edges_threshold):
        self.num_edges_threshold = num_edges_threshold

    @classmethod
    def from_config(cls, config):
        return cls(config['num_edges_threshold'])

    def run(self, mesh, mesh_face_indexes, features):
        intersecting_surfaces = get_intersecting_surfaces(mesh_face_indexes, features['surfaces'])
        for i, surface in enumerate(intersecting_surfaces):
            if surface.get('type', 'NAT') == 'Plane':
                continue
            boundary_curves = get_boundary_curves(mesh, surface, features)
            if len(boundary_curves) == 0:
                warnings.warn('Suspicious: no boundary curves extracted')
                continue
            shortest_curve_num_edges = min([len(curve['vert_indices']) - 1
                                            for curve in boundary_curves])
            if shortest_curve_num_edges < self.num_edges_threshold:
                return True

        return False


class SmellCoarseSurfacesByAngles(DataSmell):
    def __init__(self, max_angle_threshold_degrees):
        self.angle_cosine_threshold = np.cos(np.pi * max_angle_threshold_degrees / 180)

    @classmethod
    def from_config(cls, config):
        return cls(config['max_angle_threshold_degrees'])

    def run(self, mesh, mesh_face_indexes, features):
        intersecting_surfaces = get_intersecting_surfaces(mesh_face_indexes, features['surfaces'])
        for i, surface in enumerate(intersecting_surfaces):
            if surface.get('type', 'NAT') == 'Plane':
                continue
            surface_mesh = get_surface_as_mesh(mesh, surface, deduce_verts_from_faces=True)
            adjacent_normal_a = surface_mesh.face_normals[surface_mesh.face_adjacency][:, 0, :]
            adjacent_normal_b = surface_mesh.face_normals[surface_mesh.face_adjacency][:, 1, :]
            adjacent_angle_cosines = np.diag(np.dot(adjacent_normal_a, adjacent_normal_b.T))
            if np.median(adjacent_angle_cosines) < self.angle_cosine_threshold:
                return True

        return False


class SmellDeviatingResolution(DataSmell):
    def __init__(self, resolution_deviation_tolerance, resolution_3d):
        self.resolution_deviation_tolerance = resolution_deviation_tolerance
        self.resolution_3d = resolution_3d

    @classmethod
    def from_config(cls, config):
        return cls(config['resolution_deviation_tolerance'], config['resolution_3d'])

    def run(self, points):
        estimated_resolution_3d = mean_mmd(points)
        # warnings.warn(
        #     'Significant deviation in sampling density: '
        #     'resolution_3d = {resolution_3d:3.3f}, actual = {actual:3.3f} (difference = {diff:3.3f}, '
        #     'discarding patch'.format(resolution_3d=self.resolution_3d, actual=estimated_resolution_3d,
        #                               diff=np.abs(self.resolution_3d - estimated_resolution_3d)))
        return np.abs(self.resolution_3d - estimated_resolution_3d) > self.resolution_deviation_tolerance


class SmellSharpnessDiscontinuities(DataSmell):
    def run(self, points, distances):
        if len(points) == 1: 
            return False
        # validate for Lipshitz condition:
        # if for two points x_i and x_j (nearest neighbours of each other)
        # corresponding values f(x_i) and f(x_j) differ by more than ||x_i - x_j||, discard the patch
        n_omp_threads = int(os.environ.get('OMP_NUM_THREADS', 1))
        nn_distances, nn_indexes = cKDTree(points, leafsize=16).query(points, k=2, n_jobs=n_omp_threads)
        values = np.abs(distances[nn_indexes[:, 0]] - distances[nn_indexes[:, 1]]) / nn_distances[:, 1]
        # warnings.warn('Discontinuities found in SDF values, discarding patch')
        return np.any(values > 1.1)


class SmellBadFaceSampling(DataSmell):
    def __init__(self, min_points_per_face, max_points_per_face):
        self.min_points_per_face = min_points_per_face
        self.max_points_per_face = max_points_per_face

    @classmethod
    def from_config(cls, config):
        return cls(config['min_points_per_face'],
                   config['max_points_per_face'], )

    def run(self, nbhood, points):
        sampling_density = len(points) / len(nbhood.faces)
        return not (self.min_points_per_face <= sampling_density <= self.max_points_per_face)


class SmellRaycastingBackground(DataSmell):
    def run(self, image):
        return np.any(image == 0)


class SmellDepthDiscontinuity(DataSmell):
    def __init__(self, depth_discontinuity_threshold):
        self._depth_discontinuity_threshold = depth_discontinuity_threshold

    @classmethod
    def from_config(cls, config):
        return cls(config['depth_discontinuity_threshold'])

    def run(self, image):
        import skimage.filters as f
        s = f.sobel(image, mask=image != 0)
        return np.any(s > self._depth_discontinuity_threshold)


class SmellMeshSelfIntersections(DataSmell):
    def run(self, mesh):
        import pymesh
        self_intersections = pymesh.detect_self_intersection(mesh)
        return len(self_intersections) > 0
