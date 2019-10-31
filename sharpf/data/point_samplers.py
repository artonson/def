from abc import ABC, abstractmethod

import numpy as np
import trimesh
# TODO add to contrib and docker
import igl
from scipy.spatial import KDTree
import point_cloud_utils as pcu


class SamplerFunc(ABC):
    """Implements obtaining point samples from meshes.
    Given a mesh, extracts a point cloud located on the
    mesh surface, i.e. a set of 3d point locations."""
    def __init__(self, n_points):
        self.n_points = n_points

    @abstractmethod
    def sample(self, mesh):
        """Extracts a point cloud.

        :param mesh: an input mesh
        :type mesh: MeshType (must be present attributes `vertices`, `faces`, and `edges`)

        :returns: points: a point cloud
        :rtype: np.ndarray
        """
        pass

    @classmethod
    def from_config(cls, config):
        return cls(config['n_points'])


class PoissonDiskSampler(SamplerFunc):
    """Sample using the Poisson-Disk-Sampling of a mesh
    based on "Parallel Poisson Disk Sampling with Spectrum
    Analysis on Surface". (Implementation by fwilliams) """
    # https://github.com/marmakoide/mesh-blue-noise-sampling/blob/master/mesh-sampling.py
    def __init__(self, n_points, upsampling_factor, poisson_disk_radius):
        super().__init__(n_points)
        self.upsampling_factor = upsampling_factor
        self.poisson_disk_radius = poisson_disk_radius

    def sample(self, mesh):
        # Generate very dense subdivision samples on the mesh (v, f, n)
        dense_points, dense_faces = igl.upsample(mesh.vertices, mesh.faces, self.upsampling_factor)

        # compute vertex normals by pushing to trimesh
        umesh = trimesh.base.Trimesh(vertices=dense_points, faces=dense_faces, process=False, validate=False)
        dense_points = np.array(umesh.vertices, order='C')
        dense_normals = np.array(umesh.vertex_normals, order='C')

        # Downsample v_dense to be from a blue noise distribution:
        #
        # `points` is a downsampled version of `dense_points` where points are separated by approximately
        # `radius` distance, use_geodesic_distance indicates that the distance should be measured on the mesh.
        #
        # `normals` are the corresponding normals of `points`
        points, normals = pcu.sample_mesh_poisson_disk(
            dense_points, dense_faces, dense_normals,
            radius=self.poisson_disk_radius, use_geodesic_distance=True)

        # ensure that we are returning exactly n_points
        return_idx = np.random.choice(np.arange(len(points)), size=self.n_points, replace=False)
        return points[return_idx], normals[return_idx]

    @classmethod
    def from_config(cls, config):
        return cls(config['n_points'], config['upsampling_factor'], config['poisson_disk_radius'])


class TrianglePointPickingSampler(SamplerFunc):
    """Sample using the Triangle-Point-Picking method
    http://mathworld.wolfram.com/TrianglePointPicking.html
    (Implementation by trimesh package) """
    def sample(self, mesh):
        points, face_indices = trimesh.sample.sample_surface(mesh, self.n_points)
        normals = mesh.face_normals[face_indices]
        return points, normals


class LloydSampler(SamplerFunc):
    """Sample using the Lloyd algorithm"""
    def sample(self, mesh):
        points = pcu.sample_mesh_lloyd(mesh.vertices, mesh.faces, self.n_points)
        tree = KDTree(mesh.vertices)
        _, vert_indices = tree.query(points)
        normals = mesh.vertex_normals[vert_indices]
        return points, normals


SAMPLER_BY_TYPE = {
    'poisson_disk': PoissonDiskSampler,
    'tpp': TrianglePointPickingSampler,
    'lloyd': LloydSampler,
}


