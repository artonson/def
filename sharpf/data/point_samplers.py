from abc import ABC, abstractmethod

import numpy as np
from scipy.spatial import KDTree
import trimesh
import igl
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
    def sample(self, mesh):
        # Intuition: take 10x the number of needed n_points,
        # keep in mind that each call to `igl.upsample` generates 4x the points,
        # then compute the upsampling factor K from the relation:
        # 4^K n = 10 n_points
        upsampling_factor = np.ceil(np.log(self.n_points * 10. / len(mesh.vertices)) / np.log(4)).astype(int)
        
        # Generate very dense subdivision samples on the mesh (v, f, n)
        for _ in range(upsampling_factor):
            mesh = mesh.subdivide()
        #dense_points, dense_faces = igl.upsample(mesh.vertices, mesh.faces, upsampling_factor)
        dense_points, dense_faces = mesh.vertices, mesh.faces

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
        poisson_disk_radius = float(np.mean(umesh.edges_unique_length))
        i, n_iter = 0, 10
        interval_left, interval_right = None, None
        while i < n_iter:
            i += 1
            points, normals = pcu.sample_mesh_poisson_disk(
                dense_points, dense_faces, dense_normals, num_samples=1100,
                radius=poisson_disk_radius, use_geodesic_distance=True)
            if self.n_points < len(points) < 1.1 * self.n_points:
                break
            elif len(points) < self.n_points:
                interval_right = poisson_disk_radius
                if None is not interval_left:
                    poisson_disk_radius = (interval_left + interval_right) / 2.
                else:
                    poisson_disk_radius /= 2.
            else:  # if len(points) > 1.1 * self.n_points
                interval_left = poisson_disk_radius
                if None is not interval_right:
                    poisson_disk_radius = (interval_left + interval_right) / 2.
                else:
                    poisson_disk_radius *= 2.

        # ensure that we are returning exactly n_points
        return_idx = np.random.choice(np.arange(len(points)), size=self.n_points, replace=False)
        return points[return_idx], normals[return_idx]


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
        tree = KDTree(mesh.vertices, leafsize=100)
        _, vert_indices = tree.query(points)
        normals = mesh.vertex_normals[vert_indices]
        return points, normals


SAMPLER_BY_TYPE = {
    'poisson_disk': PoissonDiskSampler,
    'tpp': TrianglePointPickingSampler,
    'lloyd': LloydSampler,
}


