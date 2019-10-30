from abc import ABC, abstractmethod

import numpy as np
import trimesh
# TODO add to contrib and docker
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

        # v is a nv by 3 NumPy array of vertices
        # f is an nf by 3 NumPy array of face indexes into v
        # n is a nv by 3 NumPy array of vertex normals
        v, f, n, _ = pcu.read_ply("my_model.ply")
        bbox = np.max(v, axis=0) - np.min(v, axis=0)
        bbox_diag = np.linalg.norm(bbox)

        # Generate very dense  random samples on the mesh (v, f, n)
        # Note that this function works with no normals, just pass in an empty array np.array([], dtype=v.dtype)
        # v_dense is an array with shape (100*v.shape[0], 3) where each row is a point on the mesh (v, f)
        # n_dense is an array with shape (100*v.shape[0], 3) where each row is a the normal of a point in v_dense
        v_dense, n_dense = pcu.sample_mesh_random(v, f, n, num_samples=v.shape[0] * 100)

        # Downsample v_dense to be from a blue noise distribution:
        #
        # v_poisson is a downsampled version of v where points are separated by approximately
        # `radius` distance, use_geodesic_distance indicates that the distance should be measured on the mesh.
        #
        # n_poisson are the corresponding normals of v_poisson
        v_poisson, n_poisson = pcu.sample_mesh_poisson_disk(
            v_dense, f, n_dense, radius=0.01 * bbox_diag, use_geodesic_distance=True)

        return points, normals


class TrianglePointPickingSampler(SamplerFunc):
    """Sample using the Triangle-Point-Picking method
    http://mathworld.wolfram.com/TrianglePointPicking.html
    (Implementation by trimesh package) """
    def sample(self, mesh):
        points, face_indices = trimesh.sample.sample_surface(mesh, self.n_points)
        normals = mesh.face_normals[face_indices]
        return points, normals


SAMPLER_BY_TYPE = {
    'poisson_disk': PoissonDiskSampler,
    'tpp': TrianglePointPickingSampler,
}


