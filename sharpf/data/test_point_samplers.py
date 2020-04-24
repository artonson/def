from sharpf.utils.abc_utils.abc.abc_data import ABC7ZFile
from sharpf.data.mesh_nbhoods import RandomEuclideanSphere
from sharpf.data.point_samplers import PoissonDiskSampler
from sharpf.utils.abc_utils.mesh.io import trimesh_load
from sharpf.utils.abc_utils.unittest import ABCDownloadableTestCase, OBJ_FILENAME

SAMPLER_CONFIG = {
    "type": "poisson_disk",
    "n_points": 4096,
    "resolution_3d": 0.02,
    "crop_center": True,
    "resolution_deviation_tolerance": 0.01
}
NBHOOD_CONFIG = {
    "type": "random_euclidean_sphere",
    "max_patches_per_mesh": 48,
    "n_vertices": None,
    "centroid": None,
    "centroid_mode": "poisson_disk",
    "radius_base": 10.0,
    "radius_delta": 0.0,
    "geodesic_patches": True,
    "radius_scale_mode": "no_scale"
}


class PoissonDiskSamplerTestCase(ABCDownloadableTestCase):
    def test_dense_mesh_has_required_n_points(self):
        sampler = PoissonDiskSampler.from_config(SAMPLER_CONFIG)

        with ABC7ZFile(OBJ_FILENAME) as f:
            for item in f:
                mesh = trimesh_load(item.obj)

                for extra_points_factor in range(5, 51, 5):
                    for n_points in range(512, 8193, 512):
                        dense_mesh = sampler._make_dense_mesh(
                            mesh, point_split_factor=4, extra_points_factor=extra_points_factor)
                        self.assertGreaterEqual(len(dense_mesh.vertices), extra_points_factor * n_points)

    def test_bad_meshes_raise(self):
        pass

    def test_pcu_poisson_disk_produces_n_points(self):
        sampler = PoissonDiskSampler.from_config(SAMPLER_CONFIG)
        ne = RandomEuclideanSphere.from_config(NBHOOD_CONFIG)

        with ABC7ZFile(OBJ_FILENAME) as f:
            for item in f:
                mesh = trimesh_load(item.obj)
                ne.index(mesh)
                for repeat in range(100):
                    submesh, _, _, _ = ne.get_nbhood()
                    points, normals = sampler.sample(submesh)
                    self.assertEquals(sampler.n_points, len(points))

