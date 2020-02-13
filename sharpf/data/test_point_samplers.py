import unittest

from sharpf.data.abc.abc_data import ABC7ZFile
from sharpf.data.point_samplers import PoissonDiskSampler
from sharpf.utils.mesh_utils import trimesh_load
from sharpf.utils.unittest import ABCDownloadableTestCase, OBJ_FILENAME

SAMPLER_CONFIG = {
    "type": "poisson_disk",
    "n_points": 4096,
    "upsampling_factor": 3,
    "poisson_disk_radius": 0.1
}


class PoissonDiskSamplerTestCase(ABCDownloadableTestCase):
    def test_make_dense_mesh(self):
        sampler = PoissonDiskSampler.from_config(SAMPLER_CONFIG)
        with ABC7ZFile(OBJ_FILENAME) as f:
            for item in f:
                mesh = trimesh_load(item.obj)




    def test_bad_meshes_raise(self):


    def test_pcu_poisson_disk_produces_n_points(self):



if __name__ == '__main__':
    unittest.main()
