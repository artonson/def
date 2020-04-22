import numpy as np

from sharpf.utils.abc_utils.abc.abc_data import ABC7ZFile
from sharpf.data.mesh_nbhoods import RandomEuclideanSphere
from sharpf.utils.abc_utils.mesh.io import trimesh_load
from sharpf.utils.abc_utils.unittest import ABCDownloadableTestCase, OBJ_FILENAME


class NeighbourhoodTestCase(ABCDownloadableTestCase):
    def test_orig_indices_map_to_correct_verts_faces(self):
        def _run_test(geodesic=False):
            config = {
                "type": "random_euclidean_sphere",
                "max_patches_per_mesh": 10,
                "n_vertices": None,
                "centroid": None,
                "centroid_mode": "poisson_disk",
                "radius_base": 10.0,
                "radius_delta": 0.0,
                "geodesic_patches": geodesic,
                "radius_scale_mode": "no_scale"
            }
            nbhood_extractor = RandomEuclideanSphere.from_config(config)
            n_patches_per_mesh = 10

            with ABC7ZFile(OBJ_FILENAME) as f:
                for item in f:
                    mesh = trimesh_load(item.obj)
                    nbhood_extractor.index(mesh)
                    for patch_idx in range(n_patches_per_mesh):
                        nbhood, mesh_vertex_indexes, mesh_face_indexes, _ = nbhood_extractor.get_nbhood()
                        self.assertTrue(np.all(mesh.vertices[mesh_vertex_indexes] == nbhood.vertices))
                        self.assertTrue(
                            np.all(mesh.vertices[mesh.faces[mesh_face_indexes]] == nbhood.vertices[nbhood.faces]))

        _run_test()
        _run_test(geodesic=True)

    def test_geodesic_returns_single_cc(self):
        config = {
            "type": "random_euclidean_sphere",
            "max_patches_per_mesh": 10,
            "n_vertices": None,
            "centroid": None,
            "centroid_mode": "poisson_disk",
            "radius_base": 10.0,
            "radius_delta": 0.0,
            "geodesic_patches": True,
            "radius_scale_mode": "no_scale"
        }
        nbhood_extractor = RandomEuclideanSphere.from_config(config)
        n_patches_per_mesh = 10

        with ABC7ZFile(OBJ_FILENAME) as f:
            for item in f:
                mesh = trimesh_load(item.obj)
                nbhood_extractor.index(mesh)
                for patch_idx in range(n_patches_per_mesh):
                    nbhood, orig_vert_indices, orig_face_indexes, _ = nbhood_extractor.get_nbhood()
                    self.assertEquals(len(nbhood.split(only_watertight=False)), 1)


# TODO test that the largest CC is being actually selected
# TODO test that vertices and triangles actually selected belong to the largest CC
