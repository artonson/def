import os
import types
from abc import ABC
from io import BytesIO
import re
import subprocess
import sys
from typing import Mapping
from unittest import TestCase

import urllib.request

from sharpf.data.abc.abc_data import ABC7ZFile, ABCChunk, ABCModality, ABCItem, _extract_inar_id, MergedABCItem, ALL_ABC_MODALITIES
from sharpf.data.mesh_nbhoods import RandomEuclideanSphere
from sharpf.utils.mesh_utils.io import trimesh_load

OBJ_URL = 'https://box.skoltech.ru/index.php/s/r17FSLbWI9NRmme/download'
FEAT_URL = 'https://box.skoltech.ru/index.php/s/JapYt3PAn50gq8u/download'
CHUNK_URLS = [OBJ_URL, FEAT_URL]

OBJ_FILENAME = os.path.expanduser('~/abc_0000_obj_v00.7z')
FEAT_FILENAME = os.path.expanduser('~/abc_0000_feat_v00.7z')
CHUNK_FILENAMES = [OBJ_FILENAME, FEAT_FILENAME]
CHUNK_MODALITIES = [ABCModality.OBJ.value, ABCModality.FEAT.value]


class NeighbourhoodTestCase(TestCase):
    @classmethod
    def setUpClass(cls):
        for url, filename in zip(CHUNK_URLS, CHUNK_FILENAMES):
            print('Downloading test data {}'.format(filename), file=sys.stderr)
            urllib.request.urlretrieve(url, filename)

    @classmethod
    def tearDownClass(cls):
        for filename in CHUNK_FILENAMES:
            print('Cleaning test data {}'.format(filename), file=sys.stderr)
            os.remove(filename)

    def test_orig_indices_map_to_correct_verts_faces(self):
        def _run_test(geodesic=False):
            config = {
                "n_vertices": None,
                "centroid": None,
                "radius_base": 10.0,
                "radius_delta": 0.0,
                "geodesic_patches": geodesic,
                "radius_scale_mode": "no_scale"
            }
            nbhood_extractor = RandomEuclideanSphere.from_config(**config)
            n_patches_per_mesh = 10

            with ABC7ZFile(OBJ_FILENAME) as f:
                for item in f:
                    mesh = trimesh_load(item.obj)
                    nbhood_extractor.index(mesh)
                    for patch_idx in range(n_patches_per_mesh):
                        nbhood, orig_vert_indices, orig_face_indexes, _ = nbhood_extractor.get_nbhood()
                        self.assertEquals(mesh.vertices[orig_vert_indices], nbhood.vertices)
                        self.assertEquals(set(mesh.faces[orig_vert_indices]), set(nbhood.faces))

        _run_test()
        _run_test(geodesic=True)

    def test_geodesic_returns_single_cc(self):
        config = {
            "n_vertices": None,
            "centroid": None,
            "radius_base": 10.0,
            "radius_delta": 0.0,
            "geodesic_patches": True,
            "radius_scale_mode": "no_scale"
        }
        nbhood_extractor = RandomEuclideanSphere.from_config(**config)
        n_patches_per_mesh = 10

        with ABC7ZFile(OBJ_FILENAME) as f:
            for item in f:
                mesh = trimesh_load(item.obj)
                nbhood_extractor.index(mesh)
                for patch_idx in range(n_patches_per_mesh):
                    nbhood, orig_vert_indices, orig_face_indexes, _ = nbhood_extractor.get_nbhood()
                    self.assertEquals(len(nbhood.split(only_watertight=False)), 1)

