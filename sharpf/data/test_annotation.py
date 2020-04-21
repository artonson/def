import os
import sys
from unittest import TestCase

import urllib.request

from sharpf.utils.abc_utils.abc import ABCChunk, ABCModality

OBJ_URL = 'https://box.skoltech.ru/index.php/s/r17FSLbWI9NRmme/download'
FEAT_URL = 'https://box.skoltech.ru/index.php/s/JapYt3PAn50gq8u/download'
CHUNK_URLS = [OBJ_URL, FEAT_URL]

OBJ_FILENAME = os.path.expanduser('~/abc_0000_obj_v00.7z')
FEAT_FILENAME = os.path.expanduser('~/abc_0000_feat_v00.7z')
CHUNK_FILENAMES = [OBJ_FILENAME, FEAT_FILENAME]
CHUNK_MODALITIES = [ABCModality.OBJ.value, ABCModality.FEAT.value]


class AnnotationTestCase(TestCase):
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

    def test_resampling_aabb_annotation_coincide(self):
        with ABCChunk(CHUNK_FILENAMES) as chunk:



    def test_resampling_aabb_annotation_coincide(self):
        with ABCChunk(CHUNK_FILENAMES) as chunk:
