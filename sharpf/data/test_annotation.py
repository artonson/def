import os

from sharpf.utils.abc_utils.abc.abc_data import ABCChunk, ABCModality
from sharpf.utils.abc_utils.unittest import ABCDownloadableTestCase

OBJ_URL = 'https://box.skoltech.ru/index.php/s/r17FSLbWI9NRmme/download'
FEAT_URL = 'https://box.skoltech.ru/index.php/s/JapYt3PAn50gq8u/download'
CHUNK_URLS = [OBJ_URL, FEAT_URL]

OBJ_FILENAME = os.path.expanduser('~/abc_0000_obj_v00.7z')
FEAT_FILENAME = os.path.expanduser('~/abc_0000_feat_v00.7z')
CHUNK_FILENAMES = [OBJ_FILENAME, FEAT_FILENAME]
CHUNK_MODALITIES = [ABCModality.OBJ.value, ABCModality.FEAT.value]


class AnnotationTestCase(ABCDownloadableTestCase):
    # TODO implement this test
    def test_resampling_aabb_annotation_coincide(self):
        with ABCChunk(CHUNK_FILENAMES) as chunk:
            pass
