import os
import sys
import unittest
import urllib.request

from sharpf.utils.abc_utils.abc.abc_data import ABCModality


OBJ_URL = 'https://box.skoltech.ru/index.php/s/r17FSLbWI9NRmme/download'
FEAT_URL = 'https://box.skoltech.ru/index.php/s/JapYt3PAn50gq8u/download'
CHUNK_URLS = [OBJ_URL, FEAT_URL]

OBJ_FILENAME = os.path.expanduser('~/abc_0000_obj_v00.7z')
FEAT_FILENAME = os.path.expanduser('~/abc_0000_feat_v00.7z')
CHUNK_FILENAMES = [OBJ_FILENAME, FEAT_FILENAME]
CHUNK_MODALITIES = [ABCModality.OBJ.value, ABCModality.FEAT.value]


class ABCDownloadableTestCase(unittest.TestCase):
    """Automatically downloads ABC test data."""

    @classmethod
    def setUpClass(cls):
        print('Downloading test data {}'.format(OBJ_FILENAME), file=sys.stderr)
        urllib.request.urlretrieve(OBJ_URL, OBJ_FILENAME)
        print('Downloading test data {}'.format(FEAT_FILENAME), file=sys.stderr)
        urllib.request.urlretrieve(FEAT_URL, FEAT_FILENAME)

    @classmethod
    def tearDownClass(cls):
        print('Cleaning test data {}'.format(OBJ_FILENAME), file=sys.stderr)
        os.remove(OBJ_FILENAME)
        print('Cleaning test data {}'.format(FEAT_FILENAME), file=sys.stderr)
        os.remove(FEAT_FILENAME)
