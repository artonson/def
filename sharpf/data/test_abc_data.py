import os
import types
from io import BytesIO
import re
import subprocess
from unittest import TestCase

import urllib.request

from abc_data import ABC7ZFile, ABCChunk, ABCModality, ABCItem, _extract_inar_id


OBJ_URL = 'https://box.skoltech.ru/index.php/s/r17FSLbWI9NRmme/download'
OBJ_FILENAME = os.path.expanduser('~/abc_0000_obj_v00.7z')
FEAT_FILENAME = os.path.expanduser('~/abc_0000_feat_v00.7z')
ITEM_NAME = ''
CHUNK_FILENAMES = [OBJ_FILENAME]#[OBJ_FILENAME, FEAT_FILENAME]


def get_info_from_7z(filename):
    p_stdout = subprocess.check_output(['7z', 'l', filename], universal_newlines=True)

    item_ids, item_sizes, item_archive_pathnames = [], [], []

    regex = re.compile('^Date\s+Time\s+Attr\s+Size\s+Compressed\s+Name$')
    contents_regex = re.compile('^\d{4}-\d{2}-\d{2}\s+\d{2}:\d{2}:\d{2}\s+(?P<mode>.{5})\s+(?P<size>\d+)\s+(\d+)?\s+(?P<archive_pathname>.+)$')
    iterating_contents = False
    for line in p_stdout.splitlines():
        if not iterating_contents:
            if not regex.match(line.strip()):
                continue
            else:
                iterating_contents = True

        match = contents_regex.match(line.strip())
        if match and match.group('mode') == '....A':  # means this is a file, not a dir D....
            item_sizes.append(match.group('size'))
            item_archive_pathnames.append(match.group('archive_pathname'))
            item_ids.append(_extract_inar_id(match.group('archive_pathname')))

    len_items = len(item_ids)
    return len_items, item_ids, item_sizes, item_archive_pathnames

def extract_modality(filename):
    name = os.path.basename(filename)
    name, ext = os.path.splitext(name)
    abc, chunk, modality, version = name.split('_')
    return modality

class ABC7ZFileTestCase(TestCase):
    @classmethod
    def setUpClass(cls):
        urllib.request.urlretrieve(OBJ_URL, OBJ_FILENAME)

    @classmethod
    def tearDownClass(cls):
        os.remove(OBJ_FILENAME)

    def test_open_close(self):
        f = ABC7ZFile(OBJ_FILENAME)

        # once opened, all public variables are set
        self.assertTrue(f._isopen())
        self.assertEquals(f.filename, OBJ_FILENAME)
        self.assertEqual(f.modality, ABCModality.OBJ.value)
        self.assertIsNotNone(f.file_handle)
        self.assertIsNotNone(f.archive_handle)

        f.close()

        # once closed, file is no longer holding information
        self.assertFalse(f._isopen())
        self.assertIsNone(f.filename)
        self.assertIsNone(f.modality)
        self.assertIsNone(f.file_handle)
        self.assertIsNone(f.archive_handle)

    def test_context_manager_open_close(self):
        with ABC7ZFile(OBJ_FILENAME) as f:
            # once opened, all public variables are set
            self.assertTrue(f._isopen())
            self.assertEquals(f.filename, OBJ_FILENAME)
            self.assertEqual(f.modality, ABCModality.OBJ.value)
            self.assertIsNotNone(f.file_handle)
            self.assertIsNotNone(f.archive_handle)

        # once closed, file is no longer holding information
        self.assertFalse(f._isopen())
        self.assertIsNone(f.filename)
        self.assertIsNone(f.modality)
        self.assertIsNone(f.file_handle)
        self.assertIsNone(f.archive_handle)

    def test_iterate(self):
        len_items, item_ids, item_sizes, item_archive_pathnames = \
            get_info_from_7z(OBJ_FILENAME)

        with ABC7ZFile(OBJ_FILENAME) as f:
            self.assertEqual(len(f), len_items)
            for i, item in enumerate(f):
                self.assertIsInstance(item, ABCItem)
                self.assertTrue(hasattr(item, 'pathname'))
                self.assertEqual(item.pathname, f.filename)
                self.assertTrue(hasattr(item, 'archive_pathname'))
                self.assertEqual(item.archive_pathname, item_archive_pathnames[i])
                self.assertTrue(hasattr(item, 'item_id'))
                self.assertEqual(item.item_id, item_ids[i])
                self.assertTrue(hasattr(item, 'obj'))
                self.assertIsNotNone(item.obj)
                self.assertIsInstance(item.obj, BytesIO)
                s = item.obj.getvalue()
                self.assertIsInstance(s, bytes)
                self.assertEqual(len(s), int(item_sizes[i]))

    def test_raises_post_iterate(self):
        with ABC7ZFile(OBJ_FILENAME) as f:
            pass
        # cannot iterate when closed
        with self.assertRaises(ValueError):
            _ = next(iter(f))  # program stops after this line with assertion error appeared from __iter__

    def test_get_by_index(self):
        len_items, item_ids, item_sizes, item_archive_pathnames = \
            get_info_from_7z(OBJ_FILENAME)

        with ABC7ZFile(OBJ_FILENAME) as f:
            num_objects = len(f)
            self.assertEqual(len_items, num_objects)
            for i in range(num_objects):
                item = f[i]
                self.assertIsInstance(item, ABCItem)
                self.assertTrue(hasattr(item, 'pathname'))
                self.assertEqual(item.pathname, f.filename)
                self.assertTrue(hasattr(item, 'archive_pathname'))
                self.assertEqual(item.archive_pathname, item_archive_pathnames[i])
                self.assertTrue(hasattr(item, 'item_id'))
                self.assertEqual(item.item_id, item_ids[i])
                self.assertTrue(hasattr(item, 'obj'))
                self.assertIsNotNone(item.obj)
                self.assertIsInstance(item.obj, BytesIO)
                s = item.obj.getvalue()
                self.assertIsInstance(s, bytes)
                self.assertEqual(len(s), int(item_sizes[i]))

    def test_get_by_name(self):
        len_items, item_ids, item_sizes, item_archive_pathnames = \
            get_info_from_7z(OBJ_FILENAME)

        with ABC7ZFile(OBJ_FILENAME) as f:
            for i, name in enumerate(item_archive_pathnames):
                item = f.get(name)
                self.assertIsInstance(item, ABCItem)
                self.assertTrue(hasattr(item, 'pathname'))
                self.assertEqual(item.pathname, f.filename)
                self.assertTrue(hasattr(item, 'archive_pathname'))
                self.assertEqual(item.archive_pathname, item_archive_pathnames[i])
                self.assertTrue(hasattr(item, 'item_id'))
                self.assertEqual(item.item_id, item_ids[i])
                self.assertTrue(hasattr(item, 'obj'))
                self.assertIsNotNone(item.obj)
                self.assertIsInstance(item.obj, BytesIO)
                s = item.obj.getvalue()
                self.assertIsInstance(s, bytes)
                self.assertEqual(len(s), int(item_sizes[i]))

    def test_slice(self):
        len_items, item_ids, item_sizes, item_archive_pathnames = \
            get_info_from_7z(OBJ_FILENAME)

        with ABC7ZFile(OBJ_FILENAME) as f:
            for start in range(len_items):
                for stop in range(start + 1, len_items):
                    for step in range(1, 10):
                        iterable = f[start:stop:step]
                        self.assertIsInstance(iterable, types.GeneratorType)
                        for i, item in enumerate(iterable):
                            i = start + i * step
                            self.assertIsInstance(item, ABCItem)
                            self.assertTrue(hasattr(item, 'pathname'))
                            self.assertEqual(item.pathname, f.filename)
                            self.assertTrue(hasattr(item, 'archive_pathname'))
                            self.assertEqual(item.archive_pathname, item_archive_pathnames[i])
                            self.assertTrue(hasattr(item, 'item_id'))
                            self.assertEqual(item.item_id, item_ids[i])
                            self.assertTrue(hasattr(item, 'obj'))
                            self.assertIsNotNone(item.obj)
                            self.assertIsInstance(item.obj, BytesIO)
                            s = item.obj.getvalue()
                            self.assertIsInstance(s, bytes)
                            self.assertEqual(len(s), int(item_sizes[i]))


class ABCChunkTestCase(TestCase):
    @classmethod
    def setUpClass(cls):
        [urllib.request.urlretrieve(OBJ_URL, filename) for filename in CHUNK_FILENAMES] 

    @classmethod
    def tearDownClass(cls):
        [os.remove(OBJ_FILENAME) for filename in CHUNK_FILENAMES]

    def test_close(self, chunk):
        self.assertFalse(chunk._isopen())
        self.assertIsNone(chunk.filenames)
        self.assertIsNone(chunk.required_modalities)
        self.assertIsNone(chunk.file_handles)

    def test_item(self, i, item, filenames, chunk_len_items, chunk_item_ids, chunk_item_archive_pathnames, chunk_item_sizes):
        self.assertIsInstance(item, ABCItem)
        self.assertTrue(hasattr(item, 'pathname'))
        [self.assertTrue(item.pathname[modality] in filenames) for modality in ['obj']]#['obj', 'feat']]
        self.assertTrue(hasattr(item, 'archive_pathname'))
        [self.assertEqual(item.archive_pathname[modality], chunk_item_archive_pathnames[modality][i]) for modality in ['obj']]#['obj', 'feat']]
        self.assertTrue(hasattr(item, 'item_id'))
        [self.assertEqual(item.item_id, chunk_item_ids[modality][i]) for modality in ['obj']]#['obj', 'feat']]
        [self.assertTrue(hasattr(item, modality)) for modality in ['obj']]#['obj', 'feat']]

        self.assertIsNotNone(item.obj)
        self.assertIsInstance(item.obj, BytesIO)
        # check .feat ?                

        s = item.obj.getvalue()
        self.assertIsInstance(s, bytes)
        self.assertEqual(len(s), int(chunk_item_sizes['obj'][i]))
        # check .feat ?


    def test_open_close(self):
        chunk = ABCChunk(CHUNK_FILENAMES)

        # once opened, all public variables are set
        self.assertTrue(chunk._isopen())
        [self.assertEquals(f_1, f_2) for f_1, f_2 in zip(chunk.filenames, CHUNK_FILENAMES)] 
        [self.assertTrue(modality in chunk.required_modalities) for modality in [ABCModality.OBJ.value]]#[ABCModality.OBJ.value, ABCModality.FEAT.value]]
        self.assertIsNotNone(chunk.file_handles)

        chunk.close()

        # once closed, file is no longer holding information
        self.test_close(chunk)

    def test_context_manager_open_close(self):
        with ABCChunk(CHUNK_FILENAMES) as chunk:
            # once opened, all public variables are set
            self.assertTrue(chunk._isopen())
            [self.assertEquals(f_1, f_2) for f_1, f_2 in zip(chunk.filenames, CHUNK_FILENAMES)]
            [self.assertTrue(modality in chunk.required_modalities) for modality in [ABCModality.OBJ.value]]#[ABCModality.OBJ.value, ABCModality.FEAT.value]]
            self.assertIsNotNone(chunk.file_handles)

        # once closed, file is no longer holding information
        self.test_close(chunk)

    def test_iterate(self):
        chunk_len_items = {}
        chunk_item_ids = {}
        chunk_item_sizes = {}
        chunk_item_archive_pathnames = {}
        for filename in CHUNK_FILENAMES: 
            modality = extract_modality(filename)
            len_items, item_ids, item_sizes, item_archive_pathnames = \
                get_info_from_7z(filename)
            chunk_len_items[modality] = len_items
            chunk_item_ids[modality] = item_ids
            chunk_item_sizes[modality] = item_sizes
            chunk_item_archive_pathnames[modality] = item_archive_pathnames

        with ABCChunk(CHUNK_FILENAMES) as chunk:
            self.assertEqual(len(chunk), len_items)
            for i, item in enumerate(chunk):
                self.test_item(i, item, chunk.filenames, chunk_len_items, chunk_item_ids, chunk_item_archive_pathnames, chunk_item_sizes)
 
    def test_raises_post_iterate(self):
        with ABCChunk(CHUNK_FILENAMES) as chunk:
            pass

        # cannot iterate when closed
        with self.assertRaises(ValueError):
            _ = next(iter(chunk))  # program stops after this line with assertion error appeared from __iter__

    def test_get_by_index(self):
        chunk_len_items = {}
        chunk_item_ids = {}
        chunk_item_sizes = {}
        chunk_item_archive_pathnames = {}
        for filename in CHUNK_FILENAMES:
            modality = extract_modality(filename)
            len_items, item_ids, item_sizes, item_archive_pathnames = \
                get_info_from_7z(filename)
            chunk_len_items[modality] = len_items
            chunk_item_ids[modality] = item_ids
            chunk_item_sizes[modality] = item_sizes     
            chunk_item_archive_pathnames[modality] = item_archive_pathnames

        with ABCChunk(CHUNK_FILENAMES) as chunk:
            num_objects = len(chunk)
            [self.assertEqual(chunk_len_items[modality], num_objects) for modality in ['obj']]#['obj', 'feat']]
            for i in range(num_objects):
                item = chunk[i]
                self.test_item(i, item, chunk.filenames, chunk_len_items, chunk_item_ids, chunk_item_archive_pathnames, chunk_item_sizes)

    def test_slice(self):
        chunk_len_items = {}
        chunk_item_ids = {}
        chunk_item_sizes = {}
        chunk_item_archive_pathnames = {}
        for filename in CHUNK_FILENAMES:
            modality = extract_modality(filename)
            len_items, item_ids, item_sizes, item_archive_pathnames = \
                get_info_from_7z(filename)
            chunk_len_items[modality] = len_items
            chunk_item_ids[modality] = item_ids
            chunk_item_sizes[modality] = item_sizes
            chunk_item_archive_pathnames[modality] = item_archive_pathnames

        with ABCChunk(CHUNK_FILENAMES) as chunk:
            for start in range(len_items):
                for stop in range(start + 1, len_items):
                    for step in range(1, 10):
                        iterable = chunk[start:stop:step]
                        self.assertIsInstance(iterable, types.GeneratorType)
                        for i, item in enumerate(iterable):
                            i = start + i * step
                            self.test_item(i, item, chunk.filenames, chunk_len_items, chunk_item_ids, chunk_item_archive_pathnames, chunk_item_sizes)


class ABCDataTestCase(TestCase):
    @classmethod
    def setUpClass(cls):
        pass
    @classmethod
    def tearDownClass(cls):
        pass

if __name__ == '__main__':
    print('testing ABC7ZFile...')
    test = ABC7ZFileTestCase()
    test.setUpClass()
    print('test.test_context_manager_open_close')
    test.test_context_manager_open_close()
    print('test.test_iterate')
    test.test_iterate()
    print('test.test_raises_post_iterate')
    test.test_raises_post_iterate()
    print('test.test_get_by_index')
    test.test_get_by_index()
    print('test.test_get_by_name')
    test.test_get_by_name() 
    print('test.test_open_close')
    test.test_open_close()
    print('test.test_slice')
    test.test_slice()
    print('succesfully')
    test.tearDownClass()

    print('testing ABCChunk...')
    test = ABCChunkTestCase()
    test.setUpClass()
    print('test.test_open_close')
    test.test_open_close()
    print('test.test_iterate()')
    test.test_iterate()
    print('test.test_raises_post_iterate')
    test.test_raises_post_iterate()
    print('test.test_context_manager_open_close')
    test.test_context_manager_open_close()
    print('test.test_get_by_index')
    test.test_get_by_index()
    print('test.test_slice')
    test.test_slice()
    print('succesfully')
    test.tearDownClass()
