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

from sharpf.utils.abc_utils.abc.abc_data import ABC7ZFile, ABCChunk, ABCModality, ABCItem, _extract_inar_id, MergedABCItem, ALL_ABC_MODALITIES

from sharpf.utils.abc_utils.unittest import (
    CHUNK_URLS,
    OBJ_FILENAME, CHUNK_FILENAMES,
    CHUNK_MODALITIES,
    ABCDownloadableTestCase)


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


def get_info_from_chunk(chunk_filenames):
    chunk_len_items = {}
    chunk_item_ids = {}
    chunk_item_sizes = {}
    chunk_item_archive_pathnames = {}
    for filename in chunk_filenames:
        modality = extract_modality(filename)
        len_items, item_ids, item_sizes, item_archive_pathnames = \
            get_info_from_7z(filename)
        chunk_len_items[modality] = len_items
        chunk_item_ids[modality] = item_ids
        chunk_item_sizes[modality] = item_sizes
        chunk_item_archive_pathnames[modality] = item_archive_pathnames

    chunk_global_ids = set()
    for modality, ids in chunk_item_ids.items():
        if not chunk_global_ids:
            chunk_global_ids = set(ids)
        else:
            chunk_global_ids = chunk_global_ids.intersection(ids)
    return chunk_len_items, chunk_item_ids, chunk_item_archive_pathnames, chunk_item_sizes, chunk_global_ids


def get_i_from_each(d, i):
    return {key: value[i] for key, value in d.items()}
    # seq = ({key: value[i] for key, value in d.items()} for d in dicts)
    # return seq if len(dicts) > 1 else next(seq)


def extract_modality(filename):
    name = os.path.basename(filename)
    name, ext = os.path.splitext(name)
    abc, chunk, modality, version = name.split('_')
    return modality


class ABCTestCase(TestCase, ABC):
    def _check_item(self, item, filename, archive_pathname, item_id, item_size, modality=ABCModality.OBJ.value):
        self.assertIsInstance(item, ABCItem)
        self.assertTrue(hasattr(item, 'pathname'))
        self.assertEqual(item.pathname, filename)

        self.assertTrue(hasattr(item, 'archive_pathname'))
        self.assertEqual(item.archive_pathname, archive_pathname)

        self.assertTrue(hasattr(item, 'item_id'))
        self.assertEqual(item.item_id, item_id)

        self.assertTrue(hasattr(item, modality))
        self.assertIsNotNone(getattr(item, modality))
        self.assertIsInstance(getattr(item, modality), BytesIO)
        s = getattr(item, modality).getvalue()
        self.assertIsInstance(s, bytes)
        self.assertEqual(len(s), item_size)


class ABC7ZFileTestCase(ABCTestCase, ABCDownloadableTestCase):
    def _check_file_open(self, f):
        self.assertTrue(f._isopen())
        self.assertEquals(f.filename, OBJ_FILENAME)
        self.assertEqual(f.modality, ABCModality.OBJ.value)
        self.assertIsNotNone(f.file_handle)
        self.assertIsNotNone(f.archive_handle)

    def _check_file_closed(self, f):
        self.assertFalse(f._isopen())
        self.assertIsNone(f.filename)
        self.assertIsNone(f.modality)
        self.assertIsNone(f.file_handle)
        self.assertIsNone(f.archive_handle)

    def test_open_close(self):
        f = ABC7ZFile(OBJ_FILENAME)
        # once opened, all public variables are set
        self._check_file_open(f)
        f.close()
        # once closed, file is no longer holding information
        self._check_file_closed(f)

    def test_context_manager_open_close(self):
        with ABC7ZFile(OBJ_FILENAME) as f:
            # once opened, all public variables are set
            self._check_file_open(f)
        # once closed, file is no longer holding information
        self._check_file_closed(f)

    def test_iterate(self):
        len_items, item_ids, item_sizes, item_archive_pathnames = \
            get_info_from_7z(OBJ_FILENAME)

        with ABC7ZFile(OBJ_FILENAME) as f:
            self.assertEqual(len(f), len_items)
            for i, item in enumerate(f):
                self._check_item(
                    item, f.filename,
                    item_archive_pathnames[i], item_ids[i], int(item_sizes[i]))

    def test_raises_post_iterate(self):
        with ABC7ZFile(OBJ_FILENAME) as f:
            pass
        # cannot iterate when closed
        with self.assertRaises(ValueError):
            _ = next(iter(f))  # program stops after this line with assertion error appeared from __iter__
        # cannot subscript when closed
        with self.assertRaises(ValueError):
            _ = f[0]
        # cannot slice when closed
        with self.assertRaises(ValueError):
            _ = f[:-1]
        # cannot get by name when closed
        with self.assertRaises(ValueError):
            _ = f.get('any name')  # should raise ValueError before guessing that the name is bad (KeyError)

    def test_get_by_index(self):
        len_items, item_ids, item_sizes, item_archive_pathnames = \
            get_info_from_7z(OBJ_FILENAME)

        with ABC7ZFile(OBJ_FILENAME) as f:
            num_objects = len(f)
            self.assertEqual(len_items, num_objects)
            for i in range(num_objects):
                item = f[i]
                self._check_item(
                    item, f.filename,
                    item_archive_pathnames[i], item_ids[i], int(item_sizes[i]))

    def test_get_by_name(self):
        len_items, item_ids, item_sizes, item_archive_pathnames = \
            get_info_from_7z(OBJ_FILENAME)

        with ABC7ZFile(OBJ_FILENAME) as f:
            for i, name in enumerate(item_archive_pathnames):
                item = f.get(name)
                self._check_item(
                    item, f.filename,
                    item_archive_pathnames[i], item_ids[i], int(item_sizes[i]))

    def test_get_by_nonexistent_name_raises(self):
        with ABC7ZFile(OBJ_FILENAME) as f:
            with self.assertRaises(KeyError):
                _ = f.get('any name')

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
                            self._check_item(
                                item, f.filename,
                                item_archive_pathnames[i], item_ids[i], int(item_sizes[i]))

    def test_contains(self):
        _, item_ids, _, item_archive_pathnames = \
            get_info_from_7z(OBJ_FILENAME)

        with ABC7ZFile(OBJ_FILENAME) as f:
            for item_id in item_ids:
                self.assertIn(item_id, f)

            for archive_pathname in item_archive_pathnames:
                self.assertIn(archive_pathname, f)


class ABCChunkTestCase(ABCTestCase):
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

    def _check_close(self, chunk):
        self.assertFalse(chunk._isopen())
        self.assertIsNone(chunk.handle_by_modality)
        self.assertIsNone(chunk.filename_by_modality)

    def _check_open(self, chunk):
        self.assertTrue(chunk._isopen())
        for modality, filename in zip(CHUNK_MODALITIES, CHUNK_FILENAMES):
            self.assertIn(modality, chunk.filename_by_modality)
            self.assertIn(modality, chunk.handle_by_modality)
            self.assertEquals(chunk.filename_by_modality[modality], filename)

    def _check_merged_item(self, item, filename_by_modality, item_id_by_modality, archive_pathname_by_modality, item_size_by_modality):
        self.assertIsInstance(item, MergedABCItem)
        self.assertTrue(hasattr(item, 'pathname_by_modality'))
        self.assertIsInstance(item.pathname_by_modality, Mapping)
        self.assertTrue(hasattr(item, 'archive_pathname_by_modality'))
        self.assertIsInstance(item.archive_pathname_by_modality, Mapping)
        self.assertTrue(hasattr(item, 'item_id'))
        self.assertIsInstance(item.item_id, str)
        for modality in CHUNK_MODALITIES:
            self.assertIn(modality, item.pathname_by_modality)
            self.assertEqual(item.pathname_by_modality[modality], filename_by_modality[modality])
            self.assertEqual(item.archive_pathname_by_modality[modality], archive_pathname_by_modality[modality])
            self.assertEqual(item.item_id, item_id_by_modality[modality])
            self.assertTrue(hasattr(item, modality))

            attr = getattr(item, modality)
            self.assertIsNotNone(attr)
            self.assertIsInstance(attr, BytesIO)
            s = attr.getvalue()
            self.assertIsInstance(s, bytes)
            self.assertEqual(len(s), int(item_size_by_modality[modality]))

    def test_open_close(self):
        chunk = ABCChunk(CHUNK_FILENAMES)

        # once opened, all public variables are set
        self._check_open(chunk)

        chunk.close()

        # once closed, file is no longer holding information
        self._check_close(chunk)

    def test_context_manager_open_close(self):
        with ABCChunk(CHUNK_FILENAMES) as chunk:
            # once opened, all public variables are set
            self._check_open(chunk)

        # once closed, file is no longer holding information
        self._check_close(chunk)

    def test_iterate(self):
        chunk_len_items, chunk_item_ids, chunk_item_archive_pathnames, chunk_item_sizes, chunk_global_ids = \
            get_info_from_chunk(CHUNK_FILENAMES)

        with ABCChunk(CHUNK_FILENAMES) as chunk:
            self.assertEqual(len(chunk), len(chunk_global_ids))
            for i, item in enumerate(chunk):
                item_id_by_modality = get_i_from_each(chunk_item_ids, i)
                archive_pathname_by_modality = get_i_from_each(chunk_item_archive_pathnames, i)
                item_size_by_modality = get_i_from_each(chunk_item_sizes, i)
                self._check_merged_item(item, chunk.filename_by_modality,
                                        item_id_by_modality, archive_pathname_by_modality, item_size_by_modality)
 
    def test_raises_post_iterate(self):
        with ABCChunk(CHUNK_FILENAMES) as chunk:
            pass

        # cannot iterate when closed
        with self.assertRaises(ValueError):
            _ = next(iter(chunk))  # program stops after this line with assertion error appeared from __iter__

    def test_get_by_index(self):
        chunk_len_items, chunk_item_ids, chunk_item_archive_pathnames, chunk_item_sizes, _ = \
            get_info_from_chunk(CHUNK_FILENAMES)

        with ABCChunk(CHUNK_FILENAMES) as chunk:
            num_objects = len(chunk)
            for i in range(num_objects):
                item = chunk[i]
                item_id_by_modality = get_i_from_each(chunk_item_ids, i)
                archive_pathname_by_modality = get_i_from_each(chunk_item_archive_pathnames, i)
                item_size_by_modality = get_i_from_each(chunk_item_sizes, i)
                self._check_merged_item(item, chunk.filename_by_modality,
                                        item_id_by_modality, archive_pathname_by_modality, item_size_by_modality)

    def test_slice(self):
        chunk_len_items, chunk_item_ids, chunk_item_archive_pathnames, chunk_item_sizes, _ = \
            get_info_from_chunk(CHUNK_FILENAMES)

        with ABCChunk(CHUNK_FILENAMES) as chunk:
            num_objects = len(chunk)
            for start in range(num_objects):
                for stop in range(start + 1, num_objects):
                    for step in range(1, 10):
                        iterable = chunk[start:stop:step]
                        self.assertIsInstance(iterable, types.GeneratorType)
                        for i, item in enumerate(iterable):
                            i = start + i * step
                            item_id_by_modality = get_i_from_each(chunk_item_ids, i)
                            archive_pathname_by_modality = get_i_from_each(chunk_item_archive_pathnames, i)
                            item_size_by_modality = get_i_from_each(chunk_item_sizes, i)
                            self._check_merged_item(item, chunk.filename_by_modality,
                                                    item_id_by_modality, archive_pathname_by_modality, item_size_by_modality)

    def test_get_one(self):
        chunk_len_items, chunk_item_ids, chunk_item_archive_pathnames, chunk_item_sizes, chunk_global_ids = \
            get_info_from_chunk(CHUNK_FILENAMES)

        with ABCChunk(CHUNK_FILENAMES) as chunk:
            for modality in CHUNK_MODALITIES:

                for i, (item_id, archive_pathname, item_size) in enumerate(zip(
                        chunk_item_ids[modality], chunk_item_archive_pathnames[modality], chunk_item_sizes[modality])):
                    # get by item_id
                    item = chunk.get_one(item_id)
                    self.assertIsInstance(item, ABCItem)
                    self.assertTrue(hasattr(item, 'item_id'))
                    self.assertEqual(item.item_id, item_id)
                    self.assertTrue(all([
                        hasattr(item, some_modality) for some_modality in ALL_ABC_MODALITIES
                    ]))
                    the_modality = None
                    for some_modality in ALL_ABC_MODALITIES:
                        if hasattr(item, some_modality) and None is not getattr(item, some_modality):
                            the_modality = some_modality
                            break
                    self.assertIsNotNone(the_modality)
                    self.assertIsNotNone(getattr(item, the_modality))
                    self.assertIsInstance(getattr(item, the_modality), BytesIO)
                    s = getattr(item, the_modality).getvalue()
                    self.assertIsInstance(s, bytes)

                    # get by item_id, explicitly specifying modality
                    item = chunk.get_one(item_id, modality=modality)
                    self._check_item(item, chunk.filename_by_modality[modality],
                                     archive_pathname, item_id, int(item_size), modality=modality)

                    # get by archive_pathname
                    item = chunk.get_one(archive_pathname)
                    self._check_item(item, chunk.filename_by_modality[modality],
                                     archive_pathname, item_id, int(item_size), modality=modality)

                    # get by archive_pathname, explicitly specifying modality
                    item = chunk.get_one(archive_pathname, modality=modality)
                    self._check_item(item, chunk.filename_by_modality[modality],
                                     archive_pathname, item_id, int(item_size), modality=modality)

    def test_get_one_by_nonexistent_modality_raises(self):
        with ABCChunk(CHUNK_FILENAMES) as chunk:
            item = chunk[0]
            existing_key = item.item_id
            with self.assertRaises(ValueError):
                chunk.get_one(existing_key, modality='non existent')

    def test_get(self):
        chunk_len_items, chunk_item_ids, chunk_item_archive_pathnames, chunk_item_sizes, chunk_global_ids = \
            get_info_from_chunk(CHUNK_FILENAMES)

        with ABCChunk(CHUNK_FILENAMES) as chunk:
            for item_id in chunk_global_ids:
                idx_by_modality = {modality: chunk_item_ids[modality].index(item_id)
                                   for modality in CHUNK_MODALITIES}
                item_id_by_modality = dict.fromkeys(CHUNK_MODALITIES, item_id)
                archive_pathname_by_modality = {modality: chunk_item_archive_pathnames[modality][idx]
                                                for modality, idx in idx_by_modality.items()}
                item_size_by_modality = {modality: chunk_item_sizes[modality][idx]
                                         for modality, idx in idx_by_modality.items()}
                # get by item_id
                item = chunk.get(item_id)
                self._check_merged_item(item, chunk.filename_by_modality,
                                        item_id_by_modality, archive_pathname_by_modality, item_size_by_modality)

    def test_get_by_nonexistent_id_raises(self):
        with ABCChunk(CHUNK_FILENAMES) as chunk:
            with self.assertRaises(KeyError):
                chunk.get('non existent')

    def test_contains(self):
        _, _, chunk_item_archive_pathnames, _, chunk_global_ids = \
            get_info_from_chunk(CHUNK_FILENAMES)

        with ABCChunk(CHUNK_FILENAMES) as chunk:
            for item_id in chunk_global_ids:
                self.assertIn(item_id, chunk)

            for modality, archive_pathname_by_modality in chunk_item_archive_pathnames.items():
                for archive_pathname in archive_pathname_by_modality:
                    self.assertIn(archive_pathname, chunk)


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
