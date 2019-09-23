from collections import Iterable, ByteString, defaultdict
from contextlib import ExitStack, AbstractContextManager
from enum import Enum
import glob
from io import BytesIO
from itertools import groupby, islice
import os

import py7zlib

from utils.namedtuple import namedtuple_with_defaults as namedtuple


class ABCModality(Enum):
    META = 'meta'
    PARA = 'para'
    STEP = 'step'
    STL2 = 'stl2'
    OBJ = 'obj'
    FEAT = 'feat'
    STAT = 'stat'


ALL_ABC_MODALITIES = [modality.value for modality in ABCModality]
ABC_7Z_FILEMASK = 'abc_{chunk}_{modality}_v{version}.7z'  # abc_0000_feat_v00.7z
ABC_INAR_FILEMASK = '{dirname}/{dirname}_{hash}_{modalityex}_{number}.{ext}'  # '00000002/00000002_1ffb81a71e5b402e966b9341_features_001.yml'

def _compose_filemask(chunks, modalities, version):

    chunk_mask = '*'
    if None is not chunks:
        assert isinstance(chunks, Iterable)
        assert all(isinstance(s, ByteString) for s in chunks)

        chunk_mask = '[' + '|'.join(chunks) + ']'

    modalities

    version

    return ABC_7Z_FILEMASK.format(
        chunk=chunk_mask, modality='x', version=0,
    )


def _extract_modality(filename):
    name = os.path.basename(filename)
    name, ext = os.path.splitext(name)
    abc, chunk, modality, version = name.split('_')
    return modality


def _extract_inar_id(pathname):
    # assume name matches ABC_INAR_FILEMASK
    name = os.path.basename(pathname)
    name, ext = os.path.splitext(name)
    dirname, hash, modalityex, number = name.split('_')
    return '_'.join([dirname, hash, number])


# ABCItem is the primary data instance in ABC, containing
# all the modalities in the form of bitstreams, supporting
# file-like interfacing e.g. `.read()`.
ABCItem = namedtuple(
    'ABCItem',
    'pathname archive_pathname item_id ' + ' '.join(ALL_ABC_MODALITIES),
    defaults=(None for modality in ALL_ABC_MODALITIES))


class ABC7ZFile(Iterable, AbstractContextManager):
    """A helper class for reading 7z files."""

    def __init__(self, filename):
        self.filename = filename
        self.modality = _extract_modality(filename)
        assert self.modality in ALL_ABC_MODALITIES, 'unknown modality: "{}"'.format(self.modality)
        self._reset_handles()

    def _reset_handles(self):
        self.file_handle = None
        self.archive_handle = None
        self._names_list = None

    def _open(self):
        self.file_handle = open(self.filename, 'rb')
        self.archive_handle = py7zlib.Archive7z(self.file_handle)
        self._names_list = self.archive_handle.getnames()

    def _close(self):
        self.file_handle.close()
        self._reset_handles()

    def __enter__(self):
        self._open()
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        if self._isopen():
            self._close()

    def _get_item_by_name(self, name):
        print(name)
        bytes_io = BytesIO(self.archive_handle.getmember(name).read())
        item_id = _extract_inar_id(name)
        return ABCItem(self.filename, name, item_id, **{self.modality: bytes_io})

    def __iter__(self):
        for name in self.archive_handle.getnames():
            yield self._get_item_by_name(name)

    def _isopen(self):
        return (self.file_handle is not None and
                self.archive_handle is not None and
                self._names_list is not None)

    def __getitem__(self, key):
        assert self._isopen()
        if isinstance(key, int):
            name = self._names_list[key]
            return self._get_item_by_name(name)
        elif isinstance(key, slice):
            names = list(islice(self._names_list, key.start, key.stop, key.step))
            return (self._get_item_by_name(name) for name in names)


class ABCChunk(Iterable):
    """A chunk is a collection of files (with different modalities),
    that iterates over all files simultaneously."""

    def __init__(self, filenames, load_chunk_to_memory=False):
        self.filenames = filenames
        self.required_modalities = {_extract_modality(filename) for filename in filenames}
        self.file_handles = []
        self.load_chunk_to_memory = load_chunk_to_memory

    def __iter__(self):
        with ExitStack() as stack:
            self.file_handles = [stack.enter_context(ABC7ZFile(filename))
                                 for filename in self.filenames]

            if self.load_chunk_to_memory:
                # if we cannot be sure that archive interiors are ordered
                # in the same way for all archives, we load them into memory
                iterables = zip(*self._preload())
            else:
                # otherwise, we try to iterate as on-the-fly as possible
                # specifically, we read items from each archive until all requested modalities
                # have been found, then merge and yield a unified item
                # (this could be useful if the archives are mostly aligned, but contain broken items)
                iterables = zip(*self.file_handles)


            read_items_by_id = defaultdict(dict)  # maps item + modality into

            for items in iterables:
                # TODO implement checks for raised exceptions during iterations
                for item, file in zip(items, self.file_handles):
                    read_items_by_id[item.item_id][file.modality] = item

                ids_to_yield = [item_id for item_id, read_modalities in read_items_by_id.items()
                                      if not self.required_modalities - read_modalities.keys()]

                for item_id in ids_to_yield:
                    read_modalities = read_items_by_id.pop(item_id)
                    bytes_io = {modality: getattr(item, modality)
                                for modality, item in read_modalities.items()}
                    merged_item = ABCItem(item_id=item_id, **bytes_io)
                    yield merged_item

    def _preload(self):
        """Physically load entire chunk into working memory. Return tuple of iterables."""
        import warnings
        loaded_data = []
        for filename, file_handle in zip(self.filenames, self.file_handles):
            warnings.warn('Loading large file {} into memory'.format(filename))
            loaded_data.append([item for item in file_handle])
        return loaded_data


class ABCData(Iterable):
    def __init__(self, data_dir, modalities=ALL_ABC_MODALITIES, chunks=None,
                 shape_representation='trimesh', version='00'):
        self.data_dir = data_dir
        self.modalities = modalities
        self.chunks = chunks
        self.version = version
        self.shape_representation = shape_representation

        filemask = _compose_filemask(self.chunks, self.modalities, self.version)

        self.data_files = glob.glob(os.path.join(self.data_dir, filemask))

        chunk_getter = lambda s: ABC_7Z_FILEMASK.format(s)

        self.data_files = {chunk: chunk_files
                           for chunk, chunk_files in groupby(self.data_files, key=chunk_getter)}

    def __iter__(self):
        for chunk in self.chunks:
            filenames_by_chunk = self.data_files[chunk]
            chunk = ABCChunk(**filenames_by_chunk)
            for item in chunk:
                yield item



# testing use case #1: looping over all data in the entire dataset
# testing use case #2: looping over all data in specific chunks (subset of dataset)
# testing use case #3: looping over specified modalities in the entire dataset
# testing use case #4: parallel reading of different chunks
# testing use case #5: parallel reading of different files in the same chunk (?)
#
from joblib import Parallel, delayed
#
def read_slice(filename, slice_params):
    slice_start, slice_end = slice_params
    print(slice_start, slice_end)
    with ABC7ZFile(filename) as abc_7z_file:
        print(abc_7z_file)
        x = abc_7z_file[slice_start:slice_end]
        for i, item in enumerate(x):
            y = item.obj.getvalue()
            print(filename, slice_start + i, len(y))

if __name__ == '__main__':
    slice_start = list(range(0, 7000, 100))
    slice_end = list(range(99, 7099, 100))
    slice_params = zip(slice_start, slice_end)
#
    filename = '/home/artonson/tmp/abc/abc_0000_obj_v00.7z'
    for i,j in slice_params:
        read_slice(filename, (i, j))
#   parallel = Parallel(n_jobs=40, verbose=10)
#   delayed_read_slice = delayed(read_slice)
#
#    parallel(
#        delayed_read_slice(filename, sp) for sp in slice_params
#    )