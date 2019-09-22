from collections.abc import Iterable, ByteString, Mapping
from collections import OrderedDict
from contextlib import ExitStack, AbstractContextManager
from itertools import groupby, islice
from abc import ABC, abstractmethod
from io import BytesIO
from enum import Enum
import glob
import re
import os
from operator import attrgetter

import py7zlib

from sharpf.utils.namedtuple import namedtuple_with_defaults as namedtuple


class ABCModality(Enum):
    META = 'meta'
    PARA = 'para'
    STEP = 'step'
    STL2 = 'stl2'
    OBJ = 'obj'
    FEAT = 'feat'
    STAT = 'stat'

MODALITY_TO_MODALITYEX = {'obj': 'trimesh', 'feat': 'features'}
MODALITY_TO_EXT = {'obj': 'obj', 'feat': 'yml'}
ALL_ABC_MODALITIES = [modality.value for modality in ABCModality]
ABC_7Z_FILEMASK = 'abc_{chunk}_{modality}_v{version}.7z'  # abc_0000_feat_v00.7z
ABC_7Z_REGEX = re.compile('abc_(\d{4})_([a-z-]+)_v(\d{2}).7z')  # abc_0000_feat_v00.7z
ABC_INAR_FILEMASK = '{dirname}/{dirname}_{hash}_{modalityex}_{number}.{ext}'  # '00000002/00000002_1ffb81a71e5b402e966b9341_features_001.yml'


def _compose_filemask(chunks, modalities, version):

    chunk_mask = '*'
    if None is not chunks:
        assert isinstance(chunks, Iterable)
        assert all(isinstance(s, ByteString) for s in chunks)

        chunk_mask = '[' + '|'.join(chunks) + ']'

    #modalities

    #version

    return [ABC_7Z_FILEMASK.format(
        chunk=chunk_mask, modality=modality, version=version,
    ) for modality in modalities]


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

def _make_name_from_id(item_id, modality):
    modalityex = MODALITY_TO_MODALITYEX[modality]
    ext = MODALITY_TO_EXT[modality]
    dirname, hash, number = item_id.split('_')
    return dirname + '/' + '_'.join([dirname, hash, modalityex, number]) + '.' + ext


class AbstractABCDataHolder(Mapping, Iterable, AbstractContextManager, ABC):
    """ An abstract parent class inherited from childred which operate with ABC 7z files  """

    def __exit__(self, exc_type, exc_val, exc_tb):
        if self._isopen():
            self.close()

    def __enter__(self):
        self._open()
        return self

    @abstractmethod
    def _unset_handles(self): pass

    @abstractmethod
    def close(self): pass

    @abstractmethod
    def _isopen(self): pass

    @abstractmethod
    def _open(self): pass

    @abstractmethod
    def __len__(self): pass

    @abstractmethod
    def __iter__(self): pass

    @abstractmethod
    def __getitem__(self, key): pass

    @abstractmethod
    def __contains__(self, key): pass

# ABCItem is the primary data instance in ABC, containing
# all the modalities in the form of bitstreams, supporting
# file-like interfacing e.g. `.read()`.
ABCItem = namedtuple(
    'ABCItem',
    'pathname archive_pathname item_id ' + ' '.join(ALL_ABC_MODALITIES),
    defaults=(None for modality in ALL_ABC_MODALITIES))

MergedABCItem = namedtuple(
    'MergedABCItem',
    'pathname_by_modality archive_pathname_by_modality item_id ' + ' '.join(ALL_ABC_MODALITIES),
    defaults=(None for modality in ALL_ABC_MODALITIES))


def values_attrgetter(d, attr):
    """Returns dict with original keys and values set to getattr(value, attr)"""
    return {key: attrgetter(attr)(value) for key, value in d.items()}


class ABC7ZFile(AbstractABCDataHolder):
    """A helper class for reading 7z files."""

    def __init__(self, filename):
        self._unset_handles()
        self.filename = filename
        try:
            self.modality = _extract_modality(filename)
        except:
            raise ValueError('cannot understand data modality for file "{}"'.format(self.filename))
        if self.modality not in ALL_ABC_MODALITIES:
            raise ValueError('unknown modality: "{}"'.format(self.modality))
        self._open()

    def _unset_handles(self):
        self.filename = None
        self.modality = None
        self.file_handle = None
        self.archive_handle = None
        self._names_list = None
        self._names_set = None

    def _open(self):
        self.file_handle = open(self.filename, 'rb')
        self.archive_handle = py7zlib.Archive7z(self.file_handle)
        self._names_list = self.archive_handle.getnames()
        self._names_set = set(self._names_list)
        self._name_by_id = OrderedDict([
            (_extract_inar_id(name), name) for name in self._names_list
        ])
        self.format = self._names_list[0].split('.')[-1]

    def close(self):
        self.file_handle.close()
        self._unset_handles()

    def _check_open(self):
        if not self._isopen():
            raise ValueError('I/O operation on closed file.')

    def __contains__(self, key):
        return key in self._names_set or key in self._name_by_id

    @property
    def ids(self):
        return self._name_by_id.keys()

    @property
    def names(self):
        return self._names_set

    def get(self, key):
        """Get item from file using a string identifier.
        :param key: either a string archive pathname of the item,
                    or its string identifier
        """
        self._check_open()
        if key in self._names_set:  # O(log n)
            archive_pathname = key
            item_id = _extract_inar_id(key)
        elif key in self._name_by_id:  # O(log n)
            archive_pathname = self._name_by_id[key]
            item_id = key
        else:
            raise KeyError('Archive does not contain requested object "{}"'.format(key))

        bytes_io = BytesIO(self.archive_handle.getmember(archive_pathname).read())
        return ABCItem(self.filename, archive_pathname, item_id, **{self.modality: bytes_io})

    def __iter__(self):
        self._check_open()
        for name in self._names_list:
            yield self.get(name)

    def _isopen(self):
        return all(obj is not None for obj in
                   [self.filename, self.modality, self.file_handle,
                    self.archive_handle, self._names_list, self._names_set])

    def __len__(self):
        self._check_open()
        return len(self._names_set)

    def __getitem__(self, key):
        """Get item from file using a integer or slice-based identifer.
        :param key: either an zero-based integer index of the item,
                    or a slice of such indexes"""
        self._check_open()
        if isinstance(key, int):
            name = self._names_list[key]
            return self.get(name)

        elif isinstance(key, slice):
            names = islice(self._names_list, key.start, key.stop, key.step)
            return (self.get(name) for name in names)


class ABCChunk(AbstractABCDataHolder):
    """A chunk is a collection of files (with different modalities),
    that iterates over all files simultaneously."""

    def __init__(self, filenames):
        self._unset_handles()
        self.filename_by_modality = {_extract_modality(filename): filename
                                     for filename in filenames}
        self.handle_by_modality = {}
        self.exitstack = ExitStack()
        self._open()

    def _unset_handles(self):
        self.filename_by_modality = None
        self.handle_by_modality = None
        self._id_list = None
        self._ids = None

    def _open(self):
        self.handle_by_modality = {
            modality: self.exitstack.enter_context(ABC7ZFile(filename))
            for modality, filename in self.filename_by_modality.items()
        }
        # we can only iterate over files existing simultaneously in all archives
        self._ids = set.intersection(
            *(set(handle.ids) for handle in self.handle_by_modality.values()))
        # list of objects is the same, but ordered as in the first file
        any_handle = next(iter(self.handle_by_modality.values()))
        self._id_list = [_id for _id in any_handle.ids if _id in self._ids]

    @property
    def ids(self):
        return self._ids

    def _isopen(self):
        return all(obj is not None
                   for obj in [self.filename_by_modality, self.handle_by_modality,
                               self._id_list, self._ids])

    def close(self):
        self.exitstack.close()
        self._unset_handles()

    def _check_open(self):
        if not self._isopen():
            raise ValueError('I/O operation on closed file.')

    def get_one(self, key, modality=None):
        """Get item from file using a string identifier or a string name.
        :param key: either a string archive pathname of the item,
                    or its string identifier
        :param modality: (optional) specify which modality to use
        """
        if None is modality:
            # don't want to specify the modality: search for the match
            for known_modality, handle in self.handle_by_modality.items():
                if key in handle:
                    modality = known_modality

        handle = self.handle_by_modality.get(modality)
        if None is handle:
            raise ValueError('unknown modality: "{}"'.format(modality))
        return handle.get(key)

    def get(self, item_id):
        """Get (merged) item from chunk using its unique string identifier."""
        self._check_open()
        item_by_modality = {modality: handle.get(item_id)
                            for modality, handle in self.handle_by_modality.items()}
        merged_item = MergedABCItem(
            pathname_by_modality={modality: item.pathname
                                  for modality, item in item_by_modality.items()},
            archive_pathname_by_modality={modality: item.archive_pathname
                                          for modality, item in item_by_modality.items()},
            item_id=item_id,
            **{modality: attrgetter(modality)(item)
               for modality, item in item_by_modality.items()})
        return merged_item

    def __iter__(self):
        self._check_open()
        for item_id in self._id_list:
            yield self.get(item_id)

    def __getitem__(self, key):
        self._check_open()

        if isinstance(key, int):
            return self.get(self._id_list[key])

        elif isinstance(key, slice):
            return (self.get(item_id) for item_id in self._id_list[key])

    def __len__(self):
        return len(self._id_list)

    def __contains__(self, key):
        """Support queries such as:
        >>> item_id in chunk  # True if all modalities of this ID are present
        >>> item_name in chunk  # True if some file in the chunk has this name
        """
        we_have_id = key in self.ids
        files_have_name = any(key in handle.names
                              for handle in self.handle_by_modality.values())
        return we_have_id or files_have_name


class ABCData(AbstractABCDataHolder):
    def __init__(self, data_dir, modalities=ALL_ABC_MODALITIES, chunks=None,
                 shape_representation='trimesh', version='00'):
        self.data_dir = data_dir
        self.modalities = modalities
        self.chunks = chunks
        self.version = version
        self.shape_representation = shape_representation
        self._unset_handles()
        self._open()

    def _unset_handles(self):
        self.data_files = None

    def _open(self):
        filemasks = _compose_filemask(self.chunks, self.modalities, self.version)
        self.data_files = [glob.glob(os.path.join(self.data_dir, filemask))[0] for filemask in filemasks]
        chunk_getter = lambda s: re.search(ABC_7Z_REGEX, s).group(1)
        self.data_files = {chunk: list(chunk_files)
                           for chunk, chunk_files in groupby(self.data_files, key=chunk_getter)}
        self.chunks = list(self.data_files.keys())

        self.lens = [len(ABCChunk(self.data_files[chunk])) for chunk in self.data_files.keys()]

    def close(self):
        self._unset_handles()
        self.chunks = None
        self.lens = None

    def _isopen(self):
        return all(obj is not None for obj in [self.data_files, self.lens, self.chunks])

    def __len__(self):
        return sum(self.lens)

    def __iter__(self):
        for chunk in self.chunks:
            filenames_by_chunk = self.data_files[chunk]
            chunk = ABCChunk(filenames_by_chunk)
            with ABCChunk(filenames_by_chunk) as chunk:
                for item in chunk:
                    yield item

    def __getitem__(self, key):
        if not self._isopen():
            raise ValueError('I/O operation on closed file.')
        curr_key = 0
        if isinstance(key, int):
            for id, item_len in enumerate(self.lens):
                if key < (curr_key + item_len):
                    files_by_key = self.data_files[self.chunks[id]]
                    break
                curr_key += item_len

            relative_key = key - curr_key
            item = ABCChunk(files_by_key)[relative_key]
            return item

        elif isinstance(key, slice):
            items = []
            curr_key = 0
            if key.start < 0:
                start = sum(self.lens) + key.start
            else:
                start = key.start

            if key.stop < 0:
                stop = sum(self.lens) + key.stop
            else:
                stop = key.stop
            for id, item_len in enumerate(self.lens):
                if stop < (curr_key + item_len):
                    files_by_key = self.data_files[self.chunks[id]]
                    break
                elif start < (curr_key + item_len):
                    files_by_key = self.data_files[self.chunks[id]]
                    relative_stop = self.lens[id]
                    relative_start = start - curr_key

                    chunk = ABCChunk(files_by_key)[relative_start:relative_stop:key.step]
                    items.append(chunk)

                    if key.step is None:
                        start = curr_key
                    else:
                        start = key.step - relative_stop % key.step - 1 + curr_key

                curr_key += item_len
            if key.stop > 0:
                relative_stop = key.stop - curr_key
            else:
                relative_stop = stop
            relative_start = start - curr_key
            chunk = ABCChunk(files_by_key)[relative_start:relative_stop:key.step]
            items.append(chunk)
            return (i for i in (j for j in items))

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
    with ABC7ZFile(filename) as abc_7z_file:
        x = abc_7z_file[slice_start]
        print(x.obj.getvalue())
        #x = abc_7z_file[slice_start:slice_end]
        #for i, item in enumerate(x):
        #    y = item.obj.getvalue()
        #    print(y)
        #    print(filename, slice_start + i, len(y))

def print_chunk_test(chunk):
    for item in chunk:
            print('obj:', len(item.obj.getvalue()))
            print('feat:', len(item.feat.getvalue()))

def read_chunk(filenames, key1=0, key2=10, step=1):
    with ABCChunk(filenames) as chunk:
        print('key test #1')
        print(chunk[key1])

        print('key test #2')
        print(chunk[key2])

        print('key test #3')
        print_chunk_test(chunk[key1:key2:step])

        #print('test whole chunk')
        #print(print_chunk_test(chunk))

def read_dataset(directory, modalities=['obj', 'feat'], key1=0, key2=-1, step=1):
    dataset = ABCData(directory, modalities)
    pass

if __name__ == '__main__':
    slice_start = list(range(0, 7000, 100))
    slice_end = list(range(99, 7099, 100))
    slice_params = zip(slice_start, slice_end)
    filename = '/home/artonson/tmp/abc/abc_0000_obj_v00.7z'
    a = ABC7ZFile(filename)
#    for i,j in slice_params:
#        read_slice(filename, (i, j))
#        break

    filename_1 = '/home/artonson/tmp/abc/abc_0000_obj_v00.7z'
    filename_2 = 'abc_0001_feat_v00.7z'
    filenames = [filename_1, filename_2]
    chunk = ABCChunk(filenames)
#    print('chunk created')
#    read_chunk(filenames, key1=100, key2=110)
#    print('ABCData:')
    dataset = ABCData('/home/gbobrovskih', modalities=['obj', 'feat'])
#    for item in dataset:
#        print(item)
#        break
#    print('key test #1')
#    print(dataset[0])
#    print('key test #2')
    for chunk in dataset[7160:7180]:
        for item in chunk:
            print('item:', len(item.obj.getvalue()))

    #for i,j in slice_params:
    #    read_slice(filename, (i, j))
#   parallel = Parallel(n_jobs=40, verbose=10)
#   delayed_read_slice = delayed(read_slice)
#
#    parallel(
#        delayed_read_slice(filename, sp) for sp in slice_params
#
