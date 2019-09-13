from collections.abc import Iterable, ByteString, Mapping
from collections import defaultdict
from contextlib import ExitStack, AbstractContextManager
from itertools import groupby, islice
from abc import ABC, abstractmethod
from io import BytesIO
from enum import Enum
import glob
import re
import os

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


# ABCItem is the primary data instance in ABC, containing
# all the modalities in the form of bitstreams, supporting
# file-like interfacing e.g. `.read()`.
ABCItem = namedtuple(
    'ABCItem',
    'pathname archive_pathname item_id ' + ' '.join(ALL_ABC_MODALITIES),
    defaults=(None for modality in ALL_ABC_MODALITIES))


class ABC7ZFile(AbstractABCDataHolder):
    """A helper class for reading 7z files."""

    def __init__(self, filename):
        self.filename = filename
        self.modality = _extract_modality(filename)
        assert self.modality in ALL_ABC_MODALITIES, 'unknown modality: "{}"'.format(self.modality)
        self._unset_handles()
        self._open()

    def _unset_handles(self):
        self.file_handle = None
        self.archive_handle = None
        self._names_list = None
        self._names_set = None

    def _open(self):
        self.file_handle = open(self.filename, 'rb') 
        self.archive_handle = py7zlib.Archive7z(self.file_handle)
        self._names_list = self.archive_handle.getnames() # rewrite with list and tree
        self._names_set = set(self.archive_handle.getnames())

    def close(self):
        self.file_handle.close()
        self._unset_handles()
        self.filename = None
        self.modality = None

    def get(self, name):
        if name not in self._names_set:  # O(log n)
            raise ValueError('Archive does not contain requested filename: {}'.format(name))
        bytes_io = BytesIO(self.archive_handle.getmember(name).read())
        item_id = _extract_inar_id(name)
        return ABCItem(self.filename, name, item_id, **{self.modality: bytes_io})

    def __iter__(self):
        if not self._isopen():
            raise ValueError('I/O operation on closed file.')
        for name in self.archive_handle.getnames():
            yield self.get(name)

    def _isopen(self):
        return all(obj is not None for obj in
                   [self.filename, self.modality, self.file_handle,
                    self.archive_handle, self._names_list, self._names_set])

    def __len__(self):
        return len(self._names_set)

    def __getitem__(self, key):
        assert self._isopen()
        if isinstance(key, int):
            name = self._names_list[key]
            return self.get(name)
        elif isinstance(key, slice):
            names = list(islice(self._names_list, key.start, key.stop, key.step))
            return (self.get(name) for name in names)

    
class ABCChunk(AbstractABCDataHolder):
    """A chunk is a collection of files (with different modalities),
    that iterates over all files simultaneously."""

    def __init__(self, filenames, load_chunk_to_memory=False):
        self.filenames = filenames
        self.required_modalities = {_extract_modality(filename) for filename in filenames}
        self.file_handles = []
        self.exitstack = ExitStack()
        self.load_chunk_to_memory = load_chunk_to_memory
        self._unset_handles()

    def _unset_handles(self):
        self.file_handles = None

    def _open(self):
        self.file_handles = [self.exitstack.enter_context(ABC7ZFile(filename))
                                 for filename in self.filenames]
        self.len = set([len(file_handle) for file_handle in self.file_handles])
        try:
            if len(self.len) > 1:
                raise Exception
        except Exception:
            print('different lengths')
            raise

    def _isopen(self):
        return (self.file_handles is not None)

    def close(self):
        self.exitstack.close()
        self._unset_handles()

    def __len__(self):
        return self.len[0]

    def _get_iterable(self, iterable):
        if self.load_chunk_to_memory:
            # if we cannot be sure that archive interiors are ordered
            # in the same way for all archives, we load them into memory
            iterables = zip(*self._preload(iterable))
        else:
            # otherwise, we try to iterate as on-the-fly as possible
            # specifically, we read items from each archive until all requested modalities
            # have been found, then merge and yield a unified item
            # (this could be useful if the archives are mostly aligned, but contain broken items)
            iterables = zip(*iterable)

        return iterables
    
    def _iterator(self, iterables, read_items_by_id, pathnames, archivenames):
        for items in iterables:
            # TODO implement checks for raised exceptions during iterations
            for item, file in zip(items, self.file_handles):
                 read_items_by_id[item.item_id][file.modality] = item
                 pathnames[item.item_id][file.modality] = item.pathname
                 archivenames[item.item_id][file.modality] = item.archive_pathname

            ids_to_yield = [item_id for item_id, read_modalities in read_items_by_id.items()
                            if not self.required_modalities - read_modalities.keys()]
            
            for item_id in ids_to_yield:
                read_modalities = read_items_by_id.pop(item_id)
                read_pathnames = pathnames.pop(item_id)
                read_achivenames = archivenames.pop(item_id)
                bytes_io = {modality: getattr(item, modality)
                            for modality, item in read_modalities.items()}
                merged_item = ABCItem(pathname=read_pathnames, archive_pathname=read_achivenames, item_id=item_id, **bytes_io)

                yield merged_item

    def __iter__(self):
        iterables = self._get_iterable(self.file_handles)
        read_items_by_id = defaultdict(dict)  # maps item + modality into
        pathnames = defaultdict(dict) # pathnames for each modality
        archivenames = defaultdict(dict) # archive pathnames for each modality

        return self._iterator(iterables, read_items_by_id, pathnames, archivenames)

    def __getitem__(self, key):
#        assert self._isopen()
        if isinstance(key, int):
            single_file_handle = [[file_handle[key]] for file_handle in self.file_handles] 

            iterables = self._get_iterable(single_file_handle) 

            read_items_by_id = defaultdict(dict)  # maps item + modality into
            pathnames = defaultdict(dict) # pathnames for each modality
            archivenames = defaultdict(dict) # archive pathnames for each modality

            merged_item = [item for item in self._iterator(iterables, read_items_by_id, pathnames, archivenames)]
            return merged_item

        elif isinstance(key, slice):
            slice_file_handles = [file_handle[key.start:key.stop:key.step] for file_handle in self.file_handles]

            iterables = self._get_iterable(slice_file_handles)

            read_items_by_id = defaultdict(dict)  # maps item + modality into
            pathnames = defaultdict(dict) # pathnames for each modality
            archivenames = defaultdict(dict) # archive pathnames for each modality

            merged_item = [item for item in self._iterator(iterables, read_items_by_id, pathnames, archivenames)]
            
            return merged_item

    def _preload(self, iterable):
        """Physically load entire chunk into working memory. Return tuple of iterables."""
        import warnings
        loaded_data = []
        for filename, file_handle in zip(self.filenames, iterable):
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

        filemasks = _compose_filemask(self.chunks, self.modalities, self.version)

        self.data_files = [glob.glob(os.path.join(self.data_dir, filemask))[0] for filemask in filemasks]

        chunk_getter = lambda s: re.search(ABC_7Z_REGEX, s).group(1)
        self.data_files = {chunk: list(chunk_files)
                           for chunk, chunk_files in groupby(self.data_files, key=chunk_getter)}

    def __iter__(self):
        for chunk in self.data_files.keys():
            filenames_by_chunk = self.data_files[chunk]
            chunk = ABCChunk(filenames_by_chunk)
            with ABCChunk(filenames_by_chunk) as chunk:
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

def read_chunk(filenames, key1=0, key2=10):
    with ABCChunk(filenames) as chunk:
        print('key test #1')
        print_chunk_test(chunk[key1])

        print('key test #2')
        print_chunk_test(chunk[key2])

        print('key test #3')
        print_chunk_test(chunk[key1:key2])

        print('test whole chunk')
        print(print_chunk_test(chunk))


if __name__ == '__main__':
    slice_start = list(range(0, 7000, 100))
    slice_end = list(range(99, 7099, 100))
    slice_params = zip(slice_start, slice_end)
    filename = 'abc_0000_obj_v00.7z'
    a = ABC7ZFile(filename)
#    for i,j in slice_params:
#        read_slice(filename, (i, j))
#        break

    filename_1 = '/home/artonson/tmp/abc/abc_0000_obj_v00.7z'
    filename_2 = '/home/artonson/tmp/abc/abc_0000_feat_v00.7z'
    filenames = [filename_1, filename_2]   
    read_chunk(filenames)
     
    dataset = ABCData('/home/artonson/tmp/abc/', modalities=['obj', 'feat'])
    for item in dataset:
        print(item)
        break


    #for i,j in slice_params:
    #    read_slice(filename, (i, j))
#   parallel = Parallel(n_jobs=40, verbose=10)
#   delayed_read_slice = delayed(read_slice)
#
#    parallel(
#        delayed_read_slice(filename, sp) for sp in slice_params
#   
