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
        self._names_set = set(self._names_list)
        self.format = self._names_list[0].split('.')[-1]

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
        self.len = None
        self.exitstack = ExitStack()
        self.load_chunk_to_memory = load_chunk_to_memory
        self._unset_handles()
        self._open()

    def _unset_handles(self):
        self.file_handles = None

    def _open(self):
        self.file_handles = [self.exitstack.enter_context(ABC7ZFile(filename))
                                 for filename in self.filenames]
        self._names_set = set(map(_extract_inar_id, self.file_handles[0]._names_list))
        for file_handle in self.file_handles:
            curr_names = set(map(_extract_inar_id, file_handle._names_list))
            self._names_set = self._names_set.intersection(curr_names)
        self._names_list = list(self._names_set)
        self.len = len(self._names_set)

    def _isopen(self):
        return all(obj is not None 
                for obj in [self.filenames, self.len, self.required_modalities, self.file_handles])

    def close(self):
        self.filenames = None
        self.required_modalities = None
        self._names_set = None
        self._names_list
        self.len = None
        self.exitstack.close()
        self._unset_handles()        

    def __len__(self):
        return self.len

    def get(self, name):
        # this condition verifies whether this function
        # is used by user or by class methods
        if len(name.split('_')) == 4:
            item_id = _extract_inar_id(name)
        elif len(name.split('_')) == 3:
            item_id = name 
        
        read_items_by_id = {}  # maps item + modality into
        pathnames = {} # pathnames for each modality
        archivenames = {} # archive pathnames for each modality
        bytes_io = {}
        
        for file_handle in self.file_handles:
            modality = file_handle.modality
            name = _make_name_from_id(item_id, modality)
            item = file_handle.get(name)
            pathnames[modality] = item.pathname
            archivenames[modality] = item.archive_pathname
            bytes_io[modality] = getattr(item, modality)
        return ABCItem(pathname=pathnames, archive_pathname=archivenames, item_id=item_id, **bytes_io)

    def __iter__(self):
        if not self._isopen():
            raise ValueError('I/O operation on closed file.')
        
        for item_id in self._names_set:
            item = self.get(item_id)
            yield item

    def __getitem__(self, key):
        if not self._isopen():
            raise ValueError('I/O operation on closed file.') 
        if isinstance(key, int):
            
            return self.get(self._names_list[key])

        elif isinstance(key, slice):
            
            return (self.get(item_id) for item_id in self._names_list[key])
                
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
