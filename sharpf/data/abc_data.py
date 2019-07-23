from collections import Iterable, ByteString
from contextlib import ExitStack
from enum import Enum
import glob
from io import BytesIO
from itertools import groupby
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


ALL_ABC_MODALITIES = [mod for mod in ABCModality]
ABC_7Z_FILEMASK = 'abc_{chunk}_{modality}_v{version}.7z'

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


# ABCItem is the primary data instance in ABC, containing all the modalities
ABCItem = namedtuple(
    'ABCItem',
    'pathname archive_pathname ' + ' '.join(modality.value for modality in ALL_ABC_MODALITIES))


class ABC7ZFile(object):
    """A helper class for reading 7z files."""

    def __init__(self, filename):
        self.filename = filename
        self.file_handle = None
        self.modality = _extract_modality(filename)

    def _open(self):
        self.file_handle = open(self.filename, 'rb')
        self.archive_handle = py7zlib.Archive7z(self.file_handle)

    def _close(self):
        self.file_handle.close()

    def __enter__(self):
        self._open()

    def __exit__(self, exc_type, exc_val, exc_tb):
        if self.file_handle:
            self._close()

    def __iter__(self):
        for name in self.archive_handle.getnames():
            bytes_io = BytesIO(self.archive_handle.getmember(name).read())
            yield ABCItem(self.filename, name, **{self.modality: bytes_io})



class ABCChunk(Iterable):
    """A chunk is a collection of files (with different modalities),
    that iterates over all files simultaneously."""

    def __init__(self, filenames):
        self.filenames = filenames
        self.file_handles = []

    def __iter__(self):
        with ExitStack() as stack:
            self.file_handles = [stack.enter_context(ABC7ZFile(filename))
                                 for filename in self.filenames]


        for filename, handle in zip(self.filenames, self.file_handles):


        item = ABCItem()

        yield item



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

        chunk_getter = lambda s: ABC_7Z_FILEMASK.

        self.data_files = {chunk: chunk_files
                           for chunk, chunk_files in groupby(self.data_files, key=chunk_getter)}

    def __iter__(self):
        for chunk in self.chunks:
            filenames_by_chunk = self.data_files[chunk]
            chunk = ABCChunk(**filenames_by_chunk)
            for item in chunk:
                yield item





class SevenZFile(object):
    @classmethod
    def is_7zfile(cls, filepath):
        '''
        Class method: determine if file path points to a valid 7z archive.
        '''
        is7z = False
        fp = None
        try:
            fp = open(filepath, 'rb')
            archive = py7zlib.Archive7z(fp)
            n = len(archive.getnames())
            is7z = True
        finally:
            if fp:
                fp.close()
        return is7z

    def __init__(self, filepath):
        fp = open(filepath, 'rb')
        self.archive = py7zlib.Archive7z(fp)

    def extractall(self, path):
        for name in self.archive.getnames():
            outfilename = os.path.join(path, name)
            outdir = os.path.dirname(outfilename)
            if not os.path.exists(outdir):
                os.makedirs(outdir)
            outfile = open(outfilename, 'wb')
            outfile.write(self.archive.getmember(name).read())
            outfile.close()


    f = open('abc_0099_meta_v00.7z', 'rb')
    a = py7zlib.Archive7z(f)
    a.getnames()


    from io import BytesIO
    b = BytesIO()


# testing use case #1: looping over all data in the entire dataset
# testing use case #2: looping over all data in specific chunks (subset of dataset)
# testing use case #3: looping over specified modalities in the entire dataset
# testing use case #4: parallel reading of different chunks
# testing use case #5: parallel reading of different files in the same chunk (?)

