import os
import glob
from enum import Enum

import h5py
import numpy as np
import torch
from torch.utils.data import Dataset

from sharpf.utils.py_utils.console import eprint
from sharpf.utils.py_utils.parallel import threaded_parallel

class PreloadTypes(Enum):
    ALWAYS = 'always'
    LAZY = 'lazy'
    NEVER = 'never'


class Hdf5File(Dataset):

    def __init__(self, filename, io, data_label=None, target_label=None, labels=None, preload=PreloadTypes.ALWAYS,
                 transform=None):
        """Represents HDF5 dataset contained in a single HDF5 file.

        :param filename: name of the file
        :param io: HDF5IO object serving as a I/O interface to the HDF5 data files
        :param data_label: string label in HDF5 dataset corresponding to data to train from
        :param target_label: string label in HDF5 dataset corresponding to targets
        :param labels: a list of HDF5 dataset labels to read off the file ('*' for ALL keys)
        :param preload: determines the data loading strategy:
            'always': entire data is read off disk in constructor
            'lazy': entire data is loaded on first access
            'never': entire data never loaded, only the requested data portions are read off disk in getitem
        :param transform: callable implementing data + target transform (e.g., adding noise)
        """
        self.filename = os.path.normpath(os.path.realpath(filename))
        assert not all([value is None for value in [data_label, target_label, labels]]), \
            'Specify value for at least data_label, target_label, or labels'

        self.data_label = data_label
        self.target_label = target_label
        self.transform = transform
        self.items = None  # this is where the data internally is read to
        self.io = io
        assert preload in PreloadTypes, 'unknown preload type: {}'.format(preload)
        self.preload = preload

        with h5py.File(self.filename, 'r') as f:
            self.num_items = self._get_length(f)
            if labels == '*':
                labels = set(f.keys())
            elif None is labels:
                labels = set()
            else:
                labels = set(labels)

        default_labels = set([label for label in [data_label, target_label] if label is not None])
        self.labels = list(default_labels.union(labels))

        if self.preload == PreloadTypes.PRELOAD_ALWAYS:
            self.reload()

    def _get_length(self, hdf5_file):
        try:
            num_items = self.io.length(hdf5_file)
        except KeyError:
            eprint('File {} is not compatible with Hdf5File I/O interface {}'.format(
                self.filename, str(self.io.__class__)))
            num_items = 0
        return num_items

    def __len__(self):
        return self.num_items

    def __getitem__(self, index):
        item = self._get_item(index)

        data = None
        if None is not self.data_label:
            data = torch.from_numpy(item[self.data_label])

        target = None
        if None is not self.target_label:
            target = torch.from_numpy(item[self.target_label])

        if self.transform is not None:
            data, target = self.transform(data, target)

        if None is not self.data_label:
            item.update({self.data_label: data})

        if None is not self.target_label:
            item.update({self.target_label: target})

        return item

    def reload(self):
        with h5py.File(self.filename, 'r') as f:
            self.num_items = self._get_length(f)
            self.items = {label: self.io.read(f, label)
                          for label in self.labels}

    def load_one(self, index):
        with h5py.File(self.filename, 'r') as f:
            self.num_items = self._get_length(f)
            return {label: self.io.read_one(f, label, index)
                    for label in self.labels}

    def is_loaded(self):
        return None is not self.items

    def unload(self):
        self.items = None

    def _get_item(self, index):
        if self.preload in [PreloadTypes.PRELOAD_LAZY, PreloadTypes.ALWAYS]:
            if not self.is_loaded():
                self.reload()
            item = {label: self.items[label][index]
                    for label in self.labels}

        else:  # self.preload == PreloadTypes.NEVER
            item = self.load_one(index)

        return item


class LotsOfHdf5Files(Dataset):
    def __init__(self, data_dir, io, data_label=None, target_label=None, labels=None, partition=None,
                 transform=None, max_loaded_files=0, preload=PreloadTypes.LAZY):
        if None is not partition:
            data_dir = os.path.join(data_dir, partition)
        filenames = glob.glob(os.path.join(data_dir, '*.hdf5'))

        def _hdf5_creator(filename):
            try:
                return Hdf5File(filename, io,
                                data_label=data_label,
                                target_label=target_label,
                                labels=labels,
                                transform=transform,
                                preload=preload)
            except (OSError, KeyError) as e:
                eprint('Unable to open {}: {}'.format(filename, str(e)))
                return None

        self.files = [hdf5_file for hdf5_file in filter(lambda obj: obj is not None,
                                                        threaded_parallel(_hdf5_creator, filenames))]
        self.cum_num_items = np.cumsum([len(f) for f in self.files])
        self.current_file_idx = 0
        self.max_loaded_files = max_loaded_files

    def __len__(self):
        if len(self.cum_num_items) > 0:
            return self.cum_num_items[-1]
        return 0

    def __getitem__(self, index):
        file_index = np.searchsorted(self.cum_num_items, index, side='right')
        relative_index = index - self.cum_num_items[file_index] if file_index > 0 else index
        item = self.files[file_index][relative_index]
        loaded_file_indexes = [i for i, f in enumerate(self.files) if f.is_loaded()]
        if len(loaded_file_indexes) > self.max_loaded_files:
            file_index_to_unload = np.random.choice(loaded_file_indexes)
            self.files[file_index_to_unload].unload()
        return item
