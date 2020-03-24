import os
import glob
from collections import Callable

import h5py
import numpy as np
import torch
from torch.utils.data import Dataset

from sharpf.utils.common import eprint
from sharpf.utils.matrix_torch import random_3d_rotation_and_scale


class Random3DRotationAndScale(Callable):
    def __init__(self, scale_range):
        self.scale_range = scale_range

    def __call__(self, data):
        data = torch.cat((data, torch.ones(len(data), 1)), dim=1)
        transform = random_3d_rotation_and_scale(self.scale_range)
        data = torch.mm(data, transform)
        return data[:, :-1]


class Hdf5File(Dataset):
    def __init__(self, filename, io, data_label, target_label, labels=None, preload=True,
                 transform=None, target_transform=None):
        """Represents HDF5 dataset contained in a single HDF5 file.

        :param filename: name of the file
        :param io: HDF5IO object serving as a I/O interface to the HDF5 data files
        :param data_label: string label in HDF5 dataset corresponding to data to train from
        :param target_label: string label in HDF5 dataset corresponding to targets
        :param labels: a list of HDF5 dataset labels to read off the file
        :param preload: if True, data is read off disk in constructor; otherwise load lazily
        :param transform: callable implementing data transform (e.g., adding noise)
        :param target_transform: callable implementing target transform
        """
        self.filename = os.path.normpath(os.path.realpath(filename))
        self.data_label = data_label
        self.target_label = target_label
        self.default_labels = {data_label, target_label}
        self.transform = transform
        self.target_transform = target_transform
        self.items = None  # this is where the data internally is read to
        self.io = io
        if preload:
            self.reload()

        with h5py.File(self.filename, 'r') as f:
            self.num_items = self._get_length(f)
            if labels == '*':
                labels = set(f.keys())

        self.labels = list(self.default_labels.union(set(labels)))

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
        if not self.is_loaded():
            self.reload()

        data_items, target_items = self.items[self.data_label], \
                                   self.items[self.target_label]
        data, target = torch.from_numpy(data_items[index]), \
                       torch.from_numpy(target_items[index])

        if self.transform is not None:
            data = self.transform(data)

        if self.target_transform is not None:
            target = self.target_transform(target)

        item = {label: self.items[label][index]
                for label in self.labels}
        item.update({
            self.data_label: data,
            self.target_label: target,
        })
        return item

    def reload(self):
        with h5py.File(self.filename, 'r') as f:
            self.items = {label: self.io.read(f, label)
                          for label in self.labels}
            self.num_items = self._get_length(f)

    def is_loaded(self):
        return None is not self.items

    def unload(self):
        self.items = None


class LotsOfHdf5Files(Dataset):
    def __init__(self, data_dir, io, data_label, target_label, labels=None, partition=None,
                 transform=None, target_transform=None, max_loaded_files=0):
        self.data_label = data_label
        self.target_label = target_label
        if None is not partition:
            data_dir = os.path.join(data_dir, partition)
        filenames = glob.glob(os.path.join(data_dir, '*.hdf5'))
        self.files = [Hdf5File(filename, io, data_label, target_label, labels=labels,
                               transform=transform, target_transform=target_transform, preload=False)
                      for filename in filenames]
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
