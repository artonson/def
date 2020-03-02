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
        transform = random_3d_rotation_and_scale(self.scale_range)
        return torch.mm(data, transform)


class Hdf5File(Dataset):
    def __init__(self, filename, data_label, target_label, preload=True,
                 transform=None, target_transform=None):
        self.filename = os.path.normpath(filename)
        self.data_label = data_label
        self.target_label = target_label
        self.transform = transform
        self.target_transform = target_transform
        self.data, self.target = None, None
        if preload:
            self.reload()
        else:
            with h5py.File(self.filename, 'r') as f:
                try:
                    self.num_items = len(f['has_sharp'])
                except KeyError:
                    eprint('File {} is not compatible with Hdf5File interface'.format(self.filename))
                    self.num_items = 0

    def __len__(self):
        return self.num_items

    def __getitem__(self, index):
        if not self.is_loaded():
            self.reload()

        data, target = self.data[index], self.target[index]

        if self.transform is not None:
            data = self.transform(data)

        if self.target_transform is not None:
            target = self.target_transform(target)

        return torch.from_numpy(data), torch.from_numpy(target)

    def reload(self):
        with h5py.File(self.filename, 'r') as f:
            self.data = np.array(f[self.data_label]).astype('float32')
            self.target = np.array(f[self.target_label]).astype('float32')
            self.num_items = len(f)

    def is_loaded(self):
        return None is not self.data and None is not self.target

    def unload(self):
        self.data, self.target = None, None


class LotsOfHdf5Files(Dataset):
    def __init__(self, data_dir, data_label, target_label, partition=None,
                 transform=None, target_transform=None, max_loaded_files=0):
        self.data_label = data_label
        self.target_label = target_label
        if None is not partition:
            data_dir = os.path.join(data_dir, partition)
        filenames = glob.glob(os.path.join(data_dir, '*.hdf5'))
        self.files = [Hdf5File(filename, data_label, target_label,
                               transform=transform, target_transform=target_transform, preload=False)
                      for filename in filenames]
        self.cum_num_items = np.cumsum([len(f) for f in self.files])
        self.current_file_idx = 0
        self.max_loaded_files = max_loaded_files

    def __len__(self):
        return self.cum_num_items[-1]

    def __getitem__(self, index):
        file_index = np.searchsorted(self.cum_num_items, index, side='right')
        relative_index = index - self.cum_num_items[file_index] if file_index > 0 else index
        data, target = self.files[file_index][relative_index]
        loaded_file_indexes = [i for i, f in enumerate(self.files) if f.is_loaded()]
        if len(loaded_file_indexes) > self.max_loaded_files:
            file_index_to_unload = np.random.choice(loaded_file_indexes)
            self.files[file_index_to_unload].unload()
        return data, target
