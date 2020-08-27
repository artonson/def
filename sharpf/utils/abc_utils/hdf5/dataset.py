import glob
import logging
import os

import h5py
import numpy as np
import torch
from torch.utils.data import Dataset

from ..py_utils.parallel import threaded_parallel

log = logging.getLogger(__name__)


class Hdf5File(Dataset):
    def __init__(self, filename, io, data_label=None, target_label=None, labels=None, preload=True,
                 transform=None):
        """Represents HDF5 dataset contained in a single HDF5 file.

        :param filename: name of the file
        :param io: HDF5IO object serving as a I/O interface to the HDF5 data files
        :param data_label: string label in HDF5 dataset corresponding to data to train from
        :param target_label: string label in HDF5 dataset corresponding to targets
        :param labels: a list of HDF5 dataset labels to read off the file ('*' for ALL keys)
        :param preload: if True, data is read off disk in constructor; otherwise load lazily
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

        if preload:
            self.reload()

    def _get_length(self, hdf5_file):
        try:
            num_items = self.io.length(hdf5_file)
        except KeyError:
            log.error('File {} is not compatible with Hdf5File I/O interface {}'.format(
                self.filename, str(self.io.__class__)))
            num_items = 0
        return num_items

    def __len__(self):
        return self.num_items

    def __getitem__(self, index):
        if not self.is_loaded():
            self.reload()

        item = {label: self.items[label][index]
                for label in self.labels}

        data = None
        if None is not self.data_label:
            data = torch.from_numpy(self.items[self.data_label][index])

        target = None
        if None is not self.target_label:
            target = torch.from_numpy(self.items[self.target_label][index])

        if self.transform is not None:
            data, target = self.transform(data, target)

        item.update({self.data_label: data,
                     self.target_label: target})

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
    def __init__(self, io, data_dir=None, filenames=None, data_label=None, target_label=None, labels=None,
                 partition=None,
                 transform=None, max_loaded_files=0):
        assert (data_dir is not None) != (filenames is not None), "either provide only data_dir or only filenames arg"

        if data_dir is not None:
            if partition is not None:
                data_dir = os.path.join(data_dir, partition)
            filenames = sorted(glob.glob(os.path.join(data_dir, '*.hdf5')))
        else:
            filenames = sorted(list(filenames))

        for filename in filenames:
            assert os.path.exists(filename)

        def _hdf5_creator(filename):
            try:
                return Hdf5File(filename, io, data_label, target_label, labels=labels,
                                transform=transform, preload=False)
            except (OSError, KeyError) as e:
                log.error('Unable to open {}: {}'.format(filename, str(e)))
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


class DepthDataset(LotsOfHdf5Files):

    def __init__(self, task, io, data_dir=None, filenames=None, target_label=None, labels=None,
                 partition=None, transform=None, max_loaded_files=0):
        super().__init__(io=io, data_dir=data_dir, filenames=filenames,
                         data_label='image', target_label=target_label,
                         labels=labels,
                         partition=partition,
                         transform=None,
                         max_loaded_files=max_loaded_files)
        self.task = task
        self.transform = transform

    def __getitem__(self, index):

        # TODO it is possible to convert all these processing steps to transform objects, and then delete this class

        item = super().__getitem__(index)
        data, target = item['image'], item['distances']

        mask_1 = (np.copy(data) != 0.0).astype(float)  # mask for object
        mask_2 = np.where(data == 0)  # mask for background
        dist_new = np.copy(target)
        dist_mask = dist_new * mask_1  # select object points
        dist_mask[mask_2] = 1.0  # background points has max distance to sharp features

        if self.transform is not None:
            data, target = self.transform(data, target)

        output = {'points': torch.FloatTensor(data).unsqueeze(0)}

        if 'segmentation' in self.task:
            close_to_sharp = np.array((dist_mask != np.nan) & (dist_mask < 1.)).astype(float)
            output['close_to_sharp_mask'] = torch.FloatTensor(close_to_sharp).unsqueeze(0)
        if 'regression' in self.task:
            output['distances'] = torch.FloatTensor(dist_mask).unsqueeze(0)
        assert len(output.keys()) > 1, f"{self.task}, {output.keys()}"

        return output
