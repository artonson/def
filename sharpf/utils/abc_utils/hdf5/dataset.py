import glob
import logging
import os

import h5py
import numpy as np
from torch.utils.data import Dataset

from ..py_utils.parallel import threaded_parallel

log = logging.getLogger(__name__)


class Hdf5File(Dataset):
    def __init__(self, filename, io, labels, transform=None, preload=True, return_index=False):
        """Represents HDF5 dataset contained in a single HDF5 file.

        :param filename: name of the file
        :param io: HDF5IO object serving as a I/O interface to the HDF5 data files
        :param labels: a list of HDF5 dataset labels to read off the file ('*' for ALL keys)
        :param transform: callable implementing data + target transform (e.g., adding noise)
        :param preload: if True, data is read off disk in constructor; otherwise load lazily
        :param return_index: if True, return index of the element too
        """
        self.filename = os.path.normpath(os.path.realpath(filename))

        self.transform = transform
        self.items = None  # this is where the data internally is read to
        self.io = io
        self.return_index = return_index

        with h5py.File(self.filename, 'r') as f:
            self.num_items = self._get_length(f)
            if labels == '*':
                self.labels = set(f.keys())
            else:
                for label in labels:
                    assert label in f.keys()
                self.labels = set(labels)
            if return_index:
                assert 'index' not in self.labels

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

        item = {label: self.items[label][index] for label in self.labels}
        if self.return_index:
            item['index'] = index

        if self.transform is not None:
            item = self.transform(item)

        return item

    def reload(self):
        with h5py.File(self.filename, 'r') as f:
            self.items = {label: self.io.read(f, label) for label in self.labels}

    def is_loaded(self):
        return self.items is not None

    def unload(self):
        self.items = None


class LotsOfHdf5Files(Dataset):
    def __init__(self, io, labels, filenames=None, data_dir=None, partition=None, transform=None, max_loaded_files=0,
                 return_index=False):
        assert (data_dir is not None) != (filenames is not None), "either provide only data_dir or only filenames arg"
        assert max_loaded_files >= 0

        if data_dir is not None:
            if partition is not None:
                data_dir = os.path.join(data_dir, partition)
            assert os.path.exists(data_dir)
            filenames = sorted(glob.glob(os.path.join(data_dir, '*.hdf5')))
        else:
            filenames = sorted(list(filenames))
            for filename in filenames:
                assert os.path.exists(filename)

        def _hdf5_creator(filename):
            try:
                # return_index=False because we are not interested in a relative index
                return Hdf5File(filename, io, labels, transform, preload=False, return_index=False)
            except OSError as e:
                raise OSError(f"Unable to open {filename}") from e
            except KeyError as e:
                raise KeyError(f"Unable to open {filename}") from e

        self.files = [hdf5_file for hdf5_file in filter(lambda obj: obj is not None,
                                                        threaded_parallel(_hdf5_creator, filenames))]
        self.cum_num_items = np.cumsum([len(f) for f in self.files])
        self.current_file_idx = 0
        self.max_loaded_files = max_loaded_files
        self.return_index = return_index

    def __len__(self):
        if len(self.cum_num_items) > 0:
            return self.cum_num_items[-1]
        return 0

    def __getitem__(self, index):
        file_index = np.searchsorted(self.cum_num_items, index, side='right')
        relative_index = index - self.cum_num_items[file_index] if file_index > 0 else index

<<<<<<< HEAD

class DepthDataset(LotsOfHdf5Files):

    def __init__(self, io, data_dir, data_label, target_label, task, partition=None,
                 transform=None, normalisation=['quantile', 'standartize'], max_loaded_files=0):
        super().__init__(data_dir=data_dir, io=io,
                         data_label=data_label, target_label=target_label,
                         labels=['item_id', 'camera_pose', 'mesh_scale'],
                         partition=partition,
                         transform=transform,
                         max_loaded_files=max_loaded_files)
        self.data_dir = data_dir
        self.task = task
        self.quality = self._get_quantity()
        self.normalisation = normalisation

    def _get_quantity(self):
        data_dir_split = self.data_dir.split('_')
        if 'high' in data_dir_split:
            return 'high'
        elif 'low' in data_dir_split:
            return 'low'
        elif 'med' in data_dir_split:
            return 'med'

    def quantile_normalize(self, data):
        # mask -> min shift -> quantile

        norm_data = np.copy(data)
        mask_obj = np.where(norm_data != 0)
        mask_back = np.where(norm_data == 0)
        norm_data[mask_back] = norm_data.max() + 1.0  # new line
        norm_data -= norm_data[mask_obj].min()

        norm_data /= high_res_quantile

        return norm_data

    def standartize(self, data):
        # zero mean, unit variance

        standart_data = np.copy(data)
        standart_data -= np.mean(standart_data)
        std = np.linalg.norm(standart_data, axis=1).max()
        if std > 0:
            standart_data /= std

        return standart_data
=======
        file = self.files[file_index]
>>>>>>> origin/pl_hydra

        loaded_file_indexes = [i for i, f in enumerate(self.files) if f.is_loaded()]
        if self.max_loaded_files > 0:
            if file_index not in loaded_file_indexes and len(loaded_file_indexes) == self.max_loaded_files:
                file_index_to_unload = np.random.choice(loaded_file_indexes)
                self.files[file_index_to_unload].unload()

        item = file[relative_index]
        if self.return_index:
            assert 'index' not in item
            item['index'] = index

<<<<<<< HEAD
        item = super().__getitem__(index)
        data, target = item['image'], item['distances']
        mask_1 = (np.copy(data) != 0.0).astype(float)  # mask for object
        mask_2 = np.where(data == 0)  # mask for background

        if 'quantile' in self.normalisation:
            data = self.quantile_normalize(data)
        if 'standartize' in self.normalisation:
            data = self.standartize(data)

        dist_new = np.copy(target)
        dist_mask = dist_new * mask_1  # select object points
        dist_mask[mask_2] = 1.0  # background points has max distance to sharp features
        close_to_sharp = np.array((dist_mask != np.nan) & (dist_mask < 1.)).astype(float)

        output = {}

        if self.task == 'two-heads':
            # regression + segmentation (or two-head network) has to targets:
            # distance field and segmented close-to-sharp region of the object
            target = torch.cat([torch.FloatTensor(dist_mask).unsqueeze(0), torch.FloatTensor(close_to_sharp).unsqueeze(0)], dim=0)
            output['distance_and_close_to_sharp'] = target
        if self.task == 'segmentation':
            target = torch.FloatTensor(close_to_sharp).unsqueeze(0)
            output['close_to_sharp_mask'] = target
        elif self.task == 'regression':
            target = torch.FloatTensor(dist_mask).unsqueeze(0)
            output['distance_to_sharp'] = target

        data = torch.FloatTensor(data).unsqueeze(0)
        output['image'] = data
        output['item_id'] = item['item_id']
        output['camera_pose'] = item['camera_pose']
        output['mesh_scale'] = item['mesh_scale']

        return output
=======
        return item
>>>>>>> origin/pl_hydra
