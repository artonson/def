import collections

import h5py
import numpy as np
from torch.utils.data._utils.collate import default_collate


class HDF5Dataset:
    def __init__(self, name, dtype):
        self.name = name
        self.dtype = dtype

    @property
    def is_varlen(self):
        return True

    def set(self, hdf5_file, data):
        hdf5_file.create_dataset(self.name, data=data, dtype=self.dtype)

    def get(self, hdf5_file):
        return np.array(hdf5_file[self.name]).astype(self.dtype)


class Float64(HDF5Dataset):
    def __init__(self, name):
        super().__init__(name, dtype=np.float64)


class Bool(HDF5Dataset):
    def __init__(self, name):
        super().__init__(name, dtype=np.bool)


class Int8(HDF5Dataset):
    def __init__(self, name):
        super().__init__(name, dtype=np.int8)


class AsciiString(HDF5Dataset):
    def __init__(self, name):
        super().__init__(name, dtype=h5py.string_dtype(encoding='ascii'))

    def set(self, hdf5_file, data):
        hdf5_file.create_dataset(self.name, data=np.string_(data), dtype=self.dtype)


class VariableLenDataset(HDF5Dataset):
    @property
    def is_varlen(self):
        return False


class VarInt32(VariableLenDataset):
    def __init__(self, name):
        super().__init__(name, dtype=h5py.special_dtype(vlen=np.int32))

    def set(self, hdf5_file, data):
        dataset = hdf5_file.create_dataset(self.name, shape=(len(data), ), dtype=self.dtype)
        for i, item in enumerate(data):
            dataset[i] = item


class HDF5IO:
    def __init__(self, datasets, len_label):
        self.datasets = datasets
        self.len_label = len_label

    def write(self, hdf5_file, label, value):
        dataset = self.datasets[label]
        dataset.set(hdf5_file, value)

    def read(self, hdf5_file, label):
        dataset = self.datasets[label]
        return dataset.get(hdf5_file)

    def length(self, hdf5_file):
        return len(hdf5_file[self.len_label])


def collate_mapping_with_io(batch_mapping, io):
    assert isinstance(batch_mapping[0], collections.abc.Mapping)

    def _batch_keys_subset(batch_mapping, keys):
        return [{key: mapping[key] for key in keys} for mapping in batch_mapping]

    fixlen_keys = [key for key, value in io.dataset.items() if not value.is_varlen]
    fixlen_collatable = _batch_keys_subset(batch_mapping, fixlen_keys)
    fixlen_collated = default_collate(fixlen_collatable)

    varlen_keys = [key for key, value in io.dataset.items() if value.is_varlen]
    varlen_collatable = _batch_keys_subset(batch_mapping, varlen_keys)
    varlen_collated = collate_varlen_to_list(varlen_collatable)

    return [{**fixlen_item, **varlen_item}
            for fixlen_item, varlen_item in zip(fixlen_collated, varlen_collated)]


def collate_varlen_to_list(batch):
    # creates a list of variable-length sequences
    return {key: [d[key] for d in batch]
            for key in batch[0]}
