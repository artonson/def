import collections

import h5py
import numpy as np
# pytorch==1.2.0
import torch
from torch.utils.data._utils.collate import default_collate


class HDF5Dataset:
    def __init__(self, name, dtype):
        self.name = name
        self.dtype = dtype

    @property
    def is_fixed_len(self):
        return True

    def set(self, hdf5_file, data, compression=None):
        hdf5_file.create_dataset(self.name, data=data, dtype=self.dtype, compression=compression)

    def get(self, hdf5_file):
        return np.array(hdf5_file[self.name]).astype(self.dtype)

    def get_one(self, hdf5_file, index):
        return hdf5_file[self.name][index]


class Float64(HDF5Dataset):
    def __init__(self, name):
        super().__init__(name, dtype=np.float64)


class Bool(HDF5Dataset):
    def __init__(self, name):
        super().__init__(name, dtype=np.bool)


class Int8(HDF5Dataset):
    def __init__(self, name):
        super().__init__(name, dtype=np.int8)


class Int32(HDF5Dataset):
    def __init__(self, name):
        super().__init__(name, dtype=np.int32)


class AsciiString(HDF5Dataset):
    def __init__(self, name):
        super().__init__(name, dtype=h5py.string_dtype(encoding='ascii'))

    def set(self, hdf5_file, data, compression=None):
        hdf5_file.create_dataset(self.name, data=np.string_(data), dtype=self.dtype, compression=compression)


class VariableLenDataset(HDF5Dataset):
    @property
    def is_fixed_len(self):
        return False

    def set(self, hdf5_file, data, compression=None):
        dataset = hdf5_file.create_dataset(self.name, shape=(len(data), ), dtype=self.dtype, compression=compression)
        for i, item in enumerate(data):
            dataset[i] = item


class VarInt32(VariableLenDataset):
    def __init__(self, name):
        super().__init__(name, dtype=h5py.special_dtype(vlen=np.int32))


class VarFloat64(VariableLenDataset):
    def __init__(self, name):
        super().__init__(name, dtype=h5py.special_dtype(vlen=np.float64))


class VarBool(VariableLenDataset):
    def __init__(self, name):
        super().__init__(name, dtype=h5py.special_dtype(vlen=np.bool))


class HDF5IO:
    def __init__(self, datasets, len_label, compression=None):
        self.datasets = datasets
        self.len_label = len_label
        self.compression = compression

    def write(self, hdf5_file, label, value):
        dataset = self.datasets[label]
        dataset.set(hdf5_file, value, compression=self.compression)

    def read(self, hdf5_file, label):
        dataset = self.datasets[label]
        return dataset.get(hdf5_file)

    def read_one(self, hdf5_file, label, index):
        dataset = self.datasets[label]
        return dataset.get_one(hdf5_file, index)

    def length(self, hdf5_file):
        return len(hdf5_file[self.len_label])


def collate_mapping_with_io(batch_mapping, io):
    assert isinstance(batch_mapping[0], collections.abc.Mapping)

    def _batch_keys_subset(batch_mapping, keys):
        return [{key: mapping[key] for key in keys} for mapping in batch_mapping]

    fixlen_keys = [key for key, value in io.datasets.items()
                   if value.is_fixed_len and key in batch_mapping[0]]
    fixlen_collatable = _batch_keys_subset(batch_mapping, fixlen_keys)
    fixlen_collated = default_collate(fixlen_collatable)

    varlen_keys = [key for key, value in io.datasets.items()
                   if not value.is_fixed_len and key in batch_mapping[0]]
    varlen_collatable = _batch_keys_subset(batch_mapping, varlen_keys)
    varlen_collated = collate_varlen_to_list(varlen_collatable)

    return {**fixlen_collated, **varlen_collated}


def collate_varlen_to_list(batch):
    # creates a list of variable-length sequences
    return {key: [d[key] for d in batch]
            for key in batch[0]}


def select_items_by_predicates(batch, true_keys=None, false_keys=None):
    """Selects sub-batch where item[key] == True for each key in true_keys
    and item[key] == False for each key in false_keys"""
    any_key = next(iter(batch.keys()))
    batch_size = len(batch[any_key])

    if None is not true_keys:
        true_mask = torch.stack([batch[key] for key in true_keys]).all(axis=0)
    else:
        true_mask = torch.ones(batch_size).bool()

    if None is not false_keys:
        false_mask = torch.stack([~batch[key] for key in false_keys]).all(axis=0)
    else:
        false_mask = torch.ones(batch_size).bool()

    selected_idx = torch.where(true_mask * false_mask)[0]
    filtered_batch = {}
    for key, value in batch.items():
        if isinstance(value, torch.Tensor):
            filtered_batch[key] = value[selected_idx]
        elif isinstance(value, list):
            filtered_batch[key] = [value[i] for i in selected_idx]

    return filtered_batch
