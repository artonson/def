import collections

import h5py
import numpy as np
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
        try:
            return np.array(hdf5_file[self.name]).astype(self.dtype)
        except ValueError:
            print("NAME", self.name)
            print(hdf5_file[self.name])
            raise ValueError


class Float32(HDF5Dataset):
    def __init__(self, name):
        super().__init__(name, dtype=np.float32)


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


class VarInt32(VariableLenDataset):
    def __init__(self, name):
        super().__init__(name, dtype=h5py.special_dtype(vlen=np.int32))

    def set(self, hdf5_file, data, compression=None):
        dataset = hdf5_file.create_dataset(self.name, shape=(len(data),), dtype=self.dtype, compression=compression)
        for i, item in enumerate(data):
            dataset[i] = item


class VarFloat64(VariableLenDataset):
    def __init__(self, name):
        super().__init__(name, dtype=h5py.special_dtype(vlen=np.float64))


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

    def length(self, hdf5_file):
        return len(hdf5_file[self.len_label])


def collate_mapping_with_io(batch_mapping, io):
    assert isinstance(batch_mapping[0], collections.abc.Mapping)

    def _batch_keys_subset(batch_mapping, keys):
        return [{key: mapping[key] for key in keys} for mapping in batch_mapping]

    fixlen_keys = [key for key, value in io.datasets.items() if value.is_fixed_len]
    fixlen_collatable = _batch_keys_subset(batch_mapping, fixlen_keys)
    fixlen_collated = default_collate(fixlen_collatable)

    varlen_keys = [key for key, value in io.datasets.items() if not value.is_fixed_len]
    varlen_collatable = _batch_keys_subset(batch_mapping, varlen_keys)
    varlen_collated = collate_varlen_to_list(varlen_collatable)

    return {**fixlen_collated, **varlen_collated}


def collate_varlen_to_list(batch):
    # creates a list of variable-length sequences
    return {key: [d[key] for d in batch]
            for key in batch[0]}
