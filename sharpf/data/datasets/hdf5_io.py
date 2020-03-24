import h5py
import numpy as np


class HDF5Dataset:
    def __init__(self, name, dtype):
        self.name = name
        self.dtype = dtype

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


class VarInt32(HDF5Dataset):
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
