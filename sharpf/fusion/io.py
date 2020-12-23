from functools import partial
import glob
import os
import re

import h5py
import numpy as np

import sharpf.utils.abc_utils.hdf5.io_struct as io_struct


UnlabeledPointCloudIO = io_struct.HDF5IO({
    'points': io_struct.Float64('points'),
    'indexes_in_whole': io_struct.Int32('indexes_in_whole'),
    'distances': io_struct.Float64('distances'),
    'item_id': io_struct.AsciiString('item_id'),
},
    len_label='has_sharp',
    compression='lzf')

ComparisonsIO = io_struct.HDF5IO({
    'points': io_struct.VarFloat64('points'),
    'indexes_in_whole': io_struct.VarInt32('indexes_in_whole'),
    'distances': io_struct.VarFloat64('distances'),
    'item_id': io_struct.AsciiString('item_id'),
    'voronoi': io_struct.VarFloat64('voronoi'),
    'ecnet': io_struct.VarFloat64('ecnet'),
    'sharpness': io_struct.VarFloat64('sharpness'),
},
    len_label='points',
    compression='lzf')


def convert_npylist_to_hdf5(input_dir, output_filename):
    PointPatchPredictionsIO = io_struct.HDF5IO(
        {'distances': io_struct.VarFloat64('distances')},
        len_label='distances',
        compression='lzf')

    def save_predictions(patches, filename):
        collate_fn = partial(io_struct.collate_mapping_with_io, io=PointPatchPredictionsIO)
        patches = collate_fn(patches)
        with h5py.File(filename, 'w') as f:
            PointPatchPredictionsIO.write(f, 'distances', patches['distances'])

    def get_num(basename):
        match = re.match('^test_(\d+)\.npy$', basename)
        return int(match.groups()[0])

    datafiles = glob.glob(os.path.join(input_dir, '*.npy'))
    datafiles.sort(key=lambda name: get_num(os.path.basename(name)))
    patches = [{'distances': np.load(f)} for f in datafiles]
    save_predictions(patches, output_filename)


def save_full_model_predictions(points, predictions, filename):
    PointPatchPredictionsIO = io_struct.HDF5IO(
        {'points': io_struct.Float64('points'),
         'distances': io_struct.Float64('distances')},
        len_label='distances',
        compression='lzf')
    with h5py.File(filename, 'w') as f:
        PointPatchPredictionsIO.write(f, 'points', [points])
        PointPatchPredictionsIO.write(f, 'distances', [predictions])

