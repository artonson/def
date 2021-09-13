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

UnlabeledImageIO = io_struct.HDF5IO({
    'image': io_struct.Float64('image'),
    'distances': io_struct.Float64('distances'),
    'item_id': io_struct.AsciiString('item_id'),
},
    len_label='has_sharp',
    compression='lzf')

ComparisonsIO = io_struct.HDF5IO({
    'points': io_struct.VarFloat64('points'),
    'indexes_in_whole': io_struct.VarInt32('indexes_in_whole'),
    'distances': io_struct.VarFloat64('distances'),
    'distances_for_sh': io_struct.VarFloat64('distances_for_sh'),
    'item_id': io_struct.AsciiString('item_id'),
    'voronoi': io_struct.VarFloat64('voronoi'),
    'ecnet': io_struct.VarFloat64('ecnet'),
    'sharpness': io_struct.VarFloat64('sharpness'),
    'sharpness_seg': io_struct.VarFloat64('sharpness_seg'),
},
    len_label='points',
    compression='lzf')

PointPatchPredictionsIO = io_struct.HDF5IO({
    'distances': io_struct.VarFloat64('distances')
},
    len_label='distances',
    compression='lzf')

ImagePredictionsIO = io_struct.HDF5IO({
    'distances': io_struct.Float64('distances')
},
    len_label='distances',
    compression='lzf')


def save_predictions(patches, filename, io):
    collate_fn = partial(io_struct.collate_mapping_with_io, io=io)
    patches = collate_fn(patches)
    with h5py.File(filename, 'w') as f:
        io.write(f, 'distances', patches['distances'])


def convert_npylist_to_hdf5(input_dir, output_filename, io):
    assert 'distances' in io.datasets

    def get_num(basename):
        match = re.match('^test_(\d+)\.npy$', basename)
        return int(match.groups()[0])

    datafiles = glob.glob(os.path.join(input_dir, '*.npy'))
    datafiles.sort(key=lambda name: get_num(os.path.basename(name)))
    patches = [{'distances': np.load(f)} for f in datafiles]
    save_predictions(patches, output_filename, io)


FusedPredictionsIO = io_struct.HDF5IO(
    {'points': io_struct.Float64('points'),
     'distances': io_struct.Float64('distances')},
    len_label='distances',
    compression='lzf')


def save_full_model_predictions(points, predictions, filename):
    with h5py.File(filename, 'w') as f:
        FusedPredictionsIO.write(f, 'points', [points])
        FusedPredictionsIO.write(f, 'distances', [predictions])


AnnotatedImageIO = io_struct.HDF5IO(
    {'image': io_struct.Float64('image'),
     'distances': io_struct.Float64('distances')},
    len_label='distances',
    compression='lzf')


def save_annotated_images(annotated_images, filename):
    collate_fn = partial(io_struct.collate_mapping_with_io, io=AnnotatedImageIO)
    patches = collate_fn(annotated_images)
    with h5py.File(filename, 'w') as f:
        AnnotatedImageIO.write(f, 'image', patches['image'])
        AnnotatedImageIO.write(f, 'distances', patches['distances'])
