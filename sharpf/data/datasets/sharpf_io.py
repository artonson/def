from functools import partial

import h5py
import numpy as np

import sharpf.data.datasets.hdf5_io as io


# TODO turn this variable into a singleton
DepthIO = io.HDF5IO({
        'points': io.Float64('points'),
        'normals': io.Float64('normals'),
        'distances': io.Float64('distances'),
        'directions': io.Float64('directions'),
        'item_id': io.AsciiString('item_id'),
        'orig_vert_indices': io.VarInt32('orig_vert_indices'),
        'orig_face_indexes': io.VarInt32('orig_face_indexes'),
        'has_sharp': io.Bool('has_sharp'),
        'num_sharp_curves': io.Int8('num_sharp_curves'),
        'num_surfaces': io.Int8('num_surfaces'),
    },
    len_label='has_sharp',
    compression='lzf')


def save_point_patches(patches, filename):
    # turn a list of dicts into a dict of torch tensors:
    # default_collate([{'a': 'str1', 'x': np.random.normal()}, {'a': 'str2', 'x': np.random.normal()}])
    # Out[26]: {'a': ['str1', 'str2'], 'x': tensor([0.4252, 0.1414], dtype=torch.float64)}
    collate_fn = partial(io.collate_mapping_with_io, io=DepthIO)
    patches = collate_fn(patches)

    with h5py.File(filename, 'w') as f:
        for key in ['points', 'normals', 'distances', 'directions']:
            DepthIO.write(f, key, patches[key].numpy())
        DepthIO.write(f, 'item_id', patches['item_id'])
        DepthIO.write(f, 'orig_vert_indices', patches['orig_vert_indices'])
        DepthIO.write(f, 'orig_face_indexes', patches['orig_face_indexes'])
        DepthIO.write(f, 'has_sharp', patches['has_sharp'].numpy().astype(np.bool))
        DepthIO.write(f, 'num_sharp_curves', patches['num_sharp_curves'].numpy())
        DepthIO.write(f, 'num_surfaces', patches['num_surfaces'].numpy())
