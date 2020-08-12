from functools import partial

import h5py
import numpy as np

import sharpf.utils.abc_utils.hdf5.io_struct as io


# TODO turn this variable into a singleton
PointCloudIO = io.HDF5IO({
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
        'has_smell_coarse_surfaces_by_num_faces': io.Bool('has_smell_coarse_surfaces_by_num_faces'),
        'has_smell_coarse_surfaces_by_angles': io.Bool('has_smell_coarse_surfaces_by_angles'),
        'has_smell_deviating_resolution': io.Bool('has_smell_deviating_resolution'),
        'has_smell_sharpness_discontinuities': io.Bool('has_smell_sharpness_discontinuities'),
        'has_smell_bad_face_sampling': io.Bool('has_smell_bad_face_sampling'),
        'has_smell_mismatching_surface_annotation': io.Bool('has_smell_mismatching_surface_annotation'),
    },
    len_label='has_sharp',
    compression='lzf')


def save_point_patches(patches, filename):
    # turn a list of dicts into a dict of torch tensors:
    # default_collate([{'a': 'str1', 'x': np.random.normal()}, {'a': 'str2', 'x': np.random.normal()}])
    # Out[26]: {'a': ['str1', 'str2'], 'x': tensor([0.4252, 0.1414], dtype=torch.float64)}
    collate_fn = partial(io.collate_mapping_with_io, io=PointCloudIO)
    patches = collate_fn(patches)

    with h5py.File(filename, 'w') as f:
        for key in ['points', 'normals', 'distances', 'directions']:
            PointCloudIO.write(f, key, patches[key].numpy())
        PointCloudIO.write(f, 'item_id', patches['item_id'])
        PointCloudIO.write(f, 'orig_vert_indices', patches['orig_vert_indices'])
        PointCloudIO.write(f, 'orig_face_indexes', patches['orig_face_indexes'])
        PointCloudIO.write(f, 'has_sharp', patches['has_sharp'].numpy().astype(np.bool))
        PointCloudIO.write(f, 'num_sharp_curves', patches['num_sharp_curves'].numpy())
        PointCloudIO.write(f, 'num_surfaces', patches['num_surfaces'].numpy())
        has_smell_keys = [key for key in PointCloudIO.datasets.keys()
                          if key.startswith('has_smell')]
        for key in has_smell_keys:
            PointCloudIO.write(f, key, patches[key].numpy().astype(np.bool))


DepthMapIO = io.HDF5IO({
        'image': io.Float64('image'),
        'normals': io.Float64('normals'),
        'distances': io.Float64('distances'),
        'directions': io.Float64('directions'),
        'item_id': io.AsciiString('item_id'),
        'orig_vert_indices': io.VarInt32('orig_vert_indices'),
        'orig_face_indexes': io.VarInt32('orig_face_indexes'),
        'has_sharp': io.Bool('has_sharp'),
        'num_sharp_curves': io.Int8('num_sharp_curves'),
        'num_surfaces': io.Int8('num_surfaces'),
        'camera_pose': io.Float64('camera_pose'),
        'mesh_scale': io.Float64('mesh_scale')
    },
    len_label='has_sharp',
    compression='lzf')


def save_depth_maps(patches, filename):
    # turn a list of dicts into a dict of torch tensors:
    # default_collate([{'a': 'str1', 'x': np.random.normal()}, {'a': 'str2', 'x': np.random.normal()}])
    # Out[26]: {'a': ['str1', 'str2'], 'x': tensor([0.4252, 0.1414], dtype=torch.float64)}
    collate_fn = partial(io.collate_mapping_with_io, io=DepthMapIO)
    patches = collate_fn(patches)

    with h5py.File(filename, 'w') as f:
        for key in ['image', 'normals', 'distances', 'directions']:
            DepthMapIO.write(f, key, patches[key].numpy())
        DepthMapIO.write(f, 'item_id', patches['item_id'])
        DepthMapIO.write(f, 'orig_vert_indices', patches['orig_vert_indices'])
        DepthMapIO.write(f, 'orig_face_indexes', patches['orig_face_indexes'])
        DepthMapIO.write(f, 'has_sharp', patches['has_sharp'].numpy().astype(np.bool))
        DepthMapIO.write(f, 'num_sharp_curves', patches['num_sharp_curves'].numpy())
        DepthMapIO.write(f, 'num_surfaces', patches['num_surfaces'].numpy())
        DepthMapIO.write(f, 'camera_pose', patches['camera_pose'].numpy())
        DepthMapIO.write(f, 'mesh_scale', patches['mesh_scale'].numpy())
