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
        'mesh_scale': io.Float64('mesh_scale'),
        'has_smell_coarse_surfaces_by_num_faces': io.Bool('has_smell_coarse_surfaces_by_num_faces'),
        'has_smell_coarse_surfaces_by_angles': io.Bool('has_smell_coarse_surfaces_by_angles'),
        'has_smell_deviating_resolution': io.Bool('has_smell_deviating_resolution'),
        'has_smell_sharpness_discontinuities': io.Bool('has_smell_sharpness_discontinuities'),
        'has_smell_bad_face_sampling': io.Bool('has_smell_bad_face_sampling'),
        'has_smell_mismatching_surface_annotation': io.Bool('has_smell_mismatching_surface_annotation'),
        'has_smell_raycasting_background': io.Bool('has_smell_raycasting_background'),
        'has_smell_depth_discontinuity': io.Bool('has_smell_depth_discontinuity'),
        'has_smell_mesh_self_intersections': io.Bool('has_smell_mesh_self_intersections'),
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
        has_smell_keys = [key for key in DepthMapIO.datasets.keys()
                          if key.startswith('has_smell')]
        for key in has_smell_keys:
            DepthMapIO.write(f, key, patches[key].numpy().astype(np.bool))


WholePointCloudIO = io.HDF5IO({
        'points': io.VarFloat64('points'),
        'normals': io.VarFloat64('normals'),
        'distances': io.VarFloat64('distances'),
        'directions': io.VarFloat64('directions'),
        'indexes_in_whole': io.VarInt32('indexes_in_whole'),
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


def save_whole_patches(patches, filename):
    # turn a list of dicts into a dict of torch tensors:
    # default_collate([{'a': 'str1', 'x': np.random.normal()}, {'a': 'str2', 'x': np.random.normal()}])
    # Out[26]: {'a': ['str1', 'str2'], 'x': tensor([0.4252, 0.1414], dtype=torch.float64)}
    collate_fn = partial(io.collate_mapping_with_io, io=WholePointCloudIO)
    patches = collate_fn(patches)

    with h5py.File(filename, 'w') as f:
        for key in ['points', 'normals', 'distances', 'directions', 'indexes_in_whole']:
            WholePointCloudIO.write(f, key, patches[key])
        WholePointCloudIO.write(f, 'item_id', patches['item_id'])
        WholePointCloudIO.write(f, 'orig_vert_indices', patches['orig_vert_indices'])
        WholePointCloudIO.write(f, 'orig_face_indexes', patches['orig_face_indexes'])
        WholePointCloudIO.write(f, 'has_sharp', patches['has_sharp'].numpy().astype(np.bool))
        WholePointCloudIO.write(f, 'num_sharp_curves', patches['num_sharp_curves'].numpy())
        WholePointCloudIO.write(f, 'num_surfaces', patches['num_surfaces'].numpy())
        has_smell_keys = [key for key in WholePointCloudIO.datasets.keys()
                          if key.startswith('has_smell')]
        for key in has_smell_keys:
            WholePointCloudIO.write(f, key, patches[key].numpy().astype(np.bool))


WholeDepthMapIO = io.HDF5IO({
        'image': io.Float64('image'),
        'normals': io.Float64('normals'),
        'distances': io.Float64('distances'),
        'directions': io.Float64('directions'),
        'indexes_in_whole': io.Int32('indexes_in_whole'),
        'item_id': io.AsciiString('item_id'),
        'orig_vert_indices': io.VarInt32('orig_vert_indices'),
        'orig_face_indexes': io.VarInt32('orig_face_indexes'),
        'has_sharp': io.Bool('has_sharp'),
        'num_sharp_curves': io.Int8('num_sharp_curves'),
        'num_surfaces': io.Int8('num_surfaces'),
        'camera_pose': io.Float64('camera_pose'),
        'mesh_scale': io.Float64('mesh_scale'),
        'has_smell_coarse_surfaces_by_num_faces': io.Bool('has_smell_coarse_surfaces_by_num_faces'),
        'has_smell_coarse_surfaces_by_angles': io.Bool('has_smell_coarse_surfaces_by_angles'),
        'has_smell_deviating_resolution': io.Bool('has_smell_deviating_resolution'),
        'has_smell_sharpness_discontinuities': io.Bool('has_smell_sharpness_discontinuities'),
        'has_smell_bad_face_sampling': io.Bool('has_smell_bad_face_sampling'),
        'has_smell_mismatching_surface_annotation': io.Bool('has_smell_mismatching_surface_annotation'),
        'has_smell_raycasting_background': io.Bool('has_smell_raycasting_background'),
        'has_smell_depth_discontinuity': io.Bool('has_smell_depth_discontinuity'),
        'has_smell_mesh_self_intersections': io.Bool('has_smell_mesh_self_intersections'),
    },
    len_label='has_sharp',
    compression='lzf')


def save_whole_images(patches, filename):
    # turn a list of dicts into a dict of torch tensors:
    # default_collate([{'a': 'str1', 'x': np.random.normal()}, {'a': 'str2', 'x': np.random.normal()}])
    # Out[26]: {'a': ['str1', 'str2'], 'x': tensor([0.4252, 0.1414], dtype=torch.float64)}
    collate_fn = partial(io.collate_mapping_with_io, io=WholeDepthMapIO)
    patches = collate_fn(patches)

    with h5py.File(filename, 'w') as f:
        for key in ['image', 'normals', 'distances', 'directions', 'indexes_in_whole']:
            WholeDepthMapIO.write(f, key, patches[key].numpy())
        WholeDepthMapIO.write(f, 'item_id', patches['item_id'])
        WholeDepthMapIO.write(f, 'orig_vert_indices', patches['orig_vert_indices'])
        WholeDepthMapIO.write(f, 'orig_face_indexes', patches['orig_face_indexes'])
        WholeDepthMapIO.write(f, 'has_sharp', patches['has_sharp'].numpy().astype(np.bool))
        WholeDepthMapIO.write(f, 'num_sharp_curves', patches['num_sharp_curves'].numpy())
        WholeDepthMapIO.write(f, 'num_surfaces', patches['num_surfaces'].numpy())
        WholeDepthMapIO.write(f, 'camera_pose', patches['camera_pose'].numpy())
        WholeDepthMapIO.write(f, 'mesh_scale', patches['mesh_scale'].numpy())
        has_smell_keys = [key for key in WholeDepthMapIO.datasets.keys()
                          if key.startswith('has_smell')]
        for key in has_smell_keys:
            WholeDepthMapIO.write(f, key, patches[key].numpy().astype(np.bool))


AnnotatedViewIO = io.HDF5IO({
    'points': io.Float64('points'),
    'faces': io.VarInt32('faces'),
    'points_alignment': io.Float64('points_alignment'),
    'extrinsics': io.Float64('extrinsics'),
    'intrinsics': io.Float64('intrinsics'),
    'obj_alignment': io.Float64('obj_alignment'),
    'obj_scale': io.Float64('obj_scale'),
    'item_id': io.AsciiString('item_id'),

    'distances': io.Float64('distances'),
    'directions': io.Float64('directions'),
    'orig_vert_indices': io.VarInt32('orig_vert_indices'),
    'orig_face_indexes': io.VarInt32('orig_face_indexes'),
    'has_sharp': io.Bool('has_sharp'),
    'num_sharp_curves': io.Int8('num_sharp_curves'),
    'num_surfaces': io.Int8('num_surfaces'),
    'has_smell_sharpness_discontinuities': io.Bool('has_smell_sharpness_discontinuities'),
},
    len_label='item_id',
    compression='lzf')

def write_annotated_views_to_hdf5(scans, output_filename):
    collate_fn = partial(io.collate_mapping_with_io, io=AnnotatedViewIO)
    scans = collate_fn(scans)

    with h5py.File(output_filename, 'w') as f:
        for key in ['points', 'extrinsics', 'intrinsics', 'points_alignment',
                    'obj_alignment', 'obj_scale', 'distances', 'directions',
                    'num_sharp_curves', 'num_surfaces']:
            AnnotatedViewIO.write(f, key, scans[key].numpy())
        for key in ['item_id', 'faces', 'orig_vert_indices', 'orig_face_indexes']:
            AnnotatedViewIO.write(f, key, scans[key])
        AnnotatedViewIO.write(f, 'has_sharp', scans['has_sharp'].numpy().astype(np.bool))

    print(output_filename)


IO_SPECS = {
    'points': PointCloudIO,
    'images': DepthMapIO,
    'whole_points': WholePointCloudIO,
    'whole_images': WholeDepthMapIO,
    'annotated_view_io': AnnotatedViewIO,
}

SAVE_FNS = {
    'points': save_point_patches,
    'images': save_depth_maps,
    'whole_points': save_whole_patches,
    'whole_images': save_whole_images,
    'annotated_view_io': write_annotated_views_to_hdf5,
}
