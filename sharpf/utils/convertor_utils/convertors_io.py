from functools import partial

import h5py

import sharpf.utils.abc_utils.hdf5.io_struct as io_struct
from sharpf.utils.convertor_utils.rangevision_utils import \
    RangeVisionNames as rvn


RangeVisionIO = io_struct.HDF5IO({
    rvn.points: io_struct.VarFloat64(rvn.points),
    rvn.faces: io_struct.VarInt32(rvn.faces),
    rvn.vertex_matrix: io_struct.Float64(rvn.vertex_matrix),
    rvn.rxyz_euler_angles: io_struct.Float64(rvn.rxyz_euler_angles),
    rvn.translation: io_struct.Float64(rvn.translation),
    rvn.focal_length: io_struct.Float64(rvn.focal_length),
    rvn.pixel_size_xy: io_struct.Float64(rvn.pixel_size_xy),
    rvn.center_xy: io_struct.Float64(rvn.center_xy),
    rvn.alignment: io_struct.Float64(rvn.alignment),
    'scan_id': io_struct.AsciiString('scan_id'),
},
    len_label='scan_id',
    compression='lzf')


def write_raw_rv_scans_to_hdf5(output_filename, scans):
    collate_fn = partial(io_struct.collate_mapping_with_io, io=RangeVisionIO)
    scans = collate_fn(scans)

    with h5py.File(output_filename, 'w') as f:
        for key in [
                rvn.vertex_matrix,
                rvn.rxyz_euler_angles,
                rvn.translation,
                rvn.focal_length,
                rvn.pixel_size_xy,
                rvn.center_xy,
                rvn.alignment]:
            RangeVisionIO.write(f, key, scans[key].numpy())
        RangeVisionIO.write(f, 'scan_id', scans['scan_id'])
        RangeVisionIO.write(f, rvn.points, scans[rvn.points])
        RangeVisionIO.write(f, rvn.faces, scans[rvn.faces])

    print(output_filename)


ViewIO = io_struct.HDF5IO({
    'points': io_struct.VarFloat64('points'),
    'faces': io_struct.VarInt32('faces'),
    'points_alignment': io_struct.Float64('points_alignment'),
    'extrinsics': io_struct.Float64('extrinsics'),
    'intrinsics': io_struct.Float64('intrinsics'),
    'obj_alignment': io_struct.Float64('obj_alignment'),
    'obj_scale': io_struct.Float64('obj_scale'),
    'item_id': io_struct.AsciiString('item_id'),
},
    len_label='item_id',
    compression='lzf')


def write_realworld_views_to_hdf5(output_filename, scans):
    collate_fn = partial(io_struct.collate_mapping_with_io, io=ViewIO)
    scans = collate_fn(scans)

    with h5py.File(output_filename, 'w') as f:
        for key in ['extrinsics', 'intrinsics', 'points_alignment',
                    'obj_alignment', 'obj_scale']:
            ViewIO.write(f, key, scans[key].numpy())
        for key in ['item_id', 'points', 'faces']:
            ViewIO.write(f, key, scans[key])

    print(output_filename)


AnnotatedViewIO = io_struct.HDF5IO({
    'points': io_struct.Float64('points'),
    'faces': io_struct.VarInt32('faces'),
    'points_alignment': io_struct.Float64('points_alignment'),
    'extrinsics': io_struct.Float64('extrinsics'),
    'intrinsics': io_struct.Float64('intrinsics'),
    'obj_alignment': io_struct.Float64('obj_alignment'),
    'obj_scale': io_struct.Float64('obj_scale'),
    'item_id': io_struct.AsciiString('item_id'),

    'distances': io_struct.Float64('distances'),
    'directions': io_struct.Float64('directions'),
    'orig_vert_indices': io_struct.VarInt32('orig_vert_indices'),
    'orig_face_indexes': io_struct.VarInt32('orig_face_indexes'),
    'has_sharp': io_struct.Bool('has_sharp'),
    'num_sharp_curves': io_struct.Int8('num_sharp_curves'),
    'num_surfaces': io_struct.Int8('num_surfaces'),
},
    len_label='item_id',
    compression='lzf')


def write_annotated_views_to_hdf5(output_filename, scans):
    collate_fn = partial(io_struct.collate_mapping_with_io, io=AnnotatedViewIO)
    scans = collate_fn(scans)

    with h5py.File(output_filename, 'w') as f:
        for key in ['extrinsics', 'intrinsics', 'points_alignment',
                    'obj_alignment', 'obj_scale']:
            ViewIO.write(f, key, scans[key].numpy())
        for key in ['item_id', 'points', 'faces']:
            ViewIO.write(f, key, scans[key])

    print(output_filename)


UnlabeledPointCloudIO = io_struct.HDF5IO({
    'points': io_struct.Float64('points'),
    'indexes_in_whole': io_struct.Int32('indexes_in_whole'),
    'distances': io_struct.Float64('distances'),
    'item_id': io_struct.AsciiString('item_id'),
},
    len_label='has_sharp',
    compression='lzf')


def write_realworld_patches_to_hdf5(output_filename, patches):
    collate_fn = partial(io_struct.collate_mapping_with_io, io=UnlabeledPointCloudIO)
    patches = collate_fn(patches)

    with h5py.File(output_filename, 'w') as f:
        for key in ['points', 'indexes_in_whole', 'distances']:
            UnlabeledPointCloudIO.write(f, key, patches[key].numpy())
        UnlabeledPointCloudIO.write(f, 'item_id', patches['item_id'])

    print(output_filename)
