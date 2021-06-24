import h5py

import sharpf.utils.abc_utils.hdf5.io_struct as io_struct


ParametricCornersIO = io_struct.HDF5IO({
    'corners': io_struct.VarInt32('corners'),
    'not_corners': io_struct.VarInt32('not_corners'),
    'corner_centers': io_struct.VarInt32('corner_centers'),
    'init_connections': io_struct.VarInt32('init_connections'),
},
    len_label='distances',
    compression='lzf')


def save_parametric_corners(corners, filename):
    with h5py.File(filename, 'w') as f:
        ParametricCornersIO.write(f, 'corners', corners['corners'])
        ParametricCornersIO.write(f, 'not_corners', corners['not_corners'])
        ParametricCornersIO.write(f, 'corner_centers', corners['corner_centers'])
        ParametricCornersIO.write(f, 'init_connections', corners['init_connections'])
