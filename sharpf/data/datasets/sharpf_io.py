import sharpf.utils.abc_utils.hdf5.io_struct as io

# TODO turn this variable into a singleton
PointCloudIO = io.HDF5IO(
    {
        'points': io.Float64('points'),  #
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
    compression='lzf'
)
