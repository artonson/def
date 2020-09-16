import sharpf.utils.abc_utils.hdf5.io_struct as io


class PointCloudIO():
    def __new__(cls):
        if not hasattr(cls, 'instance'):
            cls.instance = io.HDF5IO(
                {
                    'directions': io.Float64('directions'),
                    'distances': io.Float64('distances'),
                    'has_sharp': io.Bool('has_sharp'),
                    'item_id': io.AsciiString('item_id'),
                    'normals': io.Float64('normals'),
                    'normals_estimation_10': io.Float64('normals_estimation_10'),
                    'normals_estimation_100': io.Float64('normals_estimation_100'),
                    'num_sharp_curves': io.Int8('num_sharp_curves'),
                    'num_surfaces': io.Int8('num_surfaces'),
                    'orig_face_indexes': io.VarInt32('orig_face_indexes'),
                    'orig_vert_indices': io.VarInt32('orig_vert_indices'),
                    'points': io.Float64('points'),
                    'voronoi': io.Float64('voronoi')
                },
                len_label='has_sharp',
                compression='lzf'
            )
        return cls.instance


class DepthMapIO():
    def __new__(cls):
        if not hasattr(cls, 'instance'):
            cls.instance = io.HDF5IO(
                {
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
                compression='lzf'
            )
        return cls.instance
