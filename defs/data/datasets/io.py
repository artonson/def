import defs.utils.abc_utils.hdf5.io_struct as io


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
                    'voronoi': io.Float64('voronoi'),
                    'has_smell_coarse_surfaces_by_num_faces': io.Bool('has_smell_coarse_surfaces_by_num_faces'),
                    'has_smell_deviating_resolution': io.Bool('has_smell_deviating_resolution'),
                    'has_smell_sharpness_discontinuities': io.Bool('has_smell_sharpness_discontinuities'),
                    'has_smell_bad_face_sampling': io.Bool('has_smell_bad_face_sampling'),
                    'has_smell_mismatching_surface_annotation': io.Bool('has_smell_mismatching_surface_annotation'),
                    'has_smell_coarse_surfaces_by_angles': io.Bool('has_smell_coarse_surfaces_by_angles'),
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
                    'mesh_scale': io.Float64('mesh_scale'),
                    'voronoi': io.Float64('voronoi'),
                    'has_smell_bad_face_sampling': io.Bool('has_smell_bad_face_sampling'),
                    'has_smell_coarse_surfaces_by_angles': io.Bool('has_smell_coarse_surfaces_by_angles'),
                    'has_smell_coarse_surfaces_by_num_faces': io.Bool('has_smell_coarse_surfaces_by_num_faces'),
                    'has_smell_depth_discontinuity': io.Bool('has_smell_depth_discontinuity'),
                    'has_smell_deviating_resolution': io.Bool('has_smell_deviating_resolution'),
                    'has_smell_mesh_self_intersections': io.Bool('has_smell_mesh_self_intersections'),
                    'has_smell_mismatching_surface_annotation': io.Bool('has_smell_mismatching_surface_annotation'),
                    'has_smell_raycasting_background': io.Bool('has_smell_raycasting_background'),
                    'has_smell_sharpness_discontinuities': io.Bool('has_smell_sharpness_discontinuities'),
                },
                len_label='has_sharp',
                compression='lzf'
            )
        return cls.instance


class WholePointCloudIO():
    def __new__(cls):
        if not hasattr(cls, 'instance'):
            cls.instance = io.HDF5IO(
                {
                    'points': io.VarFloat64('points'),
                    'normals': io.VarFloat64('normals'),
                    'voronoi': io.VarFloat64('voronoi'),
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
        return cls.instance


class ComparisonIO():
    def __new__(cls):
        if not hasattr(cls, 'instance'):
            cls.instance = io.HDF5IO(
                {
                    'points': io.Float64('points'),
                    'image': io.Float64('image'),
                    'distances': io.Float64('distances'),
                    'distances_for_sh': io.Float64('distances_for_sh'),
                    'sharpness': io.Float32('sharpness'),
                    'sharpness_seg': io.Float32('sharpness_seg'),
                    'voronoi': io.Float32('voronoi'),
                    'ecnet': io.Float32('ecnet'),
                },
                len_label='ecnet',
                compression='lzf'
            )
        return cls.instance


class UnlabeledDepthMapIO():
    def __new__(cls):
        if not hasattr(cls, 'instance'):
            cls.instance = io.HDF5IO(
                {
                    'image': io.Float64('image'),
                    'normals': io.Float64('normals'),
                    'item_id': io.AsciiString('item_id'),
                    'orig_vert_indices': io.VarInt32('orig_vert_indices'),
                    'orig_face_indexes': io.VarInt32('orig_face_indexes'),
                    'camera_pose': io.Float64('camera_pose'),
                    'mesh_scale': io.Float64('mesh_scale'),
                },
                len_label='image',
                compression='lzf')
        return cls.instance


class UnlabeledPointsIO():
    def __new__(cls):
        if not hasattr(cls, 'instance'):
            cls.instance = io.HDF5IO(
                {
                    'points': io.Float64('points'),
                    'voronoi': io.Float64('voronoi'),
                },
                len_label='points',
                compression='lzf')
        return cls.instance


class WholePointV2CloudIO():
    def __new__(cls):
        if not hasattr(cls, 'instance'):
            cls.instance = io.HDF5IO(
                {
                    'points': io.VarFloat64('points'),
                    'distances': io.VarFloat64('distances'),
                },
                len_label='points',
                compression='lzf')
        return cls.instance


class FusedIO():
    def __new__(cls):
        if not hasattr(cls, 'instance'):
            cls.instance = io.HDF5IO(
                {
                    'distances': io.Float64('distances'),
                },
                len_label='distances',
                compression='lzf')
        return cls.instance


class UnlabeledSLDepthMapIO():
    def __new__(cls):
        if not hasattr(cls, 'instance'):
            cls.instance = io.HDF5IO(
                {
                    'image': io.Float64('image'),
                    'image_offset': io.Float64('image_offset'),
                    'item_id': io.AsciiString('item_id'),
                    'camera_extrinsics': io.Float64('camera_extrinsics'),
                    'camera_intrinsics_f': io.Float64('camera_intrinsics_f'),
                    'camera_intrinsics_s': io.Float64('camera_intrinsics_s'),
                },
                len_label='image',
                compression='lzf')
        return cls.instance


class AnnotatedViewIO():
    def __new__(cls):
        if not hasattr(cls, 'instance'):
            cls.instance = io.HDF5IO(
                {
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
                },
                len_label='item_id',
                compression='lzf')
        return cls.instance
