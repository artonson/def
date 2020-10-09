#!/usr/bin/env python3

import argparse
from collections import defaultdict
import json
import os
import sys
import traceback

from joblib import Parallel, delayed
import numpy as np
import yaml
import trimesh

__dir__ = os.path.normpath(
    os.path.join(
        os.path.dirname(os.path.realpath(__file__)), '..', '..')
)

sys.path[1:1] = [__dir__]

from sharpf.data import DataGenerationException
from sharpf.utils.abc_utils.abc.abc_data import ABCModality, ABCChunk, ABC_7Z_FILEMASK
from sharpf.data.annotation import ANNOTATOR_BY_TYPE
from sharpf.data.camera_pose_manager import POSE_MANAGER_BY_TYPE
from sharpf.data.datasets.sharpf_io import save_depth_maps
from sharpf.data.imaging import IMAGING_BY_TYPE
from sharpf.data.noisers import NOISE_BY_TYPE
from sharpf.utils.abc_utils.abc import feature_utils
from sharpf.utils.py_utils.console import eprint_t
from sharpf.utils.py_utils.os import add_suffix
from sharpf.utils.py_utils.config import load_func_from_config
from sharpf.utils.abc_utils.mesh.io import trimesh_load
import sharpf.data.data_smells as smells


def scale_mesh(mesh, features, shape_fabrication_extent, resolution_3d,
               short_curve_quantile=0.05, n_points_per_short_curve=4):
    # compute standard size spatial extent
    mesh_extent = np.max(mesh.bounding_box.extents)
    mesh = mesh.apply_scale(shape_fabrication_extent / mesh_extent)

    # compute lengths of curves
    sharp_curves_lengths = feature_utils.get_curves_extents(mesh, features)

    least_len = np.quantile(sharp_curves_lengths, short_curve_quantile)
    least_len_mm = resolution_3d * n_points_per_short_curve

    scale = least_len_mm / least_len
    mesh = mesh.apply_scale(scale)

    return mesh, scale


# mm/pixel
HIGH_RES = 0.02
MED_RES = 0.05
LOW_RES = 0.125
XLOW_RES = 0.25


def get_annotated_patches(item, config):
    shape_fabrication_extent = config.get('shape_fabrication_extent', 10.0)
    base_n_points_per_short_curve = config.get('base_n_points_per_short_curve', 8)
    base_resolution_3d = config.get('base_resolution_3d', LOW_RES)

    short_curve_quantile = config.get('short_curve_quantile', 0.05)

    pose_manager = load_func_from_config(POSE_MANAGER_BY_TYPE, config['camera_pose'])
    imaging = load_func_from_config(IMAGING_BY_TYPE, config['imaging'])
    noiser = load_func_from_config(NOISE_BY_TYPE, config['noise'])
    annotator = load_func_from_config(ANNOTATOR_BY_TYPE, config['annotation'])

    smell_coarse_surfaces_by_num_edges = smells.SmellCoarseSurfacesByNumEdges.from_config(config['smell_coarse_surfaces_by_num_edges'])
    smell_coarse_surfaces_by_angles = smells.SmellCoarseSurfacesByAngles.from_config(config['smell_coarse_surfaces_by_angles'])
    smell_deviating_resolution = smells.SmellDeviatingResolution.from_config(config['smell_deviating_resolution'])
    smell_sharpness_discontinuities = smells.SmellSharpnessDiscontinuities.from_config(config['smell_sharpness_discontinuities'])
    smell_bad_face_sampling = smells.SmellBadFaceSampling.from_config(config['smell_bad_face_sampling'])
    smell_raycasting_background = smells.SmellRaycastingBackground.from_config(config['smell_raycasting_background'])
    smell_depth_discontinuity = smells.SmellDepthDiscontinuity.from_config(config['smell_depth_discontinuity'])
    smell_mesh_self_intersections = smells.SmellMeshSelfIntersections.from_config(config['smell_mesh_self_intersections'])

    # load the mesh and the feature curves annotations
    mesh, _, _ = trimesh_load(item.obj)
    features = yaml.load(item.feat, Loader=yaml.Loader)

    processed_mesh = trimesh.base.Trimesh(vertices=mesh.vertices, faces=mesh.faces, process=True, validate=True)
    if processed_mesh.vertices.shape != mesh.vertices.shape or \
            processed_mesh.faces.shape != mesh.faces.shape or not mesh.is_watertight:
        raise DataGenerationException('Will not process mesh {}: likely the mesh is broken'.format(item.item_id))

    has_smell_mismatching_surface_annotation = any([
        np.array(np.unique(mesh.faces[surface['face_indices']]) != np.sort(surface['vert_indices'])).all()
        for surface in features['surfaces']
    ])
    has_smell_mesh_self_intersections = smell_mesh_self_intersections.run(mesh)

    # fix mesh fabrication size in physical mm
    mesh, mesh_scale = scale_mesh(mesh, features, shape_fabrication_extent, base_resolution_3d,
                                  short_curve_quantile=short_curve_quantile,
                                  n_points_per_short_curve=base_n_points_per_short_curve)

    mesh = mesh.apply_translation(-mesh.vertices.mean(axis=0))

    # generate camera poses
    pose_manager.prepare(mesh)

    for pose_idx, camera_pose in enumerate(pose_manager):
        eprint_t("Computing images from pose {pose_idx}".format(pose_idx=pose_idx))

        # extract neighbourhood
        try:
            image, points, normals, mesh_face_indexes = \
                imaging.get_image_from_pose(mesh, camera_pose, return_hit_face_indexes=True)
        except DataGenerationException as e:
            eprint_t(str(e))
            continue

        nbhood, mesh_vertex_indexes, mesh_face_indexes = \
            feature_utils.submesh_from_hit_surfaces(mesh, features, mesh_face_indexes)

        has_smell_coarse_surfaces_by_num_edges = smell_coarse_surfaces_by_num_edges.run(mesh, mesh_face_indexes, features)
        has_smell_coarse_surfaces_by_angles = smell_coarse_surfaces_by_angles.run(mesh, mesh_face_indexes, features)
        has_smell_deviating_resolution = smell_deviating_resolution.run(points)
        has_smell_bad_face_sampling = smell_bad_face_sampling.run(nbhood, points)
        has_smell_raycasting_background = smell_raycasting_background.run(image)
        has_smell_depth_discontinuity = smell_depth_discontinuity.run(image)

        # create annotations: condition the features onto the nbhood
        nbhood_features = feature_utils.compute_features_nbhood(mesh, features, mesh_face_indexes, mesh_vertex_indexes=mesh_vertex_indexes)

        # remove vertices lying on the boundary (sharp edges found in 1 face only)
        nbhood_features = feature_utils.remove_boundary_features(nbhood, nbhood_features, how='edges')

        # create a noisy sample
        for configuration, noisy_points in noiser.make_noise(
                camera_pose.world_to_camera(points),
                normals,
                z_direction=np.array([0., 0., -1.])):

            # compute the TSharpDF
            try:
                distances, directions, has_sharp = annotator.annotate(
                    nbhood, nbhood_features, camera_pose.camera_to_world(noisy_points))
            except DataGenerationException as e:
                eprint_t(str(e))
                continue

            has_smell_sharpness_discontinuities = smell_sharpness_discontinuities.run(noisy_points, distances)

            # convert everything to images
            ray_indexes = np.where(image.ravel() != 0)[0]
            noisy_image = imaging.points_to_image(noisy_points, ray_indexes)
            normals = imaging.points_to_image(normals, ray_indexes, assign_channels=[0, 1, 2])
            distances = imaging.points_to_image(distances.reshape(-1, 1), ray_indexes, assign_channels=[0])
            directions = imaging.points_to_image(directions, ray_indexes, assign_channels=[0, 1, 2])

            # compute statistics
            num_sharp_curves = len([curve for curve in nbhood_features['curves'] if curve['sharp']])
            num_surfaces = len(nbhood_features['surfaces'])

            patch_info = {
                'image': noisy_image,
                'normals': normals,
                'distances': distances,
                'directions': directions,
                'item_id': item.item_id,
                'orig_vert_indices': mesh_vertex_indexes,
                'orig_face_indexes': mesh_face_indexes,
                'has_sharp': has_sharp,
                'num_sharp_curves': num_sharp_curves,
                'num_surfaces': num_surfaces,
                'camera_pose': camera_pose.camera_to_world_4x4,
                'mesh_scale': mesh_scale,
                'has_smell_coarse_surfaces_by_num_faces': has_smell_coarse_surfaces_by_num_edges,
                'has_smell_coarse_surfaces_by_angles': has_smell_coarse_surfaces_by_angles,
                'has_smell_deviating_resolution': has_smell_deviating_resolution,
                'has_smell_sharpness_discontinuities': has_smell_sharpness_discontinuities,
                'has_smell_bad_face_sampling': has_smell_bad_face_sampling,
                'has_smell_mismatching_surface_annotation': has_smell_mismatching_surface_annotation,
                'has_smell_raycasting_background': has_smell_raycasting_background,
                'has_smell_depth_discontinuity': has_smell_depth_discontinuity,
                'has_smell_mesh_self_intersections': has_smell_mesh_self_intersections,
            }
            yield configuration, patch_info


def generate_patches(meshes_filename, feats_filename, data_slice, config, output_file):
    slice_start, slice_end = data_slice
    with ABCChunk([meshes_filename, feats_filename]) as data_holder:
        point_patches_by_config = defaultdict(list)
        for item in data_holder[slice_start:slice_end]:
            eprint_t("Processing chunk file {chunk}, item {item}".format(
                chunk=meshes_filename, item=item.item_id))
            try:
                for configuration, patch_info in get_annotated_patches(item, config):
                    config_name = configuration.get('name')
                    point_patches_by_config[config_name].append(patch_info)

            except Exception as e:
                eprint_t('Error processing item {item_id} from chunk {chunk}: {what}'.format(
                    item_id=item.item_id, chunk='[{},{}]'.format(meshes_filename, feats_filename), what=e))
                eprint_t(traceback.format_exc())

            else:
                eprint_t('Done processing item {item_id} from chunk {chunk}'.format(
                    item_id=item.item_id, chunk='[{},{}]'.format(meshes_filename, feats_filename)))

    for config_name, point_patches in point_patches_by_config.items():
        if len(point_patches) == 0:
            continue

        output_file_config = add_suffix(output_file, config_name) if config_name else output_file
        try:
            save_depth_maps(point_patches, output_file_config)
        except Exception as e:
            eprint_t('Error writing patches to disk at {output_file}: {what}'.format(
                output_file=output_file_config, what=e))
            eprint_t(traceback.format_exc())
        else:
            eprint_t('Done writing {num_patches} patches to disk at {output_file}'.format(
                num_patches=len(point_patches), output_file=output_file_config))


def make_patches(options):
    obj_filename = os.path.join(
        options.input_dir,
        ABC_7Z_FILEMASK.format(
            chunk=options.chunk.zfill(4),
            modality=ABCModality.OBJ.value,
            version='00'
        )
    )
    feat_filename = os.path.join(
        options.input_dir,
        ABC_7Z_FILEMASK.format(
            chunk=options.chunk.zfill(4),
            modality=ABCModality.FEAT.value,
            version='00'
        )
    )

    if all([opt is not None for opt in (options.slice_start, options.slice_end)]):
        slice_start, slice_end = options.slice_start, options.slice_end
    else:
        with ABCChunk([obj_filename, feat_filename]) as abc_data:
            slice_start, slice_end = 0, len(abc_data)
        if options.slice_start is not None:
            slice_start = options.slice_start
        if options.slice_end is not None:
            slice_end = options.slice_end

    processes_to_spawn = 10 * options.n_jobs
    chunk_size = max(1, (slice_end - slice_start) // processes_to_spawn)
    abc_data_slices = [(start, start + chunk_size)
                       for start in range(slice_start, slice_end, chunk_size)]

    output_files = [
        os.path.join(
            options.output_dir,
            'abc_{chunk}_{slice_start}_{slice_end}.hdf5'.format(
                chunk=options.chunk.zfill(4), slice_start=slice_start, slice_end=slice_end)
        )
        for slice_start, slice_end in abc_data_slices]

    with open(options.dataset_config) as config_file:
        config = json.load(config_file)

    MAX_SEC_PER_PATCH = 100
    max_patches_per_mesh = config.get('max_patches_per_mesh', 1500)
    parallel = Parallel(n_jobs=options.n_jobs, backend='multiprocessing',
                        timeout=chunk_size * max_patches_per_mesh * MAX_SEC_PER_PATCH)
    delayed_iterable = (delayed(generate_patches)(obj_filename, feat_filename, data_slice, config, out_filename)
                        for data_slice, out_filename in zip(abc_data_slices, output_files))
    parallel(delayed_iterable)


def parse_args():
    parser = argparse.ArgumentParser()

    parser.add_argument('-j', '--jobs', dest='n_jobs',
                        type=int, default=4, help='CPU jobs to use in parallel [default: 4].')
    parser.add_argument('-i', '--input-dir', dest='input_dir',
                        required=True, help='input dir with ABC dataset.')
    parser.add_argument('-c', '--chunk', required=True, help='ABC chunk id to process.')
    parser.add_argument('-o', '--output-dir', dest='output_dir',
                        required=True, help='output dir.')
    parser.add_argument('-g', '--dataset-config', dest='dataset_config',
                        required=True, help='dataset configuration file.')
    parser.add_argument('-n1', dest='slice_start', type=int,
                        required=False, help='min index of data to process')
    parser.add_argument('-n2', dest='slice_end', type=int,
                        required=False, help='max index of data to process')
    parser.add_argument('--verbose', dest='verbose', action='store_true', default=False,
                        required=False, help='be verbose')
    return parser.parse_args()


if __name__ == '__main__':
    options = parse_args()
    make_patches(options)
