#!/usr/bin/env python3

import argparse
from copy import deepcopy
import json
import os
import sys
import traceback

import igl
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
import sharpf.data.datasets.sharpf_io as io
from sharpf.data.camera_pose_manager import POSE_MANAGER_BY_TYPE
from sharpf.data.noisers import NOISE_BY_TYPE
from sharpf.data.imaging import IMAGING_BY_TYPE
import sharpf.utils.abc_utils.abc.feature_utils as feature_utils
from sharpf.utils.py_utils.console import eprint_t
from sharpf.utils.py_utils.config import load_func_from_config
from sharpf.utils.abc_utils.mesh.io import trimesh_load
import sharpf.data.data_smells as smells
from sharpf.utils.camera_utils.camera_pose import CameraPose


LARGEST_PROCESSABLE_MESH_VERTICES = 20000


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


def compute_patches(
        patch,
        whole_model_points,
        annotator,
        smell_sharpness_discontinuities):
    nbhood = patch['nbhood']
    nbhood_features = patch['nbhood_features']

    distance_sq, face_indexes, _ = igl.point_mesh_squared_distance(
        whole_model_points,
        nbhood.vertices,
        nbhood.faces)
    indexes = np.where(np.sqrt(distance_sq) < HIGH_RES / 100)[0]
    noisy_points, normals = whole_model_points[indexes], nbhood.face_normals[face_indexes[indexes]]

    try:
        distances, directions, has_sharp = annotator.annotate(nbhood, nbhood_features, noisy_points)
    except DataGenerationException as e:
        eprint_t(str(e))
        return None

    has_smell_sharpness_discontinuities = smell_sharpness_discontinuities.run(noisy_points, distances)

    patch = {
        'distances': np.array(distances).astype(np.float64),
        'directions': np.array(directions).astype(np.float64),
        'has_sharp': has_sharp,
        'has_smell_sharpness_discontinuities': has_smell_sharpness_discontinuities,
        'indexes': indexes
    }
    return patch


def get_annotated_patches(data, config, n_jobs):
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

    mesh, features = data['mesh'], data['features']

    processed_mesh = trimesh.base.Trimesh(vertices=mesh.vertices, faces=mesh.faces, process=True, validate=True)
    if processed_mesh.vertices.shape != mesh.vertices.shape or \
            processed_mesh.faces.shape != mesh.faces.shape or not mesh.is_watertight:
        raise DataGenerationException('Will not process mesh {}: likely the mesh is broken'.format(data['item_id']))

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

    non_annotated_patches = []
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
        nbhood_features = feature_utils.compute_features_nbhood(
            mesh, features, mesh_face_indexes, mesh_vertex_indexes=mesh_vertex_indexes)

        # remove vertices lying on the boundary (sharp edges found in 1 face only)
        nbhood_features = feature_utils.remove_boundary_features(nbhood, nbhood_features, how='edges')

        # create a noisy sample
        noisy_points = noiser.make_noise(
            camera_pose.world_to_camera(points),
            normals,
            z_direction=np.array([0., 0., -1.]))

        # convert everything to images
        ray_indexes = np.where(image.ravel() != 0)[0]
        noisy_image = imaging.points_to_image(noisy_points, ray_indexes)
        normals_image = imaging.points_to_image(normals, ray_indexes, assign_channels=[0, 1, 2])

        # compute statistics
        num_sharp_curves = len([curve for curve in nbhood_features['curves'] if curve['sharp']])
        num_surfaces = len(nbhood_features['surfaces'])

        patch_info = {
            'image': noisy_image,
            'normals': normals_image,
            # 'distances': distances_image,
            # 'directions': directions_image,
            # 'item_id': item.item_id,
            'ray_indexes': ray_indexes,
            'orig_vert_indices': mesh_vertex_indexes,
            'orig_face_indexes': mesh_face_indexes,
            # 'has_sharp': has_sharp,
            'num_sharp_curves': num_sharp_curves,
            'num_surfaces': num_surfaces,
            'camera_pose': camera_pose.camera_to_world_4x4,
            'mesh_scale': mesh_scale,
            'has_smell_coarse_surfaces_by_num_faces': has_smell_coarse_surfaces_by_num_edges,
            'has_smell_coarse_surfaces_by_angles': has_smell_coarse_surfaces_by_angles,
            'has_smell_deviating_resolution': has_smell_deviating_resolution,
            # 'has_smell_sharpness_discontinuities': has_smell_sharpness_discontinuities,
            'has_smell_bad_face_sampling': has_smell_bad_face_sampling,
            'has_smell_mismatching_surface_annotation': has_smell_mismatching_surface_annotation,
            'has_smell_raycasting_background': has_smell_raycasting_background,
            'has_smell_depth_discontinuity': has_smell_depth_discontinuity,
            'has_smell_mesh_self_intersections': has_smell_mesh_self_intersections,
            'nbhood': nbhood,
            'nbhood_features': nbhood_features,
        }
        non_annotated_patches.append(patch_info)


    whole_model_points = []
    whole_model_point_indexes = []
    for patch in non_annotated_patches:
        image = patch['image']
        camera_to_world_4x4 = patch['camera_pose']
        points_in_camera_frame = imaging.image_to_points(image)
        camera_pose = CameraPose(camera_to_world_4x4)
        points_in_world_frame = camera_pose.camera_to_world(points_in_camera_frame)
        whole_model_points.append(points_in_world_frame)
        whole_model_point_indexes.append(np.arange(len(points_in_world_frame)))
    whole_model_points = np.concatenate(whole_model_points)
    whole_model_point_indexes = np.concatenate(whole_model_point_indexes)

    parallel = Parallel(n_jobs=n_jobs, backend='multiprocessing', verbose=100)
    delayed_iterable = (delayed(compute_patches)(
        patch,
        whole_model_points,
        annotator,
        smell_sharpness_discontinuities)
        for patch in non_annotated_patches)
    annotated_patches = parallel(delayed_iterable)

    whole_model_distances = np.ones(len(whole_model_points)) * annotator.distance_upper_bound
    whole_model_directions = np.zeros((len(whole_model_points), 3))
    for patch in annotated_patches:
        distances = patch['distances']
        directions = patch['directions']
        indexes = patch['indexes']
        assign_mask = whole_model_distances[indexes] > distances
        whole_model_distances[indexes[assign_mask]] = distances[assign_mask]
        whole_model_directions[indexes[assign_mask]] = directions[assign_mask]

    whole_patches = []
    for non_annotated, annotated in zip(non_annotated_patches, annotated_patches):
        whole_patch = deepcopy(non_annotated)
        whole_patch['image'] = whole_patch['image']
        whole_patch['normals'] = whole_patch['normals']
        whole_patch['distances'] = imaging.points_to_image(
            whole_model_distances[annotated['indexes']],
            non_annotated['ray_indexes'],
            assign_channels=[0])
        whole_patch['directions'] = imaging.points_to_image(
            whole_model_directions[annotated['indexes'], :],
            non_annotated['ray_indexes'],
            assign_channels=[0, 1, 2])
        whole_patch['has_smell_mismatching_surface_annotation'] = has_smell_mismatching_surface_annotation
        whole_patch['has_smell_sharpness_discontinuities'] = annotated['has_smell_sharpness_discontinuities']
        whole_patch['has_sharp'] = annotated['has_sharp']
        whole_patch['item_id'] = data['item_id']
        whole_patch.pop('nbhood')
        whole_patch.pop('nbhood_features')
        whole_patch.pop('ray_indexes')
        whole_patch['indexes_in_whole'] = annotated['indexes']
        whole_patches.append(whole_patch)

    return whole_patches


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

    with open(options.dataset_config) as config_file:
        config = json.load(config_file)

    with ABCChunk([obj_filename, feat_filename]) as data_holder:
        if None is not options.item_idx:
            item = data_holder[options.item_idx]
        else:
            assert None is not options.item_id
            item = data_holder.get(options.item_id)

        mesh, _, _ = trimesh_load(item.obj)
        features = yaml.load(item.feat, Loader=yaml.Loader)
        data = {'mesh': mesh, 'features': features, 'item_id': item.item_id}

        try:
            eprint_t("Processing chunk file {chunk}, item {item}".format(
                chunk=obj_filename, item=data['item_id']))
            patches = get_annotated_patches(data, config, options.n_jobs)

        except Exception as e:
            eprint_t('Error processing item {item_id} from chunk {chunk}: {what}'.format(
                item_id=data['item_id'], chunk='[{},{}]'.format(obj_filename, feat_filename), what=e))
            eprint_t(traceback.format_exc())

        else:
            eprint_t('Done processing item {item_id} from chunk {chunk}'.format(
                item_id=data['item_id'], chunk='[{},{}]'.format(obj_filename, feat_filename)))

            if len(patches) == 0:
                return

            output_filename = os.path.join(
                options.output_dir,
                'abc_{chunk}_{item_id}.hdf5'.format(
                    chunk=options.chunk.zfill(4),
                    item_id=data['item_id']))
            try:
                save_fn = io.SAVE_FNS['whole_images']
                save_fn(patches, output_filename)
            except Exception as e:
                eprint_t('Error writing patches to disk at {output_file}: {what}'.format(
                    output_file=output_filename, what=e))
                eprint_t(traceback.format_exc())
            else:
                eprint_t('Done writing {num_patches} patches to disk at {output_file}'.format(
                    num_patches=len(patches), output_file=output_filename))


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

    group = parser.add_mutually_exclusive_group(required=True)
    group.add_argument('-id', '--item-id', dest='item_id', help='data id to process.')
    group.add_argument('-n', '--item-index', dest='item_idx', type=int, help='index of data to process')

    parser.add_argument('--verbose', dest='verbose', action='store_true', default=False,
                        required=False, help='be verbose')
    return parser.parse_args()


if __name__ == '__main__':
    options = parse_args()
    make_patches(options)
