#!/usr/bin/env python3

import argparse
from copy import deepcopy
import json
import os
import sys
import traceback

from joblib import Parallel, delayed
import numpy as np
import yaml

__dir__ = os.path.normpath(
    os.path.join(
        os.path.dirname(os.path.realpath(__file__)), '..', '..')
)

sys.path[1:1] = [__dir__]

from sharpf.data import DataGenerationException
from sharpf.utils.abc_utils.abc.abc_data import ABCModality, ABCChunk, ABC_7Z_FILEMASK
from sharpf.data.annotation import ANNOTATOR_BY_TYPE
from sharpf.data.datasets.sharpf_io import save_point_patches
from sharpf.data.mesh_nbhoods import NBHOOD_BY_TYPE
from sharpf.data.noisers import NOISE_BY_TYPE
from sharpf.data.point_samplers import SAMPLER_BY_TYPE
from sharpf.utils.abc_utils.abc.feature_utils import compute_features_nbhood, remove_boundary_features, get_curves_extents
from sharpf.utils.py_utils.console import eprint_t
from sharpf.utils.py_utils.config import load_func_from_config
from sharpf.utils.abc_utils.mesh.io import trimesh_load
import sharpf.data.data_smells as smells


LARGEST_PROCESSABLE_MESH_VERTICES = 20000


def scale_mesh(mesh, features, shape_fabrication_extent, resolution_3d,
               short_curve_quantile=0.05, n_points_per_short_curve=4):
    # compute standard size spatial extent
    mesh_extent = np.max(mesh.bounding_box.extents)
    mesh = mesh.apply_scale(shape_fabrication_extent / mesh_extent)

    # compute lengths of curves
    sharp_curves_lengths = get_curves_extents(mesh, features)

    least_len = np.quantile(sharp_curves_lengths, short_curve_quantile)
    least_len_mm = resolution_3d * n_points_per_short_curve

    mesh = mesh.apply_scale(least_len_mm / least_len)

    return mesh


# mm/pixel
HIGH_RES = 0.02
MED_RES = 0.05
LOW_RES = 0.125
XLOW_RES = 0.25


def compute_patches(patch_idx, mesh, features,
                    nbhood_extractor, sampler, noiser,
                    smell_coarse_surfaces_by_num_edges,
                    smell_coarse_surfaces_by_angles,
                    smell_deviating_resolution,
                    smell_bad_face_sampling):

    nbhood_extractor.current_patch_idx = patch_idx

    # extract neighbourhood
    try:
        nbhood, mesh_vertex_indexes, mesh_face_indexes, scaler = nbhood_extractor.get_nbhood()
        if len(nbhood.vertices) > LARGEST_PROCESSABLE_MESH_VERTICES:
            raise DataGenerationException('Too large number of vertices in crop: {}'.format(len(nbhood.vertices)))
    except DataGenerationException as e:
        eprint_t(str(e))
        return None
    centroid = nbhood_extractor.centroid

    has_smell_coarse_surfaces_by_num_edges = smell_coarse_surfaces_by_num_edges.run(mesh, mesh_face_indexes, features)
    has_smell_coarse_surfaces_by_angles = smell_coarse_surfaces_by_angles.run(mesh, mesh_face_indexes, features)

    # create annotations: condition the features onto the nbhood
    nbhood_features = compute_features_nbhood(mesh, features, mesh_face_indexes,
                                              mesh_vertex_indexes=mesh_vertex_indexes)

    # remove vertices lying on the boundary (sharp edges found in 1 face only)
    nbhood_features = remove_boundary_features(nbhood, nbhood_features, how='edges')

    # sample the neighbourhood to form a point patch
    try:
        points, normals = sampler.sample(nbhood, centroid=nbhood_extractor.centroid)
    except DataGenerationException as e:
        eprint_t(str(e))
        return None

    has_smell_deviating_resolution = smell_deviating_resolution.run(points)
    has_smell_bad_face_sampling = smell_bad_face_sampling.run(nbhood, points)

    # create a noisy sample
    noisy_points = noiser.make_noise(points, normals)
    num_sharp_curves = len([curve for curve in nbhood_features['curves'] if curve['sharp']])
    num_surfaces = len(nbhood_features['surfaces'])
    patch = {
        'points': np.array(noisy_points).astype(np.float64),
        'normals': np.array(normals).astype(np.float64),
        # 'distances': np.array(distances).astype(np.float64),
        # 'directions': np.array(directions).astype(np.float64),
        'orig_vert_indices': np.array(mesh_vertex_indexes).astype(np.int32),
        'orig_face_indexes': np.array(mesh_face_indexes).astype(np.int32),
        # 'has_sharp': has_sharp,
        'num_sharp_curves': num_sharp_curves,
        'num_surfaces': num_surfaces,
        'has_smell_coarse_surfaces_by_num_faces': has_smell_coarse_surfaces_by_num_edges,
        'has_smell_coarse_surfaces_by_angles': has_smell_coarse_surfaces_by_angles,
        'has_smell_deviating_resolution': has_smell_deviating_resolution,
        # 'has_smell_sharpness_discontinuities': has_smell_sharpness_discontinuities,
        'has_smell_bad_face_sampling': has_smell_bad_face_sampling,
        # 'has_smell_mismatching_surface_annotation': has_smell_mismatching_surface_annotation,
        'nbhood': nbhood,
        'nbhood_features': nbhood_features,
        'centroid': centroid,
        'nbhood_radius': nbhood_extractor.radius_base,
    }
    return patch


def compute_annotation_nonlocal(patch, whole_model_points, config):
    centroid = patch['centroid']
    nbhood = patch['nbhood']
    nbhood_features = patch['nbhood_features']
    nbhood_radius = patch['nbhood_radius']

    annotator = load_func_from_config(ANNOTATOR_BY_TYPE, config['annotation'])
    smell_sharpness_discontinuities = smells.SmellSharpnessDiscontinuities.from_config(
        config['smell_sharpness_discontinuities'])

    whole_model_points_norm_sq = np.linalg.norm(whole_model_points, axis=1) ** 2
    distance_to_centroid = np.sqrt(
        whole_model_points_norm_sq
        - 2 * np.dot(whole_model_points, centroid)
        + np.linalg.norm(centroid) ** 2
    )
    indexes = np.where(distance_to_centroid < nbhood_radius)[0]
    noisy_points = whole_model_points[indexes]

    try:
        distances, directions, has_sharp = annotator.annotate(nbhood, nbhood_features, noisy_points)
    except DataGenerationException as e:
        eprint_t(str(e))
        return [None] * 5
    has_smell_sharpness_discontinuities = smell_sharpness_discontinuities.run(noisy_points, distances)

    return distances, directions, has_sharp, indexes, has_smell_sharpness_discontinuities


def get_annotated_patches(data, config, n_jobs):
    shape_fabrication_extent = config.get('shape_fabrication_extent', 10.0)
    base_n_points_per_short_curve = config.get('base_n_points_per_short_curve', 8)
    base_resolution_3d = config.get('base_resolution_3d', LOW_RES)
    short_curve_quantile = config.get('short_curve_quantile', 0.05)

    nbhood_extractor = load_func_from_config(NBHOOD_BY_TYPE, config['neighbourhood'])
    sampler = load_func_from_config(SAMPLER_BY_TYPE, config['sampling'])
    noiser = load_func_from_config(NOISE_BY_TYPE, config['noise'])

    smell_coarse_surfaces_by_num_edges = smells.SmellCoarseSurfacesByNumEdges.from_config(config['smell_coarse_surfaces_by_num_edges'])
    smell_coarse_surfaces_by_angles = smells.SmellCoarseSurfacesByAngles.from_config(config['smell_coarse_surfaces_by_angles'])
    smell_deviating_resolution = smells.SmellDeviatingResolution.from_config(config['smell_deviating_resolution'])
    smell_bad_face_sampling = smells.SmellBadFaceSampling.from_config(config['smell_bad_face_sampling'])

    mesh = scale_mesh(data['mesh'], data['features'],
                      shape_fabrication_extent, base_resolution_3d,
                      short_curve_quantile=short_curve_quantile,
                      n_points_per_short_curve=base_n_points_per_short_curve)

    full_model_resolution_discount = 4.0
    nbhood_extractor.radius_base = np.sqrt(
        sampler.n_points) * 0.5 * sampler.resolution_3d / full_model_resolution_discount

    nbhood_extractor.index(mesh)

    full_model_resolution_discount = 1.0
    nbhood_extractor.radius_base = np.sqrt(
        sampler.n_points) * 0.5 * sampler.resolution_3d / full_model_resolution_discount

    features = data['features']
    has_smell_mismatching_surface_annotation = any([
        np.array(np.unique(mesh.faces[surface['face_indices']]) != np.sort(surface['vert_indices'])).all()
        for surface in features['surfaces']
    ])

    parallel = Parallel(n_jobs=n_jobs, backend='multiprocessing', verbose=100)
    delayed_iterable = (delayed(compute_patches)(
        patch_idx, mesh, features,
        nbhood_extractor, sampler, noiser,
        smell_coarse_surfaces_by_num_edges,
        smell_coarse_surfaces_by_angles,
        smell_deviating_resolution,
        smell_bad_face_sampling)
        for patch_idx in range(nbhood_extractor.n_patches_per_mesh))
    point_patches = parallel(delayed_iterable)

    for patch in point_patches:
        patch.update({
            'item_id': data['item_id'],
            'has_smell_mismatching_surface_annotation': has_smell_mismatching_surface_annotation,
        })

    whole_model_points = np.concatenate([
        patch['points']
        for patch in point_patches])

    parallel = Parallel(n_jobs=n_jobs, backend='multiprocessing', verbose=100)
    delayed_iterable = (delayed(compute_annotation_nonlocal)(
        patch, whole_model_points, config) for patch in point_patches)
    result = parallel(delayed_iterable)

    whole_model_distances = np.ones(len(whole_model_points)) * np.inf
    whole_model_directions = np.ones((len(whole_model_points), 3)) * np.inf
    for i, (distances, directions, has_sharp, indexes, has_smell_sharpness_discontinuities) in enumerate(result):
        assign_mask = whole_model_distances[indexes] > distances
        whole_model_distances[indexes[assign_mask]] = distances[assign_mask]
        whole_model_directions[indexes[assign_mask]] = directions[assign_mask]

    whole_patches = []
    for i, (patch, relabeled) in enumerate(zip(point_patches, result)):
        distances, directions, has_sharp, indexes, has_smell_sharpness_discontinuities = relabeled

        whole_patch = deepcopy(patch)
        i1, i2 = i * sampler.n_points, (i + 1) * sampler.n_points
        whole_patch['distances'] = whole_model_distances[i1:i2]
        whole_patch['directions'] = whole_model_directions[i1:i2, :]
        nbhood_features = patch['nbhood_features']
        whole_patch['has_sharp'] = any(curve['sharp'] for curve in nbhood_features['curves'])
        whole_patch['num_sharp_curves'] = len([curve for curve in nbhood_features['curves'] if curve['sharp']])
        whole_patch['num_surfaces'] = len(nbhood_features['surfaces'])
        whole_patch['has_smell_mismatching_surface_annotation'] = has_smell_mismatching_surface_annotation
        whole_patch['has_smell_sharpness_discontinuities'] = has_smell_sharpness_discontinuities
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

    item_idx = options.item_idx

    with ABCChunk([obj_filename, feat_filename]) as data_holder:
        item = data_holder[item_idx]

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
                save_point_patches(patches, output_filename)
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
    parser.add_argument('-n', dest='item_idx', type=int,
                        required=True, help='index of data to process')
    parser.add_argument('--verbose', dest='verbose', action='store_true', default=False,
                        required=False, help='be verbose')
    return parser.parse_args()


if __name__ == '__main__':
    options = parse_args()
    make_patches(options)
