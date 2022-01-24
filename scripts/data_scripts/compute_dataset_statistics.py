#!/usr/bin/env python3

import argparse
from concurrent.futures import ProcessPoolExecutor, as_completed
import json
import os
import sys
import traceback
from functools import partial
from typing import Mapping, List

import igl
import numpy as np
import yaml
from torch.utils.data import DataLoader
import trimesh.transformations as tt

__dir__ = os.path.normpath(
    os.path.join(
        os.path.dirname(os.path.realpath(__file__)), '..', '..')
)
sys.path[1:1] = [__dir__]

from sharpf.utils.camera_utils.view import CameraView
from sharpf.data.imaging import IMAGING_BY_TYPE
import sharpf.data.datasets.sharpf_io as io
from sharpf.utils.geometry_utils.geometry import mean_mmd
from sharpf.utils.py_utils.config import load_func_from_config
from sharpf.utils.abc_utils.abc.abc_data import ABCChunk, ABC_7Z_FILEMASK, ABCModality
from sharpf.utils.abc_utils.abc.feature_utils import compute_features_nbhood, remove_boundary_features
from sharpf.utils.abc_utils.mesh.indexing import reindex_zerobased
from sharpf.utils.abc_utils.mesh.io import trimesh_load
from sharpf.utils.py_utils.console import eprint_t
from sharpf.utils.abc_utils.hdf5.dataset import Hdf5File, PreloadTypes
from sharpf.utils.abc_utils.hdf5.io_struct import collate_mapping_with_io
from sharpf.utils.camera_utils.camera_pose import CameraPose


PATCH_TYPES = ['Plane', 'Cylinder', 'Cone', 'Sphere', 'Torus', 'Revolution', 'Extrusion', 'BSpline', 'Other']
CURVE_TYPES = ['Line', 'Circle', 'Ellipse', 'BSpline', 'Other']


def uncollate(collated: Mapping) -> List[Mapping]:
    """
    Given a collated batch (i.e., a mapping of lists/arrays),
    produce a list of mappings.
    """
    any_value = next(iter(collated.values()))
    list_len = len(any_value)
    return [
        {key: value[idx] for key, value in collated.items()}
        for idx in range(list_len)
    ]


def get_patch(
    mesh_vertex_indexes,
    mesh_face_indexes,
    item_id,
    obj_filename,
    feat_filename,
):
    with ABCChunk([obj_filename, feat_filename]) as data_holder:
        abc_item = data_holder.get(item_id)
        mesh, _, _ = trimesh_load(abc_item.obj)
        features = yaml.load(abc_item.feat, Loader=yaml.Loader)

    nbhood = reindex_zerobased(mesh, mesh_vertex_indexes, mesh_face_indexes)
    nbhood_features = compute_features_nbhood(mesh, features, mesh_face_indexes, mesh_vertex_indexes)
    nbhood_features = remove_boundary_features(nbhood, nbhood_features, how='edges')

    return nbhood, nbhood_features


def build_patch_description(
        item,
        nbhood_features,
        points
):
    num_sharp_curves = len([curve for curve in nbhood_features['curves'] if curve['sharp']])
    num_all_curves = len(nbhood_features['curves'])
    num_surfaces = len(nbhood_features['surfaces'])
    num_samples = len(points)
    s = [
        f'num_sharp_curves {num_sharp_curves}',
        f'num_all_curves {num_all_curves}',
        f'num_surfaces {num_surfaces}',
        f'num_samples {num_samples}',
        f'mean_sampling_distance {mean_mmd(points)}',
    ]

    for curve_type in CURVE_TYPES:
        count = len([curve for curve in nbhood_features['curves'] if curve['type'] == curve_type])
        s.append(f'num_all_curve_{curve_type.lower()} {count}')

    for curve_type in CURVE_TYPES:
        count = len([curve for curve in nbhood_features['curves'] if curve['type'] == curve_type and curve['sharp']])
        s.append(f'num_sharp_curve_{curve_type.lower()} {count}')

    for surface_type in PATCH_TYPES:
        count = len([surface for surface in nbhood_features['surfaces'] if surface['type'] == surface_type])
        s.append(f'num_surface_{surface_type.lower()} {count}')

    if item is not None:
        s.append(f'has_sharp {int(item["has_sharp"])}')
        for key, value in item.items():
            if key.startswith('has_smell_'):
                s.append(f'{key} {int(value)}')

    return s


def process_images(
        item,
        imaging,
        obj_filename,
        feat_filename
):
    nbhood, nbhood_features = get_patch(
        item['orig_vert_indices'],
        item['orig_face_indexes'],
        str(item['item_id'].decode('utf-8')),
        obj_filename,
        feat_filename)
    points = imaging.image_to_points(item["image"].numpy())
    s = build_patch_description(
        item,
        nbhood_features,
        points)
    return s


def process_points(
        item,
        imaging,
        obj_filename,
        feat_filename
):
    nbhood, nbhood_features = get_patch(
        item['orig_vert_indices'],
        item['orig_face_indexes'],
        str(item['item_id'].decode('utf-8')),
        obj_filename,
        feat_filename)
    points = item['points']
    s = build_patch_description(
        item,
        nbhood_features,
        points)
    return s


def build_complete_model_description(
        item_id,
        points,
        mesh,
        features
):
    s = build_patch_description(None, features, points)

    n_verts = len(mesh.vertices)
    n_faces = len(mesh.faces)
    n_surfaces = len(features['surfaces'])
    n_curves = len(features['curves'])
    s.extend([
        f'n_verts {n_verts}',
        f'n_faces {n_faces}',
        f'n_surfaces {n_surfaces}',
        f'n_curves {n_curves}',
    ])

    squared_distance, point_face_indexes, _ = igl.point_mesh_squared_distance(
        points,
        mesh.vertices,
        mesh.faces)
    s.append(f'mean_distance_to_mesh {np.mean(np.sqrt(squared_distance))} ')
    unique, counts = np.unique(point_face_indexes, return_counts=True)
    for min_count in [1, 2, 4, 8, 16, 32, 64]:
        s.append(f'fraction_covered_{min_count} '
                 f'{len(unique[counts >= min_count]) / len(mesh.faces)}')

    return s


def process_images_whole(
        dataset,
        imaging,
        obj_filename,
        feat_filename
):
    item_id = str(dataset[0]['item_id'].decode('utf-8'))

    with ABCChunk([obj_filename, feat_filename]) as data_holder:
        abc_item = data_holder.get(item_id)
        mesh, _, _ = trimesh_load(abc_item.obj)
        features = yaml.load(abc_item.feat, Loader=yaml.Loader)

    shape_fabrication_extent = 10.
    mesh_extent = np.max(mesh.bounding_box.extents)
    mesh = mesh.apply_scale(shape_fabrication_extent / mesh_extent)
    mesh = mesh.apply_scale(dataset[0]['mesh_scale'])
    mesh = mesh.apply_translation(-mesh.vertices.mean(axis=0))

    gt_images = [view['image'] for view in dataset]
    gt_distances = [view.get('distances', np.ones_like(view['image'])) for view in dataset]
    gt_cameras = [view['camera_pose'] for view in dataset]

    points = []
    for camera_to_world_4x4, image, distances in zip(gt_cameras, gt_images, gt_distances):
        points_in_world_frame = CameraPose(camera_to_world_4x4).camera_to_world(imaging.image_to_points(image))
        points.append(points_in_world_frame)
    points = np.concatenate(points)

    s = build_complete_model_description(
        item_id,
        points,
        mesh,
        features)
    return s


def process_view(
        dataset,
        imaging,
        obj_filename,
        feat_filename
):
    item_id = str(dataset[0]['item_id'].decode('utf-8'))

    with ABCChunk([obj_filename, feat_filename]) as data_holder:
        abc_item = data_holder.get(item_id)
        obj_mesh, _, _ = trimesh_load(abc_item.obj)
        features = yaml.load(abc_item.feat, Loader=yaml.Loader)

    obj_alignment_transform = dataset[0]['obj_alignment']
    obj_scale = dataset[0]['obj_scale']
    mesh = obj_mesh.copy() \
        .apply_scale(obj_scale) \
        .apply_transform(obj_alignment_transform)

    views = [
        CameraView(
            depth=scan['points'],
            signal=None,
            faces=scan['faces'].reshape((-1, 3)),
            extrinsics=scan['extrinsics'],
            intrinsics=scan['intrinsics'],
            state='pixels')
        for scan in dataset]
    view_alignments = [scan['points_alignment'] for scan in dataset]

    points = []
    for view, view_alignment in zip(views, view_alignments):
        points_view = view.to_points()
        points.append(tt.transform_points(points_view.depth, view_alignment))
    points = np.concatenate(points)

    s = build_complete_model_description(
        item_id,
        points,
        mesh,
        features)
    return s


def main(options):
    # % \LA{for submission -- add figures displaying:
    # % 1) distribution of types of patches and curves over the patch-based dataset;
    # % 2) distribution of the number of patches and curves over the patch-based dataset;
    # % 3) empirical histogram of sampling density;
    # % 4) statistics about which percentage of surface is captured;
    # % 5) empirical histogram of number of points in scan}

    schema = io.IO_SPECS[options.io_spec]
    process_fn = {
        'images': process_images,
        'points': process_points,
        # 'whole_points': process_points_whole,
        'whole_images': process_images_whole,
        'annotated_view_io': process_view,
    }[options.io_spec]

    dataset = Hdf5File(
        options.input_file,
        io=schema,
        labels='*',
        preload=PreloadTypes.LAZY)

    batch_size = 128
    loader = DataLoader(
        dataset,
        num_workers=options.n_jobs,
        batch_size=batch_size,
        shuffle=False,
        collate_fn=partial(collate_mapping_with_io, io=schema))

    with open(options.dataset_config) as config_file:
        config = json.load(config_file)
    imaging = None
    if options.io_spec in ['images', 'whole_images']:
        imaging = load_func_from_config(IMAGING_BY_TYPE, config['imaging'])

    obj_filename = os.path.join(
        options.abc_input_dir,
        ABC_7Z_FILEMASK.format(
            chunk=options.chunk.zfill(4),
            modality=ABCModality.OBJ.value,
            version='00'))
    feat_filename = os.path.join(
        options.abc_input_dir,
        ABC_7Z_FILEMASK.format(
            chunk=options.chunk.zfill(4),
            modality=ABCModality.FEAT.value,
            version='00'))
    if options.verbose:
        eprint_t('Obj filename: {}, feat filename: {}'.format(obj_filename, feat_filename))

    if options.io_spec in ['whole_points', 'whole_images', 'annotated_view_io']:
        item_id = str(dataset[0]['item_id'].decode('utf-8'))
        s = process_fn(
            dataset,
            imaging,
            obj_filename,
            feat_filename)
        with open(options.output_file, 'a') as out_file:
            lines = ['{} '.format(item_id) + line for line in s]
            out_file.write('\n'.join(lines) + '\n')
        if options.verbose:
            eprint_t('Processed item {}'.format(item_id))

    else:
        max_queued_tasks = 1024
        with ProcessPoolExecutor(max_workers=options.n_jobs) as executor:
            index_by_future = {}
            item_idx = 0
            for batch_idx, batch in enumerate(loader):
                # submit the full batch for processing
                items = uncollate(batch)
                for item in items:
                    item_idx += 1
                    item_id = str(item['item_id'].decode('utf-8'))
                    future = executor.submit(process_fn, item, imaging, obj_filename, feat_filename)
                    index_by_future[future] = (item_idx, item_id)

                if len(index_by_future) >= max_queued_tasks:
                    # don't enqueue more tasks until all previously submitted
                    # are complete and dumped to disk

                    for future in as_completed(index_by_future):
                        item_idx, item_id = index_by_future[future]
                        try:
                            s = future.result()
                        except Exception as e:
                            if options.verbose:
                                eprint_t('Error getting item {}: {}'.format(item_id, str(e)))
                                eprint_t(traceback.format_exc())
                        else:
                            with open(options.output_file, 'a') as out_file:
                                lines = ['{} {} '.format(item_id, item_idx) + line for line in s]
                                out_file.write('\n'.join(lines) + '\n')
                            if options.verbose:
                                eprint_t('Processed item {}, {}'.format(item_id, item_idx))

                    index_by_future = {}

                    if options.verbose:
                        seen_fraction = batch_idx * batch_size / len(loader.dataset)
                        eprint_t(f'Processed {batch_idx * batch_size:d} items '
                                 f'({seen_fraction * 100:3.1f}% of data)')


def parse_args():
    parser = argparse.ArgumentParser()

    parser.add_argument('-i', '--input-file', dest='input_file',
                        required=True, help='input HDF5 file.')
    parser.add_argument('-a', '--abc-input-dir', dest='abc_input_dir',
                        required=True, help='input dir with ABC dataset.')
    parser.add_argument('-c', '--chunk', required=True, help='ABC chunk id to process.')
    parser.add_argument('-o', '--output-file', dest='output_file',
                        required=True, help='output file.')
    parser.add_argument('-g', '--dataset-config', dest='dataset_config',
                        required=True, help='dataset configuration file.')
    parser.add_argument('-j', '--jobs', dest='n_jobs',
                        type=int, default=4, help='CPU jobs to use in parallel [default: 4].')
    parser.add_argument('-io', '--io-schema', dest='io_spec',
                        choices=io.IO_SPECS.keys(), help='i/o spec to use.')

    parser.add_argument('--verbose', dest='verbose', action='store_true', default=False,
                        required=False, help='be verbose')
    return parser.parse_args()


if __name__ == '__main__':
    options = parse_args()
    main(options)
