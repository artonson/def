#!/usr/bin/env python3

import argparse
import json
import os
import sys

from joblib import Parallel, delayed
import numpy as np
import yaml

__dir__ = os.path.normpath(
    os.path.join(
        os.path.dirname(os.path.realpath(__file__)), '..', '..')
)

sys.path[1:1] = [__dir__]

from sharpf.utils.abc_utils.abc.abc_data import ABCModality, ABCChunk, ABC_7Z_FILEMASK
from sharpf.data.point_samplers import SAMPLER_BY_TYPE
from sharpf.utils.abc_utils.abc.feature_utils import get_curves_extents
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
LARGEST_ESTIMATED_NUM_POINTS = 1000000  # 1M


def check_models_slice(meshes_filename, feats_filename, data_slice, config, output_file):
    slice_start, slice_end = data_slice
    with ABCChunk([meshes_filename, feats_filename]) as data_holder:
        for item in data_holder[slice_start:slice_end]:
            mesh, _, _ = trimesh_load(item.obj)
            features = yaml.load(item.feat, Loader=yaml.Loader)

            smell_mesh_self_intersections = smells.SmellMeshSelfIntersections()
            has_smell_mesh_self_intersections = smell_mesh_self_intersections.run(mesh)

            shape_fabrication_extent = config.get('shape_fabrication_extent', 10.0)
            base_n_points_per_short_curve = config.get('base_n_points_per_short_curve', 8)
            base_resolution_3d = config.get('base_resolution_3d', LOW_RES)
            short_curve_quantile = config.get('short_curve_quantile', 0.05)

            mesh = scale_mesh(mesh, features, shape_fabrication_extent, base_resolution_3d,
                              short_curve_quantile=short_curve_quantile,
                              n_points_per_short_curve=base_n_points_per_short_curve)

            sampler = load_func_from_config(SAMPLER_BY_TYPE, config['sampling'])
            n_points = np.ceil(mesh.area / (np.pi * sampler.resolution_3d ** 2 / 4)).astype(int)

            with open(output_file, 'a') as f:
                f.write('{item_id} {n_points} {has_smell_mesh_self_intersections}\n'.format(
                    item_id=item.item_id,
                    n_points=n_points,
                    has_smell_mesh_self_intersections=has_smell_mesh_self_intersections))


def main(options):
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
            'abc_{chunk}_{slice_start}_{slice_end}.txt'.format(
                chunk=options.chunk.zfill(4), slice_start=slice_start, slice_end=slice_end)
        )
        for slice_start, slice_end in abc_data_slices]

    with open(options.dataset_config) as config_file:
        config = json.load(config_file)

    MAX_SEC_PER_PATCH = 100 * 6
    max_patches_per_mesh = config['neighbourhood'].get('max_patches_per_mesh', 32)
    parallel = Parallel(n_jobs=options.n_jobs, backend='multiprocessing',
                        timeout=chunk_size * max_patches_per_mesh * MAX_SEC_PER_PATCH)
    delayed_iterable = (delayed(check_models_slice)(obj_filename, feat_filename, data_slice, config, out_filename)
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
    main(options)
