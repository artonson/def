#!/usr/bin/env python3

import argparse
from concurrent.futures import ProcessPoolExecutor, as_completed
import os
import sys
import traceback

import yaml

__dir__ = os.path.normpath(
    os.path.join(
        os.path.dirname(os.path.realpath(__file__)), '..', '..')
)
sys.path[1:1] = [__dir__]

from sharpf.utils.abc_utils.abc.abc_data import ABCChunk, ABC_7Z_FILEMASK, ABCModality
from sharpf.utils.abc_utils.mesh.io import trimesh_load
from sharpf.utils.py_utils.console import eprint_t


PATCH_TYPES = ['Plane', 'Cylinder', 'Cone', 'Sphere', 'Torus', 'Revolution', 'Extrusion', 'BSpline', 'Other']
CURVE_TYPES = ['Line', 'Circle', 'Ellipse', 'BSpline', 'Other']


def get_patch_info(abc_item):
    mesh, _, _ = trimesh_load(abc_item.obj)
    features = yaml.load(abc_item.feat, Loader=yaml.Loader)

    item_id = str(abc_item.item_id)
    n_verts = len(mesh.vertices)
    n_faces = len(mesh.faces)
    n_surfaces = len(features['surfaces'])
    n_curves = len(features['curves'])

    s = [
        f'{item_id}',
        f'num_verts={n_verts}',
        f'num_faces={n_faces}',
        f'num_surfaces={n_surfaces}',
        f'num_curves={n_curves}',
    ]

    for curve_type in CURVE_TYPES:
        count = len([curve for curve in features['curves'] if curve['type'] == curve_type])
        s.append(f'num_all_curve_{curve_type.lower()}={count}')

    for curve_type in CURVE_TYPES:
        count = len(
            [curve for curve in features['curves'] if curve['type'] == curve_type and curve['sharp']])
        s.append(f'num_sharp_curve_{curve_type.lower()}={count}')

    for surface_type in PATCH_TYPES:
        count = len([surface for surface in features['surfaces'] if surface['type'] == surface_type])
        s.append(f'num_surface_{surface_type.lower()}={count}')

    return ' '.join(s)


def get_patch_worker(
        obj_filename,
        feat_filename,
        item_idx=0,
):
    with ABCChunk([obj_filename, feat_filename]) as data_holder:
        abc_item = data_holder[item_idx]
        s = get_patch_info(abc_item)
        return s


def run_parallel(
        obj_filename,
        feat_filename,
        output_file,
        verbose=False,
        n_jobs=1,
):
    max_queued_tasks = 128
    with ProcessPoolExecutor(max_workers=n_jobs) as executor:
        index_by_future = {}

        for item_idx in range(8000):
            future = executor.submit(get_patch_worker, obj_filename, feat_filename, item_idx)
            index_by_future[future] = item_idx

            if len(index_by_future) >= max_queued_tasks:
                # don't enqueue more tasks until all previously submitted
                # are complete and dumped to disk

                for future in as_completed(index_by_future):
                    item_idx = index_by_future[future]
                    try:
                        s = future.result()
                    except Exception as e:
                        if verbose:
                            eprint_t(f'Error getting item {item_idx}: {str(e)}')
                            eprint_t(traceback.format_exc())
                    else:
                        with open(output_file, 'a') as out_file:
                            out_file.write(f'{item_idx} {s}\n')
                        if verbose:
                            eprint_t(f'Processed item {item_idx}')

                index_by_future = {}

                if verbose:
                    seen_fraction = item_idx / 8000
                    eprint_t(f'Processed {item_idx:d} items ({seen_fraction * 100:3.1f}% of data)')


def run_single_threaded(
        obj_filename,
        feat_filename,
        output_file,
        verbose=False,
):
    with ABCChunk([obj_filename, feat_filename]) as data_holder:
        for item_idx, abc_item in enumerate(data_holder):
            s = get_patch_info(abc_item)
            with open(output_file, 'a') as out_file:
                out_file.write(f'{item_idx} {s}\n')
            if verbose:
                eprint_t(f'Processed item {item_idx}')


def main(options):
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
        eprint_t(f'Obj filename: {obj_filename}, feat filename: {feat_filename}')

    if 1 == options.n_jobs:
        run_single_threaded(
            obj_filename,
            feat_filename,
            options.output_file,
            options.verbose)
    else:
        run_parallel(
            obj_filename,
            feat_filename,
            options.output_file,
            options.verbose,
            options.n_jobs)


def parse_args():
    parser = argparse.ArgumentParser()

    parser.add_argument('-a', '--abc-input-dir', dest='abc_input_dir',
                        required=True, help='input dir with ABC dataset.')
    parser.add_argument('-c', '--chunk', required=True, help='ABC chunk id to process.')
    parser.add_argument('-o', '--output-file', dest='output_file',
                        required=True, help='output file.')
    parser.add_argument('-j', '--jobs', dest='n_jobs',
                        type=int, default=1, help='CPU jobs to use in parallel [default: 1].')

    parser.add_argument('--verbose', dest='verbose', action='store_true', default=False,
                        required=False, help='be verbose')
    return parser.parse_args()


if __name__ == '__main__':
    options = parse_args()
    main(options)
