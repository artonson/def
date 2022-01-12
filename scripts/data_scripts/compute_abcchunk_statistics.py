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


def get_patch_info(
    obj_filename,
    feat_filename,
    item_idx=0,
):
    with ABCChunk([obj_filename, feat_filename]) as data_holder:
        abc_item = data_holder[item_idx]

        mesh, _, _ = trimesh_load(abc_item.obj)
        features = yaml.load(abc_item.feat, Loader=yaml.Loader)

        item_id = str(abc_item['item_id'].decode('utf-8'))
        n_verts = len(mesh.vertices)
        n_faces = len(mesh.faces)
        n_surfaces = len(features['surfaces'])
        n_curves = len(features['curves'])

        return f'{item_id} {n_verts} {n_faces} {n_surfaces} {n_curves}'


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

    max_queued_tasks = 1024
    with ProcessPoolExecutor(max_workers=options.n_jobs) as executor:
        index_by_future = {}

        for item_idx in range(8000):
            future = executor.submit(get_patch_info, obj_filename, feat_filename, item_idx)
            index_by_future[future] = item_idx

            if len(index_by_future) >= max_queued_tasks:
                # don't enqueue more tasks until all previously submitted
                # are complete and dumped to disk

                for future in as_completed(index_by_future):
                    item_idx, item_id = index_by_future[future]
                    try:
                        s = future.result()
                    except Exception as e:
                        if options.verbose:
                            eprint_t(f'Error getting item {item_id}: {str(e)}')
                            eprint_t(traceback.format_exc())
                    else:
                        with open(options.output_file, 'a') as out_file:
                            lines = [f'{item_id} {item_idx} ' + line for line in s]
                            out_file.write('\n'.join(lines) + '\n')
                        if options.verbose:
                            eprint_t(f'Processed item {item_id}, {item_idx}')

                index_by_future = {}

                if options.verbose:
                    seen_fraction = item_idx / 8000
                    eprint_t(f'Processed {item_idx:d} items ({seen_fraction * 100:3.1f}% of data)')


def parse_args():
    parser = argparse.ArgumentParser()

    parser.add_argument('-a', '--abc-input-dir', dest='abc_input_dir',
                        required=True, help='input dir with ABC dataset.')
    parser.add_argument('-c', '--chunk', required=True, help='ABC chunk id to process.')
    parser.add_argument('-o', '--output-file', dest='output_file',
                        required=True, help='output file.')
    parser.add_argument('-j', '--jobs', dest='n_jobs',
                        type=int, default=4, help='CPU jobs to use in parallel [default: 4].')

    parser.add_argument('--verbose', dest='verbose', action='store_true', default=False,
                        required=False, help='be verbose')
    return parser.parse_args()


if __name__ == '__main__':
    options = parse_args()
    main(options)
