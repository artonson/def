#!/usr/bin/env python3

import argparse
from itertools import chain
import json
import os
import sys

import trimesh
from joblib import Parallel, delayed

__dir__ = os.path.normpath(
    os.path.join(
        os.path.dirname(os.path.realpath(__file__)), '..', '..')
)
sys.path[1:1] = [__dir__]

from sharpf.utils.shape import load_from_options
from sharpf.data.abc.abc_data import ABCModality, ABCChunk, ABC_7Z_FILEMASK, MergedABCItem


@delayed
def filter_meshes_worker(filter_fn, data_filename, slice_start, slice_end):
    processed_items = []
    with ABCChunk([data_filename]) as data_holder:
        for item in data_holder[slice_start:slice_end]:
            item_trm = MergedABCItem(
                pathname_by_modality=item.pathname_by_modality,
                archive_pathname_by_modality=item.archive_pathname_by_modality,
                item_id=item.item_id,
                obj=trimesh.load(item.obj, 'obj'))
            is_ok = filter_fn(item_trm)
            processed_items.append((item.pathname_by_modality, item.archive_pathname_by_modality, item.item_id, is_ok))
    return processed_items


def filter_meshes(options):
    """Filter the shapes using a number of filters, saving intermediate results."""

    # create the required filter objects
    with open(options.filter_config) as filter_config:
        json_config = json.load(filter_config)
        shape_filter = load_from_options(json_config)

    # create the data source iterator
    # abc_data = ABCData(options.data_dir,
    #                    modalities=[ABCModality.FEAT.value, ABCModality.OBJ.value],
    #                    chunks=[options.chunk],
    #                    shape_representation='trimesh')

    obj_filename = os.path.join(
        options.input_dir,
        ABC_7Z_FILEMASK.format(
            chunk=options.chunk.zfill(4),
            modality=ABCModality.OBJ.value,
            version='00'
        )
    )
    # feat_filename = os.path.join(
    #     options.data_dir,
    #     ABC_7Z_FILEMASK.format(
    #         chunk=options.chunk.zfill(4),
    #         modality=ABCModality.FEAT.value,
    #         version='00'
    #     )
    # )
    abc_data = ABCChunk([obj_filename])

    processes_to_spawn = 10 * options.n_jobs
    chunk_size = len(abc_data) // processes_to_spawn
    abc_data_slices = [(start, start + chunk_size)
                       for start in range(0, len(abc_data), chunk_size)]

    # run the filtering job in parallel
    parallel = Parallel(n_jobs=options.n_jobs)
    delayed_iterable = (filter_meshes_worker(shape_filter, obj_filename, *data_slice)
                        for data_slice in abc_data_slices)
    output_by_slice = parallel(delayed_iterable)

    # write out the results
    output_filename = os.path.join(options.output_dir,'{}.txt'.format(options.chunk.zfill(4)))
    with open(output_filename, 'w') as output_file:
        output_file.write('\n'.join([
            '{} {}'.format(pathname_by_modality['obj'], archive_pathname_by_modality['obj'])
            for pathname_by_modality, archive_pathname_by_modality, item_id, is_ok in chain(*output_by_slice) if is_ok
        ]))


def parse_args():
    parser = argparse.ArgumentParser()

    parser.add_argument('-j', '--jobs', dest='n_jobs',
                        type=int, default=4, help='CPU jobs to use in parallel [default: 4].')
    parser.add_argument('-i', '--input-dir', dest='input_dir',
                        required=True, help='input dir with ABC dataset.')
    parser.add_argument('-c', '--chunk', required=True, help='ABC chunk id to process.')
    parser.add_argument('-o', '--output-dir', dest='output_dir',
                        required=True, help='output dir.')
    parser.add_argument('-g', '--filter-config', dest='filter_config',
                        required=True, help='filter configuration file.')

    return parser.parse_args()


if __name__ == '__main__':
    options = parse_args()
    filter_meshes(options)
