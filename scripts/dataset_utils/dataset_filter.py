#!/usr/bin/env python3

import argparse
import json
import os

import trimesh

from scripts.dataset_utils.shape import load_from_options
from sharpf.data.abc_data import ABCData, ABCModality, ABCChunk, ABC_7Z_FILEMASK, MergedABCItem

from joblib import Parallel, delayed


@delayed
def filter_meshes_worker(filter_fn, item):
    item_trm = MergedABCItem(
        pathname_by_modality=item.pathname_by_modality,
        archive_pathname_by_modality=item.archive_pathname_by_modality,
        item_id=item.item_id,
        obj=trimesh.load(item.obj, 'obj'))
    is_ok = filter_fn(item_trm)
    return item.pathname, item.archive_filename, item.item_id, is_ok


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
        options.data_dir,
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

    # run the filtering job in parallel
    parallel = Parallel(n_jobs=options.n_jobs)
    delayed_iterable = (filter_meshes_worker(shape_filter, item) for item in abc_data)
    output = parallel(delayed_iterable)

    # write out the results
    output_filename = os.path.join(options.output_dir,'{}.txt'.format(options.chunk.zfill(4)))
    with open(output_filename, 'w') as output_file:
        output_file.write('\n'.join([
            '{} {}'.format(pathname, archive_filename)
            for pathname, archive_filename, is_ok in output if is_ok
        ]))


def parse_args():
    parser = argparse.ArgumentParser()

    parser.add_argument('-j', '--jobs', type=int, default=4, help='CPU jobs to use in parallel [default: 4].')
    parser.add_argument('-i', '--input-dir', required=True, help='input dir with ABC dataset.')
    parser.add_argument('-c', '--chunk', required=True, help='ABC chunk id to process.')
    parser.add_argument('-o', '--output-dir', required=True, help='output dir.')
    parser.add_argument('-g', '--filter-config', dest='filter_config',
                        required=True, help='filter configuration file.')

    return parser.parse_args()


if __name__ == '__main__':
    options = parse_args()
    filter_meshes(options)
