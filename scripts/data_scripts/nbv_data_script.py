#!/usr/bin/env python3

import argparse
import os
import sys
import traceback

from joblib import Parallel, delayed
import yaml

__dir__ = os.path.normpath(
    os.path.join(
        os.path.dirname(os.path.realpath(__file__)), '..', '..')
)

sys.path[1:1] = [__dir__]

from sharpf.data.abc.abc_data import ABCModality, ABCChunk, ABC_7Z_FILEMASK
from sharpf.utils.common import eprint_t


def generate_patches(meshes_filename, feats_filename, data_slice, max_surfaces, output_dir):
    slice_start, slice_end = data_slice
    with ABCChunk([meshes_filename, feats_filename]) as data_holder:
        for item in data_holder[slice_start:slice_end]:
            eprint_t("Processing chunk file {chunk}, item {item}".format(
                chunk=meshes_filename, item=item.item_id))
            try:
                # load the mesh and the feature curves annotations
                features = yaml.load(item.feat, Loader=yaml.Loader)

                if len(features['surfaces']) < max_surfaces:
                    mesh_as_string = item.obj.read().decode('utf-8')
                    out_filename = os.path.join(output_dir, item.item_id + '.obj')
                    with open(out_filename, 'w') as out_file:
                        out_file.write(mesh_as_string)

            except Exception as e:
                eprint_t('Error processing item {item_id} from chunk {chunk}: {what}'.format(
                    item_id=item.item_id, chunk='[{},{}]'.format(meshes_filename, feats_filename), what=e))
                eprint_t(traceback.format_exc())

            else:
                eprint_t('Done processing item {item_id} from chunk {chunk}'.format(
                    item_id=item.item_id, chunk='[{},{}]'.format(meshes_filename, feats_filename)))


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

    parallel = Parallel(n_jobs=options.n_jobs, backend='multiprocessing')
    delayed_iterable = (delayed(generate_patches)(obj_filename, feat_filename, data_slice,
                                                  options.max_surfaces, options.output_dir)
                        for data_slice in abc_data_slices)
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
    parser.add_argument('-m', '--max-surfaces', dest='max_surfaces',
                        required=True, help='select only files with surfaces not exceeding this value.')
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
