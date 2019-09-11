#!/usr/bin/env python3

import argparse
import os

# from scripts.dataset_utils.shape import SequentialFilter, load_from_options
# from sharpf.data.abc_data import ABCData, ABCModality

from joblib import Parallel, delayed


@delayed
def filter_meshes_worker(filter_fn, item):
    is_ok = filter_fn(item.obj)
    return item.pathname, item.archive_filename, is_ok


@delayed
def test(item):
    pid = os.getpid()
    import socket
    host = socket.gethostname()
    chunk, i = item
    import numpy as np 
    size = 4096
    A = np.random.random((size, size))
    svd = np.linalg.svd(A, full_matrices=False)
    return "On host {} process {} has chunk {} and input {}".format(host, pid, chunk, i)


def filter_meshes(options):
    """Filter the shapes using a number of filters, saving intermediate results."""

    # # create the required filter objects
    # sequential_filter = SequentialFilter([
    #     load_from_options(filter_name, options.__dict__)
    #     for filter_name in options.filters])
    #
    # # create the data source iterator
    # abc_data = ABCData(options.data_dir,
    #                    modalities=[ABCModality.FEAT.value, ABCModality.OBJ.value],
    #                    chunks=[options.chunk],
    #                    shape_representation='trimesh')

    # TEST: run `jobs` number of jobs
    abc_data = [(options.chunk, i) for i in range(options.jobs)]

    # run the filtering job in parallel
    parallel = Parallel(n_jobs=options.jobs)
    # delayed_iterable = (filter_meshes_worker(sequential_filter, item) for item in abc_data)
    delayed_iterable = (test(item) for item in abc_data)
    output = parallel(delayed_iterable)

    # write out the results
    output_filename = os.path.join(options.output_dir,'{}.txt'.format(str(options.chunk).zfill(4)))
    with open(output_filename, 'w') as output_file:
        output_file.write('\n'.join(output) + '\n')
#       output_file.write('\n'.join([
#           '{} {}'.format(pathname, archive_filename)
#           for pathname, archive_filename, is_ok in output if is_ok
#       ]))


def parse_args():
    parser = argparse.ArgumentParser()

    parser.add_argument('-j', '--jobs', type=int, default=4, help='CPU jobs to use in parallel [default: 4].')
    parser.add_argument('-i', '--input-dir', required=True, help='input dir with ABC dataset.')
    parser.add_argument('-c', '--chunk', type=int, required=True, help='ABC chunk id to process.')
    parser.add_argument('-o', '--output-dir', required=True, help='output dir.')

    return parser.parse_args()


if __name__ == '__main__':
    options = parse_args()
    filter_meshes(options)
