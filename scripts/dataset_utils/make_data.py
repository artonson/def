#!/usr/bin/env python3

import argparse

from sharpf.data.abc_data import ABCData

from joblib import Parallel, delayed
import datetime
from abc import ABC

import numpy as np
import yaml

# Method scheme:
# 1) Obtain an archive filename as input.
# 2) Locate the corresponding object and feature files.
# 3) Filter the shapes using a number of filters, saving intermediate results.
# 4) Crop patches from the shapes, saving to intermediate output files.
# 5) Save dataset from the patches, saving to torch.FloatStorage .


def filter_shapes_worker(filter_fn, ):

    for filter_fn in filters:
        filter(filter_fn, shapes)



def filter_shapes(options):
    """Filter the shapes using a number of filters, saving intermediate results."""

    sequential_filter =

    abc_data = ABCData(options.data_dir, modalities=['feat'])

    parallel = Parallel(n_jobs=options.n_jobs)
    delayed_iterable = (delayed(filter_shapes_worker)(item) for item in abc_data)
    output = parallel(delayed_iterable)



def make_patches(options):
    pass



def make_torch_storage(options):
    pass


def parse_args():
    parser = argparse.ArgumentParser()

    parser.add_argument('-j', '--jobs', type=int, default=4, help='CPU jobs to use in parallel [default: 4].')

    subparsers = parser.add_subparsers(dest='command', help='sub-command help')

    # create the parser for the "filter" command
    stats_parser = subparsers.add_parser('filter', help='select meshes according to the specified filters.')
    stats_parser.add_argument('--data-root', required=True, dest='data_root', help='root of the data tree (directory).')

    stats_parser.add_argument('bar', type=int, help='bar help')
    stats_parser.add_argument('bar', type=int, help='bar help')
    stats_parser.add_argument('bar', type=int, help='bar help')
    stats_parser.add_argument('bar', type=int, help='bar help')
    stats_parser.add_argument('bar', type=int, help='bar help')


    # create the parser for the "patches" command
    patches_parser = subparsers.add_parser('patches', help='generate patches ')
    patches_parser.add_argument('bar', type=int, help='bar help')



    # create the parser for the "dataset" command
    dataset_parser = subparsers.add_parser('dataset', help='generate torch-based dataset from selected patches.')
    dataset_parser.add_argument('bar', type=int, help='bar help')


    parser.add_argument('-e', '--epochs', type=int, default=1, help='how many epochs to train [default: 1].')
    parser.add_argument('-b', '--train-batch-size', type=int, default=128, dest='train_batch_size',
                        help='train batch size [default: 128].')
    parser.add_argument('-B', '--val-batch-size', type=int, default=128, dest='val_batch_size',
                        help='val batch size [default: 128].')
    parser.add_argument('--batches-before-val', type=int, default=1024, dest='batches_before_val',
                        help='how many batches to train before validation [default: 1024].')
    parser.add_argument('--batches-before-imglog', type=int, default=12, dest='batches_before_imglog',
                        help='log images each batches-before-imglog validation batches [default: 12].')
    parser.add_argument('--mini-val-batches-n-per-subset', type=int, default=12, dest='mini_val_batches_n_per_subset',
                        help='how many batches per subset to run for mini validation [default: 12].')

    parser.add_argument('--model-spec', dest='model_spec_filename', required=True,
                        help='model specification JSON file to use [default: none].')
    parser.add_argument('--infer-from-spec', dest='infer_from_spec', action='store_true', default=False,
                        help='if set, --model, --save-model-file, --logging-file, --tboard-json-logging-file,'
                             'and --tboard-dir are formed automatically [default: False].')
    parser.add_argument('--log-dir-prefix', dest='logs_dir', default='/logs',
                        help='path to root of logging location [default: /logs].')
    parser.add_argument('-m', '--init-model-file', dest='init_model_filename',
                        help='Path to initializer model file [default: none].')

    parser.add_argument('-s', '--save-model-file', dest='save_model_filename',
                        help='Path to output vectorization model file [default: none].')
    parser.add_argument('--batches_before_save', type=int, default=1024, dest='batches_before_save',
                        help='how many batches to run before saving the model [default: 1024].')

    parser.add_argument('--data-root', required=True, dest='data_root', help='root of the data tree (directory).')
    parser.add_argument('--data-type', required=True, dest='dataloader_type',
                        help='type of the train/val data to use.', choices=dataloading.prepare_loaders.keys())
    parser.add_argument('--handcrafted-train', required=False, action='append',
                        dest='handcrafted_train_paths', help='dirnames of handcrafted datasets used for training '
                                                             '(sought for in preprocessed/synthetic_handcrafted).')
    parser.add_argument('--handcrafted-val', required=False, action='append',
                        dest='handcrafted_val_paths', help='dirnames of handcrafted datasets used for validation '
                                                           '(sought for in preprocessed/synthetic_handcrafted).')
    parser.add_argument('--handcrafted-val-part', required=False, type=float, default=.1,
                        dest='handcrafted_val_part', help='portion of handcrafted_train used for validation')
    parser.add_argument('-M', '--memory-constraint', required=True, type=int, dest='memory_constraint',help='maximum RAM usage in bytes.')

    parser.add_argument('-r', '--render-resolution', dest='render_res', default=64, type=int,
                        help='resolution used for rendering.')

    parser.add_argument('--verbose', action='store_true', default=False, dest='verbose',
                        help='verbose output [default: False].')
    parser.add_argument('-l', '--logging-file', dest='logging_filename',
                        help='Path to output logging text file [default: output to stdout only].')
    parser.add_argument('-tl', '--tboard-json-logging-file', dest='tboard_json_logging_file',
                        help='Path to output logging JSON file with scalars [default: none].')
    parser.add_argument('-x', '--tboard-dir', dest='tboard_dir',
                        help='Path to tensorboard [default: do not log events].')
    parser.add_argument('-w', '--overwrite', action='store_true', default=False,
                        help='If set, overwrite existing logs [default: exit if output dir exists].')

    return parser.parse_args()


if __name__ == '__main__':
    options = parse_args()

    function_to_run = {
        'filter': filter_shapes,
        'patch': make_patches,
        'dataset': make_torch_storage,
    }[options.command]

    function_to_run(options)
