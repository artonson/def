#!/usr/bin/env python3

# Given an input dir with predictions (list of npy files)
# and an input file with ground truth (multiple point patches),
# this script:
#  1) builds a single file with ground truth predictions
#  2) runs a series of various prediction consolidation algorithms,
#     saving consolidated predictions into an output folder
#
# Convenient file structure for using this script:
#

import argparse
from functools import partial
import glob
import os
import re
import sys

import h5py
import numpy as np
from tqdm import tqdm
import sklearn.linear_model as lm

__dir__ = os.path.normpath(
    os.path.join(
        os.path.dirname(os.path.realpath(__file__)), '..')
)

sys.path[1:1] = [__dir__]

from sharpf.utils.abc_utils.hdf5.dataset import Hdf5File, PreloadTypes
import sharpf.utils.abc_utils.hdf5.io_struct as io_struct
import sharpf.data.datasets.sharpf_io as sharpf_io
import sharpf.consolidation.combiners as cc
import sharpf.consolidation.smoothers as cs


def convert_npylist_to_hdf5(input_dir, output_filename):
    PointPatchPredictionsIO = io_struct.HDF5IO(
        {'distances': io_struct.VarFloat64('distances')},
        len_label='distances',
        compression='lzf')

    def save_predictions(patches, filename):
        collate_fn = partial(io_struct.collate_mapping_with_io, io=PointPatchPredictionsIO)
        patches = collate_fn(patches)
        with h5py.File(filename, 'w') as f:
            PointPatchPredictionsIO.write(f, 'distances', patches['distances'])

    def get_num(basename):
        match = re.match('^test_(\d+)\.npy$', basename)
        return int(match.groups()[0])

    datafiles = glob.glob(os.path.join(input_dir, '*.npy'))
    datafiles.sort(key=lambda name: get_num(os.path.basename(name)))
    patches = [{'distances': np.load(f)} for f in datafiles]
    save_predictions(patches, output_filename)


def save_full_model_predictions(points, predictions, filename):
    PointPatchPredictionsIO = io_struct.HDF5IO(
        {'points': io_struct.Float64('points'),
         'distances': io_struct.Float64('distances')},
        len_label='distances',
        compression='lzf')
    with h5py.File(filename, 'w') as f:
        PointPatchPredictionsIO.write(f, 'points', [points])
        PointPatchPredictionsIO.write(f, 'distances', [predictions])


def main(options):
    name = os.path.splitext(os.path.basename(options.true_filename))[0]

    UnlabeledPointCloudIO = io_struct.HDF5IO({
            'points': io_struct.Float64('points'),
            'indexes_in_whole': io_struct.Int32('indexes_in_whole'),
            'distances': io_struct.Float64('distances'),
            'item_id': io_struct.AsciiString('item_id'),
        },
        len_label='has_sharp',
        compression='lzf')

    if options.unlabeled:
        data_io = UnlabeledPointCloudIO
    else:
        data_io = sharpf_io.WholePointCloudIO

    # load ground truth and save to a single patch
    ground_truth_dataset = Hdf5File(
        options.true_filename,
        io=data_io,
        preload=PreloadTypes.LAZY,
        labels='*')
    ground_truth = [patch for patch in ground_truth_dataset]

    n_points = np.concatenate([patch['indexes_in_whole'] for patch in ground_truth]).max() + 1
    if n_points >= 10_000_000:
        print('Too large file ({} points); not computing'.format(n_points))
        return

    whole_model_points_gt = np.zeros((n_points, 3))
    whole_model_distances_gt = np.ones(n_points) * np.inf
    whole_model_directions_gt = np.ones((n_points, 3)) * np.inf

    for patch in tqdm(ground_truth):
        distances = patch['distances']
        directions = patch['directions'].reshape((-1, 3))
        indexes = patch['indexes_in_whole'].astype(int)
        whole_model_points_gt[indexes] = patch['points'].reshape((-1, 3))

        assign_mask = whole_model_distances_gt[indexes] > distances
        whole_model_distances_gt[indexes[assign_mask]] = np.minimum(distances[assign_mask], 1.0)
        whole_model_directions_gt[indexes[assign_mask]] = directions[assign_mask]

    ground_truth_filename = os.path.join(options.output_dir, '{}__{}.hdf5'.format(name, 'ground_truth'))
    save_full_model_predictions(whole_model_points_gt, whole_model_distances_gt, ground_truth_filename)

    # convert and load predictions
    if options.pred_key is not None:
        ComparisonsIO = io_struct.HDF5IO({
                'points': io_struct.VarFloat64('points'),
                'indexes_in_whole': io_struct.VarInt32('indexes_in_whole'),
                'distances': io_struct.VarFloat64('distances'),
                'item_id': io_struct.AsciiString('item_id'),
                'voronoi': io_struct.VarFloat64('voronoi'),
                'ecnet': io_struct.VarFloat64('ecnet'),
                'sharpness': io_struct.VarFloat64('sharpness'),
            },
            len_label='points',
            compression='lzf')
        predictions_dataset = Hdf5File(
            options.pred_dir,
            io=ComparisonsIO,
            preload=PreloadTypes.LAZY,
            labels=[options.pred_key, 'points', 'indexes_in_whole'])
        list_predictions = [patch[options.pred_key] for patch in predictions_dataset]
        
    else:
        predictions_filename = os.path.join(options.output_dir, '{}__{}.hdf5'.format(name, 'predictions'))
        convert_npylist_to_hdf5(options.pred_dir, predictions_filename)
        predictions_dataset = Hdf5File(
            predictions_filename,
            io=sharpf_io.WholePointCloudIO,
            preload=PreloadTypes.LAZY,
            labels='*')
        list_predictions = [patch['distances'] for patch in predictions_dataset]

    # run various algorithms for consolidating predictions
    combiners = [
#       AvgProbaPredictionsCombiner(thr=0.25),
#       AvgProbaPredictionsCombiner(thr=0.5),
#       AvgProbaPredictionsCombiner(thr=0.75),
#       MedianPredictionsCombiner(),
#       MinPredictionsCombiner(),
#       AvgPredictionsCombiner(),
#       TruncatedAvgPredictionsCombiner(),
#       CenterCropPredictionsCombiner(brd_thr=80, func=np.min, tag='crop__min'),
#       MinsAvgPredictionsCombiner(signal_thr=0.9),
#       SmoothingCombiner(
#           combiner=CenterCropPredictionsCombiner(brd_thr=80, func=np.min),
#           smoother=L2Smoother(regularizer_alpha=0.01)
#       ),
#       SmoothingCombiner(
#           combiner=CenterCropPredictionsCombiner(brd_thr=80, func=np.min),
#           smoother=TotalVariationSmoother(regularizer_alpha=0.001)
#       ),
        # cc.SmoothingCombiner(
        #     combiner=cc.CenterCropPredictionsCombiner(brd_thr=80, func=np.min),
        #     smoother=cs.RobustLocalLinearFit(
        #         lm.HuberRegressor(epsilon=4., alpha=1.),
        #         n_jobs=32
        #     )
        # ),
        # cc.CenterCropPredictionsCombiner(brd_thr=80, func=cc.TruncatedMean(0.6, func=np.min), tag='crop__adv60__min'),
        # cc.MinPredictionsCombiner(),
        # cc.CenterCropPredictionsCombiner(brd_thr=80, func=np.min, tag='crop__min')
        cc.AvgProbaPredictionsCombiner(),
    ]

    list_indexes_in_whole = [patch['indexes_in_whole'] for patch in ground_truth_dataset]
    list_points = [patch['points'].reshape((-1, 3)) for patch in ground_truth_dataset]

    for combiner in combiners:
        if options.verbose:
            print('Running {}'.format(combiner.tag))

        combined_predictions, prediction_variants = \
            combiner(n_points, list_predictions, list_indexes_in_whole, list_points)

        output_filename = os.path.join(options.output_dir, '{}__{}.hdf5'.format(name, combiner.tag))
        save_full_model_predictions(whole_model_points_gt, combined_predictions, output_filename)


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('-v', '--verbose', action='store_true', default=False, dest='verbose',
                        help='verbose output [default: False].')
    parser.add_argument('-w', '--overwrite', action='store_true', default=False, dest='overwrite',
                        help='overwrite each output [default: False].')

    parser.add_argument('-t', '--true-filename', dest='true_filename', required=True,
                        help='Path to GT file with whole model point patches.')
    parser.add_argument('-p', '--pred-dir', dest='pred_dir', required=True,
                        help='Path to prediction directory with npy files.')
    parser.add_argument('-o', '--output-dir', dest='output_dir', required=True,
                        help='Path to output (suffixes indicating various methods will be added).')
    parser.add_argument('-u', '--unlabeled', dest='unlabeled', action='store_true', default=False,
                        help='set if input data is unlabeled.')
    parser.add_argument('-k', '--key', dest='pred_key', help='if set, switch to compare-io and use this key.')
    parser.add_argument('-fl', '--files-list')
    return parser.parse_args()

def main_errors(options):
    try: 
        main(options)
    except Exception as e:
        print(e)

if __name__ == '__main__':
    from copy import deepcopy
    from multiprocessing import Pool
    from tqdm import tqdm
    import pathlib
    import os

    options = parse_args()
    pool_args = []

    # with open(options.files_list, "r") as f:
    #     for dirname in f:
    #         pair = []
    #         dirname = dirname.strip()
    #         if "points" not in dirname:
    #             continue
    #         for fname in os.listdir(pathlib.Path(options.true_filename) / dirname):
    #             input_path = pathlib.Path(options.true_filename) / dirname / fname
    #             pred_path = (
    #                 pathlib.Path(options.pred_dir) / 
    #                 (str(dirname).split(".json")[0].replace("data_v2_cvpr/", "") + "_" +
    #                 "".join(str(dirname).split(".json")[1].split("/")))
    #             )
    #             pred_path = pred_path / os.listdir(str(pred_path))[0] / input_path.stem / "predictions"
    #             out_path = pathlib.Path(options.output_dir) / dirname / input_path.stem
    #             try:
    #                 out_path.mkdir(parents=True, exist_ok=False)
    #             except FileExistsError:
    #                 pass
    #             pool_args.append((input_path, pred_path, out_path))

    for tfname in os.listdir(options.true_filename):
        name = tfname.split(".")[0]
        input_path = f"{options.true_filename}/{tfname}"
        pred_dir = f"{options.pred_dir}/{name}/predictions"
        # out_path = pathlib.Path(f"{options.output_dir}/{name}/")
        out_path = pathlib.Path(f"{options.output_dir}/")
        try:
            out_path.mkdir(parents=True, exist_ok=False)
        except FileExistsError:
            pass
        pool_args.append((input_path, pred_dir, out_path))

    # for tfname in list(pathlib.Path(options.true_filename).rglob("*.hdf5")):
    #     stem = tfname.stem
    #     input_path = tfname
    #     pred_dir = f"{options.pred_dir}/{tfname.name}"
    #     out_path = pathlib.Path(f"{options.output_dir}")
    #     try:
    #         out_path.mkdir(parents=True, exist_ok=False)
    #     except FileExistsError:
    #         pass
    #     pool_args.append((input_path, pred_dir, out_path))        

    pool_options = []
    for arg in pool_args:
        local_options = deepcopy(options)
        local_options.true_filename = str(arg[0])
        local_options.pred_dir = str(arg[1])
        local_options.output_dir = str(arg[2])
        pool_options.append(local_options)
    
    pool = Pool(processes=20)
    print("=== Start multiprocessing pool")
    r = list(tqdm(pool.imap(main_errors, pool_options), total=len(pool_options)))
