#!/usr/bin/env python3

import argparse
import os
import sys

import numpy as np
import yaml

__dir__ = os.path.normpath(
    os.path.join(
        os.path.dirname(os.path.realpath(__file__)), '..', '..')
)
sys.path[1:1] = [__dir__]

from sharpf.data.patch_cropping import farthest_point_sampling
import sharpf.fusion.io as fusion_io
import sharpf.parametric.topological_graph as tg
from sharpf.utils.abc_utils.hdf5.dataset import Hdf5File, PreloadTypes
from sharpf.utils.py_utils.logging import create_logger


def parse_args():
    parser = argparse.ArgumentParser(
        description='Given an input set of fused points with predictions, '
                    'produce a subsampled set of points to extract parametric curves.',
        formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parser.add_argument(
        '--verbose',
        dest='verbose',
        action='store_true',
        default=False,
        help='be verbose')
    parser.add_argument(
        '--logging_filename',
        dest='logging_filename',
        help='specifies an output filename to log to.')

    parser.add_argument(
        '-i', '--input',
        dest='input_filename',
        required=True,
        help='path to file with fused points and predictions.')
    parser.add_argument(
        '-id', '--input-distances',
        dest='input_distances_filename',
        required=False,
        help='path to file with fused predictions; if specified, '
             'this replaces predictions from INPUT_FILENAME.')

    parser.add_argument(
        '-o', '--output',
        dest='output_filename',
        required=True,
        help='path to save the filtered results (in the same format as input)')

    parser.add_argument(
        '-c', '--config',
        dest='config',
        required=True,
        help='path to file with configuration.')

    return parser.parse_args()


def main(options):
    config = yaml.load(options.config)
    logger = create_logger(options)

    try:
        logger.debug('Loading data from {}'.format(options.input_filename))
        dataset = Hdf5File(
            options.input_filename,
            io=fusion_io.FusedPredictionsIO,
            preload=PreloadTypes.LAZY,
            labels='*')
        points, distances = dataset[0]['points'], dataset[0]['distances']
    except Exception as e:
        logger.error('Cannot load {}: {}; stopping'.format(options.input_filename, str(e)))
        exit(1)

    if None is not options.input_distances_filename:
        try:
            logger.debug('Loading predicted distances from {}'.format(options.input_distances_filename))
            dataset = Hdf5File(
                options.input_distances_filename,
                io=fusion_io.FusedPredictionsIO,
                preload=PreloadTypes.LAZY,
                labels=['distances'])
            distances = dataset[0]['distances']
        except Exception as e:
            logger.error('Cannot load {}: {}; stopping'.format(options.input_distances_filename, str(e)))
            exit(1)

    logger.debug('Loaded {} points'.format(len(points)))

    distances_near_threshold = config['resolution_3d'] * config['distances_near_thr_factor']
    logger.debug(
        'Filtering input fused points/distances by selecting points '
        'closer than {} to sharp feature curves'.format(distances_near_threshold))
    indexes_to_keep_mask = distances < distances_near_threshold
    points, distances = points[indexes_to_keep_mask], distances[indexes_to_keep_mask]
    logger.debug('Selected a subset of close points containing {} points'.format(len(points)))

    if config['min_cc_points_to_keep'] > 0:
        knn_absolute_radius = config['resolution_3d'] * config['knn_radius_factor']
        logger.debug(
            'Filtering input fused points/distances by removing clusters of points '
            'with pairwise distances less than {} and number of nodes less than {}'.format(
                knn_absolute_radius, config['min_cc_points_to_keep']))
        try:
            filtered_clusters = tg.separate_points_to_subsets(
                points,
                knn_radius=knn_absolute_radius,
                filtering_mode=True,
                min_cluster_size_to_return=config['min_cc_points_to_keep'])
        except Exception:
            logger.error('Cannot run separate_graph_connected_components; skipping this step')
        else:
            indexes_to_keep = np.unique(np.concatenate(filtered_clusters))
            points, distances = points[indexes_to_keep], distances[indexes_to_keep]
            logger.debug('Selected a subset of close points containing {} points'.format(len(points)))

    if config['subsample_rate'] > 0:
        points_to_sample = np.ceil(len(points) * config['subsample_rate']).astype(np.int)
        logger.debug(
            'Subsampling input fused points/distances '
            'by randomly choosing {} points out of {}'.format(points_to_sample, len(points)))
        try:
            indexes_to_keep, _ = farthest_point_sampling(points, k=points_to_sample)
        except Exception:
            logger.error('Cannot run subsampling; skipping this step')
        else:
            indexes_to_keep = indexes_to_keep[0]  # batch of size 1
            points, distances = points[indexes_to_keep], distances[indexes_to_keep]
            logger.debug('Selected a subset of close points containing {} points'.format(len(points)))

    fusion_io.save_full_model_predictions(
        points,
        distances,
        options.output_filename)
    logger.debug('Saved selected points to {}'.format(options.output_filename))


if __name__ == '__main__':
    options = parse_args()
    main(options)
