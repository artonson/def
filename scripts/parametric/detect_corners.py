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
import sharpf.parametric.io as parametric_io
import sharpf.parametric.topological_graph as tg
from sharpf.utils.py_utils.logging import create_logger
from sharpf.utils.abc_utils.hdf5.dataset import Hdf5File, PreloadTypes


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
    parser.add_argument(
        '-j', '--jobs',
        dest='n_jobs',
        default=4,
        type=int,
        required=False,
        help='number of jobs to use for corner extraction.')

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
    logger.debug('Loaded {} points'.format(len(points)))

    try:
        corner_detector_params = {
            'seeds_rate': config['seeds_rate'],
            'corner_detector_radius': config['resolution_3d'] * config['corner_detector_radius_factor'],
            'upper_variance_threshold': config['upper_variance_threshold'],
            'lower_variance_threshold': config['lower_variance_threshold'],
            'cornerness_threshold': config['cornerness_threshold'],
            'corner_connected_components_radius': config['resolution_3d'] * config['knn_radius_factor'],
            'box_margin': config['box_margin'],
            'quantile': config['quantile'],
            'n_jobs': options.n_jobs
        }
        logger.debug('Running a corner detector with params: \n{}'.format(
            '\n'.join(['{} = {}'.format(key, value)for key, value in corner_detector_params.items()])
        ))
        corners, _, corner_centers, init_connections = tg.detect_corners(
            points,
            distances,
            **corner_detector_params)
    except Exception as e:
        logger.error('Cannot load {}: {}; stopping'.format(options.input_filename, str(e)))
        exit(1)

    not_corners = np.setdiff1d(np.arange(len(points)), corners)

    detected_corners = {
        'corners': corners,
        'not_corners': not_corners,
        'corner_centers': corner_centers,
        'init_connections': init_connections,
    }
    logger.debug('Saving detected corners to {}'.format(options.output_filename))
    parametric_io.save_parametric_corners(
        detected_corners,
        options.output_filename)


if __name__ == '__main__':
    options = parse_args()
    main(options)
