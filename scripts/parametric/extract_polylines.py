import argparse
import os
import sys

import numpy as np

__dir__ = os.path.normpath(
    os.path.join(
        os.path.dirname(os.path.realpath(__file__)), '..', '..')
)

import yaml

sys.path[1:1] = [__dir__]

import sharpf.parametric.io as parametric_io
from sharpf.data.patch_cropping import farthest_point_sampling
import sharpf.fusion.io as fusion_io
import sharpf.parametric.optimization as opt
import sharpf.parametric.topological_graph as tg
import sharpf.parametric.utils as utils
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
        '-r', '--corners',
        dest='corners_filename',
        required=True,
        help='path to file with corner predictions.')

    parser.add_argument(
        '-o', '--output',
        dest='output_filename',
        required=True,
        help='path to save the initialised topological graph.')


    parser.add_argument('--res', type=float, default=0.02, 
                        help='point cloud resolution (avg pointwise distance) [default: 0.02]')
    parser.add_argument('--sharp', type=float, default=1.5, 
                        help='sharpness threshold [default: 1.5]')
    
    parser.add_argument('--knn_radius', type=float, default=3, 
                        help='max distance to connect a pair in knn graph [default: 3]')
    parser.add_argument('--filt_factor', type=int, default=30, 
                        help='min number of points in connected component to get through the filtering [default: 30]')
    parser.add_argument('--filt_mode', type=bool, default=True, 
                        help='if do filtering [default: True]')
    
    parser.add_argument('--subsample', type=int, default=0, 
                        help='if subsample [default: 0]')
    parser.add_argument('--fps_factor', type=int, default=5, 
                        help='how much less points to sample for fps [default: 5]')
    
    parser.add_argument('--corner_R', type=float, default=6, 
                        help='ball radius for corner detection [default: 6]')
#     parser.add_argument('--corner_r', type=float, default=4, 
#                         help='ball radius for corner separation [default: 4]')
    parser.add_argument('--corner_up_thr', type=float, default=0.2, 
                        help='upper variance threshold to compute cornerness [default: 0.2]')
    parser.add_argument('--corner_low_thr', type=float, default=0.1, 
                        help='lower variance threshold to compute cornerness [default: 0.1]')
    parser.add_argument('--cornerness', type=float, default=1.25, 
                        help='threshold to consider neighbourhood as a corner [default: 1.25]')
    parser.add_argument('--quantile', type=float, default=0.8, 
                        help='anti-double corner rate quantile [default: 0.8]')
    parser.add_argument('--box_margin', type=float, default=1.5, 
                        help='anti-double corner rate quantile [default: 1.5]')
    
    parser.add_argument('--endpoint_R', type=float, default=6, 
                        help='ball radius for endpoint detection [default: 6]')
    parser.add_argument('--endpoint_thr', type=float, default=0.4, 
                        help='threshold to consider neighbourhood as an enpoint [default: 0.4]')
    
    parser.add_argument('--connect_R', type=float, default=20, 
                        help='distance for endpoint connection to the corners [default: 20]')
    
    parser.add_argument('--init_thr', type=float, default=3, 
                        help='initial polyline subdivision distance [default: 3]')
    parser.add_argument('--opt_thr', type=float, default=3, 
                        help='final polyline subdivision distance [default: 3]')
#     parser.add_argument('--alpha_fid', type=float, default=0, 
#                         help='corner optimization fidelity term weight [default: 0]')
#     parser.add_argument('--alpha_fit', type=float, default=1, 
#                         help='corner optimization fit term weight [default: 1]')
    parser.add_argument('--alpha_ang', type=float, default=1, 
                        help='corner optimization rigidity term weight [default: 1]')
    
    parser.add_argument('--draw', type=bool, default=True, 
                        help='whether to draw result [default: True]')

    return parser.parse_args()


def main(options):
    path_to_points = options.points
    #     path_to_preds = options.preds
    path_to_save = options.save_folder

    RES = options.res
    sharpness_threshold = RES * options.sharp

    filtering_radius = RES * options.knn_radius  # max distance to connect a pair in knn filtering
    corner_connected_components_radius = RES * options.knn_radius  # max distance to connect a pair in knn corner separation
    curve_connected_components_radius = RES * options.knn_radius  # max distance to connect a pair in knn curve separation

    subsample_rate = options.subsample
    filtering_factor = options.filt_factor
    filtering_mode = options.filt_mode
    fps_factor = options.fps_factor

    corner_detector_radius = RES * options.corner_R
    #     corner_extractor_radius = RES * options.corner_r
    upper_variance_threshold = options.corner_up_thr
    lower_variance_threshold = options.corner_low_thr
    cornerness_threshold = options.cornerness
    box_margin = RES * options.box_margin
    quantile = options.quantile

    endpoint_detector_radius = RES * options.endpoint_R
    endpoint_threshold = options.endpoint_thr

    corner_connector_radius = RES * options.connect_R

    initial_split_threshold = RES * options.init_thr
    optimization_split_threshold = RES * options.opt_thr
    #     alpha_fid = options.alpha_fid
    #     alpha_fit = options.alpha_fit
    alpha_ang = options.alpha_ang

    draw_result = options.draw

    # ~~~~~~~~~~~~~~~~~~~~~~~~

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
        logger.debug('Loading data from {}'.format(options.corners_filename))
        dataset = Hdf5File(
            options.corners_filename,
            io=parametric_io.ParametricCornersIO,
            preload=PreloadTypes.LAZY,
            labels='*')
        corners = dataset[0]['corners']
        not_corners = dataset[0]['not_corners']
        corner_centers = dataset[0]['corner_centers']
        init_connections = dataset[0]['init_connections']
    except Exception as e:
        logger.error('Cannot load {}: {}; stopping'.format(options.corners_filename, str(e)))
        exit(1)
    logger.debug('Loaded {} corners'.format(len(points)))

    try:
        logger.debug('Segmenting individual curves')
        curves = tg.separate_points_to_subsets(
            points[not_corners],
            knn_radius=curve_connected_components_radius)
    except Exception as e:
        logger.error('Cannot segment curves: {}; stopping'.format(str(e)))
        exit(1)
    logger.debug('Segmented {} individual curves'.format(len(curves)))

    try:
        logger.debug('Initializing topological graph')
        corner_positions, corner_pairs = tg.initialize_topological_graph(
            points,
            distances,
            not_corners,
            curves,
            corners,
            corner_centers,
            init_connections,
            endpoint_detector_radius,
            endpoint_threshold,
            initial_split_threshold,
            corner_connector_radius)
    except Exception as e:
        logger.error('Initializing topological graph failed: {}; stopping'.format(str(e)))
        exit(1)
    logger.debug('Segmented {} individual curves'.format(len(curves)))

    topological_graph = {
        'vertices': corners,
        'edges': not_corners,
    }
    parametric_io.save_parametric_corners(
        topological_graph,
        options.output_filename)
    logger.debug('Saved detected corners to {}'.format(options.output_filename))


    filename = path_to_save.split('/')[-2]
    #     np.save('{path_to_save}/{filename}__corner_positions_unopt.npy'.format(path_to_save=path_to_save, filename=filename), corner_positions)
    #     np.save('{path_to_save}/{filename}__corner_pairs_unopt.npy'.format(path_to_save=path_to_save, filename=filename), corner_pairs)
    np.save('{path_to_save}/{filename}__sharp_points.npy'.format(path_to_save=path_to_save, filename=filename), points)
    np.save('{path_to_save}/{filename}__sharp_distances.npy'.format(path_to_save=path_to_save, filename=filename),
            distances)
    print('optimizing topological graph')
    #     corner_positions, corner_pairs = optimize_topological_graph(corner_positions, corner_pairs,
    #                                                                 points, distances,
    #                                                                 optimization_split_threshold, alpha_ang)

    corners, _, _, _, _ = opt.get_paths_and_corners(corner_pairs, corner_positions)
    print('saving result')
    filename = path_to_save.split('/')[-2]
    np.save('{path_to_save}/{filename}__corner_positions.npy'.format(path_to_save=path_to_save, filename=filename),
            corner_positions)
    np.save('{path_to_save}/{filename}__corner_pairs.npy'.format(path_to_save=path_to_save, filename=filename),
            corner_pairs)
    np.save('{path_to_save}/{filename}__corners.npy'.format(path_to_save=path_to_save, filename=filename), corners)

    if draw_result:
        print('drawing')
        DISPLAY_RES = RES * 1.5
        utils.draw(points, corner_positions, corner_pairs, path_to_save, filename, DISPLAY_RES)

    print('done!')


if __name__ == '__main__':
    options = parse_args()
    main(options)
