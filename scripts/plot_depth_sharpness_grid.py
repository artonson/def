#!/usr/bin/env python3

import argparse
import os
import sys

import matplotlib.pyplot as plt
import numpy as np

__dir__ = os.path.normpath(
    os.path.join(
        os.path.dirname(os.path.realpath(__file__)), '..'))
sys.path[1:1] = [__dir__]

from sharpf.data.datasets.sharpf_io import WholeDepthMapIO
from sharpf.utils.convertor_utils.convertors_io import AnnotatedViewIO
from sharpf.utils.abc_utils.hdf5.dataset import Hdf5File, PreloadTypes
from sharpf.utils.camera_utils.view import CameraView
from sharpf.utils.plotting import display_depth_sharpness


def sharpness_with_depth_background(
        sharpness_image,
        depth_image,
        depth_bg_value=1.0,
        sharpness_bg_value=0.0,
):
    assert sharpness_image.shape == depth_image.shape
    image = sharpness_image.copy()
    image[depth_image == depth_bg_value] = sharpness_bg_value
    return image


def load_data_by_label(filename, label, real_world=False):
    if real_world:
        dataset = Hdf5File(
            filename,
            io=AnnotatedViewIO,
            preload=PreloadTypes.LAZY,
            labels='*')
        pixel_views = [
            CameraView(
                depth=scan['points'],
                signal=scan['distances'],
                faces=scan['faces'].reshape((-1, 3)),
                extrinsics=scan['extrinsics'],
                intrinsics=scan['intrinsics'],
                state='pixels')
            for scan in dataset]
        dataset = [
            {'image': view.depth, 'distances': view.signal}
            for view in pixel_views]

    else:
        dataset = Hdf5File(
            options.input_filename,
            io=WholeDepthMapIO,
            preload=PreloadTypes.LAZY,
            labels=[label])

    list_data = [patch[label] for patch in dataset]
    return list_data


def min_bbox_for_all_images(images, bg_value):
    """Given a sequence of 2d images, find max bounding box
    that covers the set image != bg_value for each image."""
    top, left = images[0].shape
    bottom, right = 0, 0
    for image in images:
        y, x = np.where((image != bg_value).astype(float))
        top_i, bottom_i = np.min(y), np.max(y)
        top, bottom = min(top, top_i), max(bottom, bottom_i)
        left_i, right_i = np.min(x), np.max(x)
        left, right = min(left, left_i), max(right, right_i)
    return top, bottom, left, right


def cw_to_tblr(cx, cy, sx, sy):
    halfsize_y, halfsize_x = sy // 2, sx // 2
    top, bottom, left, right = cy - halfsize_y, cy + halfsize_y, \
                               cx - halfsize_x, cx + halfsize_x
    return top, bottom, left, right


def main(options):
    if options.verbose:
        print('Loading datasets...')
    if not options.depth_images and not options.sharpness_images:
        print('At least one of -di or -si must be set', file=sys.stderr)
        exit(1)
    depth_images = sharpness_images = None
    if options.depth_images:
        depth_images = load_data_by_label(options.input_filename, 'image', options.real_world)
    if options.sharpness_images:
        sharpness_images = load_data_by_label(options.input_filename, 'distances', options.real_world)

    res_x, res_y = options.resolution
    center_x, center_y = res_x // 2, res_y // 2
    if isinstance(options.crop_size, int):
        image_size_y, image_size_x = options.crop_size, options.crop_size
        top, bottom, left, right = cw_to_tblr(center_x, center_y, image_size_y, image_size_x)
    elif isinstance(options.crop_size, str) and options.crop_size == 'full':
        images = depth_images or sharpness_images
        image_size_y, image_size_x = images[0].shape
        top, bottom, left, right = cw_to_tblr(center_x, center_y, image_size_y, image_size_x)
    elif isinstance(options.crop_size, str) and options.crop_size == 'auto':
        images = depth_images or sharpness_images
        top, bottom, left, right = min_bbox_for_all_images(images, options.depth_bg_value)
    else:
        print('Cannot interpret -c/--crop_size option: "{}"'.format(options.crop_size), file=sys.stderr)
        exit(1)

    slices = slice(top, bottom), slice(left, right)

    depth_images_for_display = None
    if options.depth_images:
        depth_images_for_display = [image[slices] for image in depth_images]

    sharpness_images_for_display = None
    if options.sharpness_images:
        sharpness_images_for_display = [distances[slices] for distances in sharpness_images]

    if options.depth_images and options.sharpness_images and options.bg_from_depth:
        sharpness_images_for_display = [sharpness_with_depth_background(
            sharpness_image,
            depth_image,
            depth_bg_value=options.depth_bg_value,
            sharpness_bg_value=options.sharpness_bg_value)
        for sharpness_image, depth_image in
            zip(sharpness_images_for_display, depth_images_for_display)]

    f_x, f_y = map(float, options.figsize)
    if options.verbose:
        print('Plotting...')
    display_depth_sharpness(
        depth_images=depth_images_for_display,
        sharpness_images=sharpness_images_for_display,
        ncols=options.ncols,
        axes_size=(f_x, f_y),
        max_sharpness=options.max_distance_to_feature, 
        bgcolor=options.bgcolor,
        sharpness_hard_thr=options.sharpness_hard_thr,
        sharpness_hard_values=options.sharpness_hard_values,
        depth_bg_value=options.depth_bg_value,
        sharpness_bg_value=options.sharpness_bg_value)

    if options.verbose:
        print('Saving...')
    plt.savefig(options.output_filename)


def parse_args():
    parser = argparse.ArgumentParser()

    parser.add_argument('-i', '--input', dest='input_filename',
                        required=True, help='input files with prediction.')
    parser.add_argument('-o', '--output', dest='output_filename',
                        required=True, help='output .png filename.')
    parser.add_argument('--verbose', dest='verbose', action='store_true', default=False,
                        help='be verbose')
    parser.add_argument('-s', '--max_distance_to_feature', dest='max_distance_to_feature',
                        default=1.0, type=float, required=False, help='max distance to sharp feature to compute.')
    parser.add_argument('-c', '--crop_size', dest='crop_size',
                        default='auto', required=False,
                        help='make a crop of pixels of this size '
                             '[default: "auto" to crop an image of smallest size 2^K centered '
                             'on image center, "full" to show the entire image].')
    parser.add_argument('-di', '--depth_images', dest='depth_images',
                        default=False, action='store_true', required=False,
                        help='display depth images.')
    parser.add_argument('-si', '--sharpness_images', dest='sharpness_images',
                        default=False, action='store_true', required=False,
                        help='display sharpness images.')
    parser.add_argument('-r', '--resolution', dest='resolution', nargs=2, type=int,
                        required=True, help='resolution of the input image in pixels [width, height].')
    parser.add_argument('--ncols', dest='ncols',
                        default=1, type=int, required=False, help='number of cols.')
    parser.add_argument('-f', '--figsize', dest='figsize', nargs=2, default=(16, 16),
                        required=False, help='figure size in inches.')
    parser.add_argument('-w', '--real_world', dest='real_world',
                        default=False, action='store_true', required=False,
                        help='if set, this will read the input file as Views.')
    parser.add_argument('-bg', '--bgcolor', dest='bgcolor',
                        default='white', help='set background color for print.')
    parser.add_argument('-t', '--sharpness_hard_thr', dest='sharpness_hard_thr',
                        default=None, type=float, help='if set, forces to compute and paint hard labels.')
    parser.add_argument('-v', '--sharpness_hard_values', dest='sharpness_hard_values', nargs=2, default=None, type=float, 
                        help='if set, specifies min and max sharpness values for hard labels.')
    parser.add_argument('-bgd', '--bg_from_depth', dest='bg_from_depth',
                        default=False, action='store_true', required=False,
                        help='if set, this will set background in the sharpness image '
                             'to background in the depth image.')
    parser.add_argument('-dv', '--depth_bg_value', dest='depth_bg_value', default=0.0, type=float,
                        help='if set, specifies depth value to be treated as background (0.0 by default).')
    parser.add_argument('-sv', '--sharpness_bg_value', dest='sharpness_bg_value', default=0.0, type=float,
                        help='if set, specifies sharpness value to be treated as background (0.0 by default).')
    return parser.parse_args()


if __name__ == '__main__':
    options = parse_args()
    main(options)
