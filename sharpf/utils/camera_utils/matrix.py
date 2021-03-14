import numpy as np


def get_camera_intrinsic_f(f):
    return np.array([
        [f, 0, 0],
        [0, f, 0],
        [0, 0, 1],
    ])


def get_camera_intrinsic_s(s_x, s_y, o_x, o_y):
    """Construct matrix to transform from image UV coordinates
    to (real-valued) pixel IJ coordinates.

    :param s_x: pixel width in metric units (e.g. in mm)
    :param s_y: pixel height in metric units (e.g. in mm)
    :param o_x: image resolution in pixels along X axis (in pixels)
    :param o_y: image resolution in pixels along Y axis (in pixels)
    :return: matrix to transform between image and pixel frames
    """
    return np.array([
        [1. / s_x, 0, o_x / 2.],
        [0, 1. / s_y, o_y / 2.],
        [0, 0, 1]
    ])
