import numpy as np


def get_camera_intrinsic_f(f):
    return np.array([
        [f, 0, 0],
        [0, f, 0],
        [0, 0, 1],
    ])


def get_camera_intrinsic_s(s_x, s_y, o_x, o_y):
    return np.array([
        [1. / s_x, 0, o_x],
        [0, 1. / s_y, o_y],
        [0, 0, 1]
    ])
