from collections import defaultdict
from glob import glob
from io import StringIO
import os
from typing import Tuple

import numpy as np
import trimesh.transformations as tt
import trimesh
from tqdm import tqdm

from sharpf.utils.camera_utils.camera_pose import CameraPose


def read_calibration_params(filename):
    PARAM_NAMES = ['CImgGeom',
                   'CProjCentr',
                   'f',
                   'fnom',
                   'alf,om,kap',
                   'X0',
                   'CParamDig1i',
                   'm',
                   'b',
                   'adp']
    params_by_name = defaultdict(list)
    with open(filename) as f:
        for line in f:
            s = line.strip()
            if s in PARAM_NAMES:
                key_to_set = s
            else:
                if None is not key_to_set:
                    params_by_name.setdefault(key_to_set, list())
                    params_by_name[key_to_set].append(s)

    params_by_name = {
        name: np.loadtxt(
            StringIO(' '.join(values)))
        for name, values in params_by_name.items()
    }

    params_by_name = {
        name: values if len(values.shape) > 0 else float(values)
        for name, values in params_by_name.items()
    }
    return params_by_name


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


def get_camera_extrinsic(angles, translation):
    return CameraPose.from_camera_axes(
        R=tt.euler_matrix(*angles, axes='sxyz')[:3, :3],
        t=translation
    )


def assign_by_min_depth(
        image_size,
        image_offset,
        x_pixel: np.ndarray,
        depth: np.ndarray,
        func=np.min
):
    """Given a [n, 3] array of rounded pixel coordinates (indexes)
    and corresponding [n,] array of depth values,
    return a 2d image with pixel values assigned to smallest (closest to camera) depth values.

    Normally we do:
    >>> image = np.zeros((image_size[1], image_size[0]))
    >>> image[
    >>>     x_pixel[:, 1] - image_offset[1],
    >>>     x_pixel[:, 0] - image_offset[0]] = depth

    However, if multiple values in x_pixel index into the same (x, y) pixel in image,
    i.e. for i != j we have x_pixel[i] == x_pixel[j],
    we end up assigning arbitrary depth value in image.

    Instead we do (conceptually):
    >>> image[x_pixel] = func(depth[i] for i where x_pixel[i] == c)
    """

    # sort XY array by row/col index
    sort_idx_1 = x_pixel[:, 1].argsort()
    x_pixel = x_pixel[sort_idx_1]
    sort_idx_0 = x_pixel[:, 0].argsort(kind='mergesort')
    x_pixel = x_pixel[sort_idx_0]

    idx_sort = sort_idx_1[sort_idx_0]

    # returns the unique values, the index of the first occurrence of a value, and the count for each element
    vals, idx_start, count = np.unique(x_pixel, axis=0, return_counts=True, return_index=True)

    # splits the indices into separate arrays
    res = np.split(idx_sort, idx_start[1:])

    image = np.zeros((image_size[1], image_size[0]))
    for pixel_xy, idx_arr in zip(vals, res):
        image[
            pixel_xy[1] - image_offset[1],
            pixel_xy[0] - image_offset[0]] = func(depth[idx_arr])

    return x_pixel, image


def project_to_camera(
        points,
        pose: CameraPose,
        Kf: np.array,
        Ks: np.array,
        image_size: Tuple[int, int],
):
    X_world = points

    # transform to camera frame
    X_camera = pose.world_to_camera(X_world)

    # project using perspective projection
    x_image = np.dot(Kf, X_camera.T).T

    # separate XY (image coordinates) and Z
    x_image_saved = x_image.copy()
    depth = x_image[:, 2].copy()
    x_image[:, 2] = 1.

    # obtain XY in pixel coordinates
    x_pixel = np.dot(Ks, x_image.T).T
    x_pixel = np.round(x_pixel).astype(np.int32)

    # construct image
    image_offset = np.min(x_pixel, axis=0)[:2]
    image_size = np.max(x_pixel, axis=0) - np.min(x_pixel, axis=0) + 1
    print('Overwriting image_size with ', (image_size[1], image_size[0]))

    x_pixel, image = assign_by_min_depth(
        image_size, image_offset, x_pixel, depth, func=np.max)

#     image = np.zeros((image_size[1], image_size[0]))
#     image[
#         x_pixel[:, 1] - image_offset[1],
#         x_pixel[:, 0] - image_offset[0]] = depth

    # image = np.zeros((image_size[1] + 1, image_size[0] + 1))
    # image[x_pixel[:, 1], x_pixel[:, 0]] = depth

    return X_camera, x_pixel, x_image_saved, image, (image_offset[0], image_offset[1])


def pad_to_size(image, target_size=(512, 512), pad_value=1.0):
    image = np.atleast_3d(image)
    h, w, c = image.shape

    # create new image of desired size and color (blue) for padding
    ww, hh = target_size
    padded = np.full((hh, ww, c), pad_value, dtype=np.float)

    # compute center offset
    xx = (ww - w) // 2
    yy = (hh - h) // 2

    # copy img image into center of result image
    padded[yy:yy + h, xx:xx + w] = image

    return padded.squeeze()


def reproject_image_to_points(
        image,
        pose: CameraPose,
        Kf: np.array,
        Ks: np.array,
        image_offset: Tuple[int, int],
):
    image_size = (
        image.shape[1] + image_offset[0],
        image.shape[0] + image_offset[1])

    #     x_mm = np.linspace(0, 2048 / s_x, 2048)
    #     y_mm = np.linspace(0, 1536 / s_y, 1536)
    i = np.arange(image_offset[0], image_offset[0] + image.shape[1])
    j = np.arange(image_offset[1], image_offset[1] + image.shape[0])
    i, j = np.meshgrid(i, j)

    x_pixel = np.stack((i, j, np.ones_like(i)))  # .reshape((3, -1)).T
    x_pixel = x_pixel[:, image != 0].T

    x_image = np.dot(
        np.linalg.inv(Ks),
        x_pixel.T).T
    x_image[:, 2] = image[image != 0].ravel()

    X_camera = np.dot(
        np.linalg.inv(Kf),
        x_image.T).T

    X = pose.camera_to_world(X_camera)

    return x_pixel, x_image, X_camera, X


def rangevision_project_to_hdf5(
        input_dir: str,
        camera='a',
        #     target_size=(512, 512)
):
    scans = sorted(glob(os.path.join(input_dir, 'scan_res_*')))
    ply_files = sorted(glob(os.path.join(input_dir, '*.ply')))

    images, image_offsets = [], []
    for scan_dir, ply_file in tqdm(zip(scans, ply_files)):
        print(scan_dir)

        scan = trimesh.load(ply_file)

        params_by_name = read_calibration_params(
            os.path.join(scan_dir, 'Raw/impar{}01.txt'.format(camera)))

        angles = params_by_name['alf,om,kap']
        X0 = params_by_name['X0']
        f = params_by_name['f']
        s_x, s_y = params_by_name['m'] * 1e3
        o_x, o_y = params_by_name['b']

        pose = get_camera_extrinsic(angles, X0)
        Kf = get_camera_intrinsic_f(f)
        Ks = get_camera_intrinsic_s(s_x, s_y, o_x, o_y)

        X_camera, x_pixel, x_image, image, image_offset = project_to_camera(
            scan.vertices,
            pose,
            Kf,
            Ks,
            image_size=(2048, 1536))
        #         padded = pad_to_size(image, target_size=target_size, pad_value=0.)

        images.append(image)
        image_offsets.append(image_offset)

    return images, image_offsets

