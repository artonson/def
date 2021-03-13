from collections import defaultdict
from dataclasses import dataclass
from glob import glob
from io import StringIO
import os
from typing import Tuple

import numpy as np
import trimesh.transformations as tt
import trimesh
from tqdm import tqdm

from sharpf.utils.camera_utils import matrix
from sharpf.utils.camera_utils.camera_pose import CameraPose


@dataclass
class RangeVisionNames:
    points: str = 'points'
    faces: str = 'faces'
    vertex_matrix: str = 'vertex_matrix'
    rxyz_euler_angles: str = 'alf,om,kap'
    translation = 'X0'
    focal_length: str = 'f'
    focal_length_nom: str = 'fnom'
    pixel_size_xy: str = 'm'
    center_xy: str = 'b'
    alignment: str = 'alignment'
    correction: str = 'adp'

    sec_image_geometry: str = 'CImgGeom'
    sec_projective_transform : str = 'CProjCentr'
    sec_camera_params: str = 'CParamDig1i'



def read_calibration_params(filename):
    PARAM_NAMES = [
        RangeVisionNames.sec_image_geometry,

        RangeVisionNames.sec_projective_transform,
        RangeVisionNames.focal_length,
        RangeVisionNames.focal_length_nom,
        RangeVisionNames.rxyz_euler_angles,
        RangeVisionNames.translation,

        RangeVisionNames.sec_camera_params,
        RangeVisionNames.pixel_size_xy,
        RangeVisionNames.center_xy,
        RangeVisionNames.correction]

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


def get_camera_extrinsic(angles, translation):
    return CameraPose.from_camera_axes(
        R=tt.euler_matrix(*angles, axes='sxyz')[:3, :3],
        t=translation
    )


def get_right_camera_extrinsics(angles, translation):
    R = tt.euler_matrix(*angles, axes='rxyz')[:3, :3]
    R = R[[1, 0, 2]].T[[1, 0, 2]]
    flip_z = [[1, 0, 0], [0, 1, 0], [0, 0, -1]]
    flip_y = [[1, 0, 0], [0, -1, 0], [0, 0, 1]]
    R = np.dot(
        flip_z,
        np.dot(flip_y, R))

    g = np.zeros((4, 4))
    g[3, 3] = 1
    g[:3, :3] = R
    g[:3, 3] = np.dot(-R, translation)

    return CameraPose(np.linalg.inv(g))



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
        Kf = matrix.get_camera_intrinsic_f(f)
        Ks = matrix.get_camera_intrinsic_s(s_x, s_y, o_x, o_y)

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

