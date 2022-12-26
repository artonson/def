import math
from math import sin, cos, tan, pi, radians

import numpy as np
import torch


def create_scale_matrix(sx, sy, sz):
    return torch.tensor([[sx, 0.0, 0.0, 0.0],
                         [0.0, sy, 0.0, 0.0],
                         [0.0, 0.0, sz, 0.0],
                         [0.0, 0.0, 0.0, 1.0]], dtype=torch.float32)


def create_rotation_matrix_x(rx):
    return torch.tensor([[1.0, 0.0, 0.0, 0.0],
                         [0.0, cos(rx), -sin(rx), 0.0],
                         [0.0, sin(rx), cos(rx), 0.0],
                         [0.0, 0.0, 0.0, 1.0]], dtype=torch.float32)


def create_rotation_matrix_y(ry):
    return torch.tensor([[cos(ry), 0.0, sin(ry), 0.0],
                         [0.0, 1.0, 0.0, 0.0],
                         [-sin(ry), 0.0, cos(ry), 0.0],
                         [0.0, 0.0, 0.0, 1.0]], dtype=torch.float32)


def create_rotation_matrix_z(rz):
    return torch.tensor([[cos(rz), -sin(rz), 0.0, 0.0],
                         [sin(rz), cos(rz), 0.0, 0.0],
                         [0.0, 0.0, 1.0, 0.0],
                         [0.0, 0.0, 0.0, 1.0]], dtype=torch.float32)


def create_translation_matrix(tx, ty, tz):
    return torch.tensor([[1.0, 0.0, 0.0, tx],
                         [0.0, 1.0, 0.0, ty],
                         [0.0, 0.0, 1.0, tz],
                         [0.0, 0.0, 0.0, 1.0]], dtype=torch.float32)


def create_projection_matrix(vertical_fov, aspect_ratio, near_z, far_z):
    f = 1.0 / (tan((vertical_fov / 2.0) * (pi / 180.0)))

    return torch.tensor([[f / aspect_ratio, 0.0, 0.0, 0.0],
                         [0.0, f, 0.0, 0.0],
                         [0.0, 0.0, (near_z + far_z) / (near_z - far_z), (2.0 * near_z * far_z) / (near_z - far_z)],
                         [0.0, 0.0, -1.0, 0.0]], dtype=torch.float32)


def create_transform(local_rotation, local_position, scale, rotate_around=None, angle=None, axis=None):
    angle_x, angle_y, angle_z = local_rotation
    rotation_x = create_rotation_matrix_x(angle_x)
    rotation_y = create_rotation_matrix_y(angle_y)
    rotation_z = create_rotation_matrix_z(angle_z)

    shift_x, shift_y, shift_z = local_position
    translation = create_translation_matrix(-shift_x, -shift_y, -shift_z)
    rotation = torch.mm(rotation_x, rotation_y)
    rotation = torch.mm(rotation, rotation_z)
    transform = torch.mm(rotation, translation)

    ra_translation = create_translation_matrix(rotate_around[0],
                                               rotate_around[1],
                                               rotate_around[2])
    ra_translation_inverse = create_translation_matrix(-rotate_around[0],
                                                       -rotate_around[1],
                                                       -rotate_around[2])
    ra_rotation = create_rotation_matrix_y(radians(-angle))
    ra_transform = torch.mm(transform, ra_translation)
    ra_transform = torch.mm(ra_transform, ra_rotation)
    ra_transform = torch.mm(ra_transform, ra_translation_inverse)

    return ra_transform.t()


def random_3d_rotation_matrix():
    angle_x, angle_y, angle_z = (torch.rand(3) * 2 * math.pi).numpy()

    rotation_x = create_rotation_matrix_x(angle_x)
    rotation_y = create_rotation_matrix_y(angle_y)
    rotation_z = create_rotation_matrix_z(angle_z)

    rotation = torch.mm(
        torch.mm(rotation_x, rotation_y),
        rotation_z)

    return rotation


def random_scale_matrix(min_scale, max_scale):
    scale_value = min_scale + torch.rand(1).item() * (max_scale - min_scale)
    scale = create_scale_matrix(scale_value, scale_value, scale_value)
    return scale


def image_to_points(image, resolution_3d):
    """
    Depth image to point cloud transformation
        input: image: depth image, shape = (image_height, image_width)
               resolution_3d: sampling resolution, float

        output: points: point cloud obtained from the depth image, shape = (image_height * image_width, 3)
    """
    image_height, image_width = image.shape[0], image.shape[1]
    screen_aspect_ratio = image_width / image_height
    rays_screen_coords = np.mgrid[0:image_height, 0:image_width].reshape(2, image_height * image_width).T

    rays_origins = (rays_screen_coords / np.array([[image_height, image_width]]))  # [h, w, 2], in [0, 1]
    factor = image_height / 2 * resolution_3d
    rays_origins[:, 0] = (-2 * rays_origins[:, 0] + 1) * factor  # to [-1, 1] + aspect transform
    rays_origins[:, 1] = (-2 * rays_origins[:, 1] + 1) * factor * screen_aspect_ratio
    rays_origins = np.concatenate([
        rays_origins,
        np.zeros_like(rays_origins[:, [0]])
    ], axis=1)
    points = np.zeros((image_height * image_width, 3))
    points[:, 0] = rays_origins[:, 0]
    points[:, 1] = rays_origins[:, 1]
    points[:, 2] = image.ravel()
    return points
