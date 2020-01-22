"""Various transformation matrices."""
# Copyright © 2014 Mikko Ronkainen <firstname@mikkoronkainen.com>
# License: MIT, see the LICENSE file.

import torch
from math import sin, cos, tan, pi, radians

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

def get_view(local_rotation, local_position, rotate_around=None, angle=None, axis=None):
    rotation_x = create_rotation_matrix_x(radians(-local_rotation[0]))
    rotation_y = create_rotation_matrix_y(radians(-local_rotation[1]))
    rotation_z = create_rotation_matrix_z(radians(-local_rotation[2]))
    
    translation = create_translation_matrix(-local_position[0], -local_position[1], -local_position[2])
    rotation = torch.mm(rotation_x, rotation_y)
    rotation = torch.mm(rotation, rotation_z)
    transform = torch.mm(rotation, translation)
    
    ra_translation = create_translation_matrix(rotate_around[0], \
                                             rotate_around[1], \
                                             rotate_around[2])
    ra_translation_inverse = create_translation_matrix(-rotate_around[0], \
                                             -rotate_around[1], \
                                             -rotate_around[2])
    ra_rotation = create_rotation_matrix_y(radians(-angle))
    ra_transform = torch.mm(transform, ra_translation)
    ra_transform = torch.mm(ra_transform, ra_rotation)
    ra_transform = torch.mm(ra_transform, ra_translation_inverse)
    
    return ra_transform.t()
