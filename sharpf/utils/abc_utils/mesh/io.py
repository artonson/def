from io import BytesIO
from typing import Tuple

import trimesh
import numpy as np


def trimesh_load(io: BytesIO, need_decode=True) -> Tuple[trimesh.base.Trimesh, np.ndarray, np.ndarray]:
    """Read the mesh: since trimesh messes the indices, this has to be done manually."""

    vertices, vertex_indices, vertex_normals, vertex_normal_indices = [], [], [], []

    file_contents = io.read()
    if need_decode:
        file_contents = file_contents.decode('utf-8')

    for line in file_contents.splitlines():
        values = line.strip().split()
        if not values:
            continue

        if values[0] == 'v':
            vertices.append(values[1:4])

        elif values[0] == 'f':
            vertex_indices.append([value.split('//')[0] for value in values[1:4]])
            vertex_normal_indices.append([value.split('//')[1] for value in values[1:4]])

        elif values[0] == 'vn':
            vertex_normals.append(values[1:4])

    vertices = np.array(vertices, dtype='float')
    vertex_indices = np.array(vertex_indices, dtype='int') - 1
    vertex_normals = np.array(vertex_normals, dtype='float')
    vertex_normal_indices = np.array(vertex_normal_indices, dtype='int') - 1

    mesh = trimesh.base.Trimesh(
        vertices=vertices,
        faces=vertex_indices,
        process=False,
        validate=False
    )  # create a mesh from the vertices
    return mesh, vertex_normals, vertex_normal_indices
