from io import BytesIO

import trimesh
import numpy as np


def trimesh_load(io: BytesIO) -> trimesh.base.Trimesh:
    """Read the mesh: since trimesh messes the indices, this has to be done manually."""

    vertices, faces = [], []

    for line in io.read().decode('utf-8').splitlines():
        values = line.strip().split()
        if not values:
            continue
        if values[0] == 'v':
            vertices.append(np.array(values[1:4], dtype='float'))
        elif values[0] == 'f':
            faces.append(np.array([values[1].split('//')[0], values[2].split('//')[0], values[3].split('//')[0]],
                                  dtype='int'))

    vertices = np.array(vertices)
    faces = np.array(faces) - 1

    mesh = trimesh.base.Trimesh(vertices=vertices, faces=faces, process=False)  # create a mesh from the vertices
    return mesh

