import trimesh
import numpy as np


def reindex_zerobased(mesh, vert_indices, face_indices):
    """Returns a submesh with reindexed faces."""
    selected_vertices = mesh.vertices[vert_indices]
    selected_faces = np.array(mesh.faces[face_indices])
    for reindex, index in zip(np.arange(len(selected_vertices)), vert_indices):
        selected_faces[np.where(selected_faces == index)] = reindex

    submesh = trimesh.base.Trimesh(
        vertices=selected_vertices,
        faces=selected_faces,
        process=False,
        validate=False)
    return submesh


def compute_relative_indexes(mesh, sub_mesh):
    """Returns vertex_indexes and face_indexes such that
    mesh.vertices[vertex_indices] ~ sub_mesh.vertices
    mesh.faces[face_indexes] ~ sub_mesh.faces."""

    # get vertex indexes in mesh by searching for identical vertices
    vertex_indexes = np.where(
        (mesh.vertices[:, np.newaxis] == sub_mesh.vertices[np.newaxis, ...]).all(axis=2).any(axis=1)
    )[0]
    face_indexes = mesh.vertex_faces[vertex_indexes]
    face_indexes = np.unique(face_indexes[face_indexes > -1])

    # fix face indexing by removing faces with vertices
    # in mesh but not in sub_mesh
    fix_mask = np.all(
        np.isin(mesh.vertices[mesh.faces[face_indexes]],
                sub_mesh.vertices[sub_mesh.faces]),
        axis=(2, 1)
    )

    return vertex_indexes, face_indexes[fix_mask]


def test_compute_relative_indexes():
    test_faces = np.array([[0, 1, 2], [2, 3, 4]])
    test_verts = np.array([[0, 0, 0], [0, 1, 0], [1, 0, 0], [2, 1, 0], [2, 0, 0], ])

    test_mesh = trimesh.base.Trimesh(
        vertices=test_verts,
        faces=test_faces,
        process=False,
        validate=False)

    sub_meshes = test_mesh.split(only_watertight=False)
    assert len(sub_meshes) == 2

    for sub_mesh in sub_meshes:
        vertex_indexes, face_indexes = compute_relative_indexes(test_mesh, sub_mesh)
        assert np.all(test_mesh.vertices[vertex_indexes] == sub_mesh.vertices, axis=1)

