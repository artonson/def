from collections import defaultdict
from copy import deepcopy

import numpy as np

from sharpf.utils.abc_utils.mesh.indexing import reindex_array, reindex_zerobased


def get_adjacent_features_by_bfs_with_depth1(
        surface_idx,
        adjacent_sharp_features,
        adjacent_surfaces,
        always_check_adjacent_surfaces=False
):
    """If adjacent sharp curves exist, return one of them.
    If not, return ones adjacent to adjacent surfaces. """

    adjacent_sharp_indexes = deepcopy(adjacent_sharp_features[surface_idx])
    if always_check_adjacent_surfaces or len(adjacent_sharp_indexes) == 0:

        for adjacent_surface_idx in adjacent_surfaces[surface_idx]:
            adjacent_surface_adjacent_sharp_features = \
                {adjacent_surface_idx: adjacent_sharp_features[adjacent_surface_idx]}

            adjacent_surface_adjacent_sharp_indexes = \
                get_adjacent_features_by_bfs_with_depth1(
                    adjacent_surface_idx, adjacent_surface_adjacent_sharp_features,
                    defaultdict(list))

            adjacent_sharp_indexes.extend(adjacent_surface_adjacent_sharp_indexes)

    return adjacent_sharp_indexes


def get_surface_vert_indexes(mesh, surface, get_from_faces=False):
    if get_from_faces:
        face_indexes = surface['face_indices']
        return np.unique(mesh.faces[face_indexes].ravel())
    else:
        return np.array(surface['vert_indices'])


def build_surface_patch_graph(mesh, features):
    adjacent_sharp_features = defaultdict(list)
    adjacent_surfaces = defaultdict(list)

    for surface_idx, surface in enumerate(features['surfaces']):
        surface_vertex_indexes = get_surface_vert_indexes(mesh, surface, get_from_faces=True)

        for curve_idx, curve in enumerate(features['curves']):
            curve_vertex_indexes = np.array(curve['vert_indices'])

            if curve['sharp'] and np.any(np.isin(surface_vertex_indexes, curve_vertex_indexes)):
                adjacent_sharp_features[surface_idx].append(curve_idx)

        for other_surface_idx, other_surface in enumerate(features['surfaces']):
            if other_surface_idx != surface_idx:
                other_surface_vertex_indexes = get_surface_vert_indexes(mesh, other_surface, get_from_faces=True)

                if np.any(np.isin(surface_vertex_indexes, other_surface_vertex_indexes)):
                    adjacent_surfaces[surface_idx].append(other_surface_idx)

    return adjacent_sharp_features, adjacent_surfaces


def compute_features_nbhood(
        mesh,
        features,
        mesh_face_indexes,
        mesh_vertex_indexes=None,
        deduce_verts_from_faces=False,
):
    """Extracts curves for the neighbourhood."""

    if deduce_verts_from_faces:
        mesh_vertex_indexes = np.unique(mesh.faces[mesh_face_indexes].ravel())

    nbhood_curves = []
    for curve in features['curves']:
        curve_vertex_indexes = np.array(curve['vert_indices'])
        curve_vertex_indexes = curve_vertex_indexes[
            np.where(np.isin(curve_vertex_indexes, mesh_vertex_indexes))[0]]
        if len(curve_vertex_indexes) == 0:
            continue

        curve_vertex_indexes = reindex_array(curve_vertex_indexes, mesh_vertex_indexes)

        nbhood_curve = deepcopy(curve)
        nbhood_curve['vert_indices'] = curve_vertex_indexes
        nbhood_curves.append(nbhood_curve)

    nbhood_surfaces = []
    for idx, surface in enumerate(features['surfaces']):
        surface_face_indexes = np.array(surface['face_indices'])
        surface_face_indexes = surface_face_indexes[
            np.where(np.isin(surface_face_indexes, mesh_face_indexes))[0]]
        if len(surface_face_indexes) == 0:
            continue

        surface_faces = reindex_array(mesh.faces[surface_face_indexes], mesh_vertex_indexes)

        nbhood_surface = deepcopy(surface)
        nbhood_surface['face_indices'] = reindex_array(surface_face_indexes, mesh_face_indexes)
        nbhood_surface['vert_indices'] = np.unique(surface_faces)
        nbhood_surfaces.append(nbhood_surface)

    nbhood_features = {
        'curves': nbhood_curves,
        'surfaces': nbhood_surfaces,
    }
    return nbhood_features


def get_boundary_curves(mesh, surface, features):
    # represent surface patch as mesh
    surface_mesh = get_surface_as_mesh(mesh, surface, deduce_verts_from_faces=True)

    # extract surface mesh boundary (edges with only 1 adjacent face)
    boundary_vertex_indexes, _ = get_boundary_vertex_indexes(surface_mesh)

    # extract subset of curves belonging to surface patch
    surface_features = compute_features_nbhood(
        mesh, features, surface['face_indices'], deduce_verts_from_faces=True)

    # extract subset of curves belonging to surface mesh boundary
    boundary_curves = [
        curve for curve in surface_features['curves']
        if len(curve['vert_indices']) > 1
           and np.all(np.isin(curve['vert_indices'], boundary_vertex_indexes))]

    return boundary_curves


def get_intersecting_surfaces(mesh_face_indexes, features_surfaces):
    intersecting_surfaces = []
    for i, surface in enumerate(features_surfaces):
        face_indices = np.array(surface['face_indices'])
        if np.any(np.in1d(mesh_face_indexes, face_indices)):
            intersecting_surfaces.append(surface)
    return intersecting_surfaces


def get_surface_as_mesh(mesh, surface, deduce_verts_from_faces=False):
    # if deduce_verts_from_faces is True, extract vert_indices by pooling faces
    # rather than extracting these from surface directly
    vert_indices = get_surface_vert_indexes(mesh, surface, get_from_faces=deduce_verts_from_faces)
    face_indices = np.array(surface['face_indices'])
    return reindex_zerobased(mesh, vert_indices, face_indices)


def get_boundary_vertex_indexes(mesh):
    mesh_edge_indexes, mesh_edge_counts = np.unique(
        mesh.faces_unique_edges.flatten(), return_counts=True
    )
    boundary_edges = mesh.edges_unique[
        mesh_edge_indexes[
            np.where(mesh_edge_counts == 1)[0]
        ]
    ]
    boundary_vertex_indexes = np.unique(boundary_edges.flatten())
    return boundary_vertex_indexes, boundary_edges


def remove_boundary_features(mesh, features, how='none'):
    """Removes features indexed into vertex edges adjacent to 1 face only.
    :param how: 'all_verts': remove entire feature curve if all vertices are boundary
                'edges': remove vertices that belong to boundary edges only (not to other edges)
                'verts': remove vertices that are boundary
                'none': do nothing
    """
    if how == 'none':
        return features

    boundary_vertex_indexes, boundary_edges = get_boundary_vertex_indexes(mesh)

    non_boundary_curves = []
    for curve in features['curves']:
        non_boundary_curve = deepcopy(curve)

        if how == 'all_verts':
            if np.all([vert_index in boundary_vertex_indexes
                       for vert_index in curve['vert_indices']]):
                continue

        elif how == 'verts':
            non_boundary_vert_indices = np.array([
                vert_index for vert_index in curve['vert_indices']
                if vert_index not in boundary_vertex_indexes
            ])
            if len(non_boundary_vert_indices) == 0:
                continue
            non_boundary_curve['vert_indices'] = non_boundary_vert_indices

        elif how == 'edges':
            curve_edges = mesh.edges_unique[
                np.where(
                    np.all(np.isin(mesh.edges_unique, curve['vert_indices']), axis=1)
                )[0]
            ]
            non_boundary = (curve_edges[:, None] != boundary_edges).any(2).all(1)
            non_boundary_vert_indices = np.unique(curve_edges[non_boundary])
            if len(non_boundary_vert_indices) == 0:
                continue
            non_boundary_curve['vert_indices'] = non_boundary_vert_indices

        non_boundary_curves.append(non_boundary_curve)

    non_boundary_features = {
        'curves': non_boundary_curves,
        'surfaces': features.get('surfaces', [])
    }
    return non_boundary_features


def get_curves_extents(mesh, features):
    sharp_verts = [mesh.vertices[np.array(c['vert_indices'])]
                   for c in features['curves'] if c['sharp']]

    eps = 1e-8
    extents = np.array([
        np.max(verts, axis=0) - np.min(verts, axis=0) + eps
        for verts in sharp_verts])

    if len(extents) == 0:
        return []
    else:
        return extents.max(axis=1)


def get_curves_lengths_edges(mesh, features):
    sharp_verts = [mesh.vertices[np.array(c['vert_indices'])]
                   for c in features['curves'] if c['sharp']]
    return np.array([
        np.sum(np.linalg.norm(curve_vertices[:-1] - curve_vertices[1:]))
        for curve_vertices in sharp_verts
    ])


def submesh_from_hit_surfaces(mesh, features, mesh_face_indexes):
    # compute indexes of faces and vertices in the original mesh
    hit_surfaces_face_indexes = []
    for idx, surface in enumerate(features['surfaces']):
        surface_face_indexes = np.array(surface['face_indices'])
        if np.any(np.isin(surface_face_indexes, mesh_face_indexes, assume_unique=True)):
            hit_surfaces_face_indexes.extend(surface_face_indexes)
    mesh_face_indexes = np.unique(hit_surfaces_face_indexes)
    mesh_vertex_indexes = np.unique(mesh.faces[mesh_face_indexes])

    # assemble mesh fragment into a submesh
    nbhood = reindex_zerobased(mesh, mesh_vertex_indexes, mesh_face_indexes)

    return nbhood, mesh_vertex_indexes, mesh_face_indexes
