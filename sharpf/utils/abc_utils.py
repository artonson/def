from collections import defaultdict
from copy import deepcopy

import numpy as np


def get_adjacent_features_by_bfs_with_depth1(surface_idx, adjacent_sharp_features, adjacent_surfaces):
    """If adjacent sharp curves exist, return one of them.
    If not, return ones adjacent to adjacent surfaces. """

    adjacent_sharp_indexes = deepcopy(adjacent_sharp_features[surface_idx])

    # if not adjacent_sharp_indexes:
    #     adjacent_sharp_indexes = []
    #
    for adjacent_surface_idx in adjacent_surfaces[surface_idx]:
        adjacent_surface_adjacent_sharp_features = \
            {adjacent_surface_idx: adjacent_sharp_features[adjacent_surface_idx]}

        adjacent_surface_adjacent_sharp_indexes = \
            get_adjacent_features_by_bfs_with_depth1(
                adjacent_surface_idx, adjacent_surface_adjacent_sharp_features,
                defaultdict(list))

        adjacent_sharp_indexes.extend(adjacent_surface_adjacent_sharp_indexes)

    return adjacent_sharp_indexes


def build_surface_patch_graph(features):
    adjacent_sharp_features = defaultdict(list)
    adjacent_surfaces = defaultdict(list)

    for surface_idx, surface in enumerate(features['surfaces']):
        surface_vertex_indexes = np.array(surface['vert_indices'])

        for curve_idx, curve in enumerate(features['curves']):
            curve_vertex_indexes = np.array(curve['vert_indices'])

            if curve['sharp'] and np.any(np.isin(surface_vertex_indexes, curve_vertex_indexes)):
                adjacent_sharp_features[surface_idx].append(curve_idx)

        for other_surface_idx, other_surface in enumerate(features['surfaces']):
            if other_surface_idx != surface_idx:
                other_surface_vertex_indexes = np.array(other_surface['vert_indices'])

                if np.any(np.isin(surface_vertex_indexes, other_surface_vertex_indexes)):
                    adjacent_surfaces[surface_idx].append(other_surface_idx)

    return adjacent_sharp_features, adjacent_surfaces


def compute_features_nbhood(mesh, features, mesh_vertex_indexes, mesh_face_indexes):
    """Extracts curves for the neighbourhood."""
    nbhood_curves = []
    for curve in features['curves']:
        curve_vertex_indexes = np.array(curve['vert_indices'])
        nbhood_vertex_indexes = curve_vertex_indexes[
            np.where(np.isin(curve_vertex_indexes, mesh_vertex_indexes))[0]]
        if len(nbhood_vertex_indexes) == 0:
            continue

        for index, reindex in zip(np.sort(mesh_vertex_indexes), np.arange(len(mesh_vertex_indexes))):
            nbhood_vertex_indexes[np.where(nbhood_vertex_indexes == index)] = reindex

        nbhood_curve = deepcopy(curve)
        nbhood_curve['vert_indices'] = nbhood_vertex_indexes
        nbhood_curves.append(nbhood_curve)

    nbhood_surfaces = []
    for idx, surface in enumerate(features['surfaces']):
        surface_face_indexes = np.array(surface['face_indices'])
        nbhood_face_indexes = surface_face_indexes[
            np.where(np.isin(surface_face_indexes, mesh_face_indexes, assume_unique=True))[0]
        ]
        if len(nbhood_face_indexes) == 0:
            continue

        surface_faces = np.array(mesh.faces[nbhood_face_indexes])
        for index, reindex in zip(np.sort(mesh_vertex_indexes), np.arange(len(mesh_vertex_indexes))):
            surface_faces[np.where(surface_faces == index)] = reindex

        #         surface_vertex_indexes = np.array(surface['vert_indices'])
        #         nbhood_vertex_indexes = surface_vertex_indexes[
        #             np.where(np.isin(surface_vertex_indexes, mesh_vertex_indexes))[0]]

        #         for index, reindex in zip(np.sort(mesh_vertex_indexes), np.arange(len(mesh_vertex_indexes))):
        #             nbhood_vertex_indexes[np.where(nbhood_vertex_indexes == index)] = reindex

        nbhood_surface = deepcopy(surface)
        #         nbhood_surface['vert_indices'] = nbhood_vertex_indexes
        nbhood_surface['face_indices'] = surface_faces
        nbhood_surface['vert_indices'] = np.unique(surface_faces)

        nbhood_surfaces.append(nbhood_surface)

    nbhood_features = {
        'curves': nbhood_curves,
        'surfaces': nbhood_surfaces,
    }
    return nbhood_features


def remove_boundary_features(mesh, features, how='none'):
    """Removes features indexed into vertex edges adjacent to 1 face only.
    :param how: 'all_verts': remove entire feature curve if all vertices are boundary
                'edges': remove vertices that belong to boundary edges only (not to other edges)
                'verts': remove vertices that are boundary
                'none': do nothing
    """
    if how == 'none':
        return features

    mesh_edge_indexes, mesh_edge_counts = np.unique(
        mesh.faces_unique_edges.flatten(), return_counts=True)

    boundary_edges = mesh.edges_unique[mesh_edge_indexes[np.where(mesh_edge_counts == 1)[0]]]
    boundary_vertex_indexes = np.unique(boundary_edges.flatten())

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

    extents = extents.max(axis=1)

    return extents


def get_curves_lengths_edges(mesh, features):
    sharp_verts = [mesh.vertices[np.array(c['vert_indices'])]
                   for c in features['curves'] if c['sharp']]
    return np.array([
        np.sum(np.linalg.norm(curve_vertices[:-1] - curve_vertices[1:]))
        for curve_vertices in sharp_verts
    ])
