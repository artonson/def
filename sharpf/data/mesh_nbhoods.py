from abc import ABC, abstractmethod

import numpy as np
from scipy.spatial import cKDTree
import point_cloud_utils as pcu

from sharpf.data import DataGenerationException
from sharpf.utils.mesh_utils.indexing import reindex_zerobased, compute_relative_indexes


class NeighbourhoodFunc(ABC):
    """Implements obtaining neighbourhoods from meshes.
    Given a mesh and a vertex, extracts its sub-mesh, i.e.
    a subset of vertices and edges, that correspond to
    a neighbourhood of some type."""

    @abstractmethod
    def get_nbhood(self):
        """Extracts a mesh neighbourhood.

        :returns: neighbourhood: mesh whose faces are within a a
        :rtype: MeshType (must be present attributes `vertices`, `faces`, and `edges`)
        """
        pass

    @classmethod
    def from_config(cls, config):
        pass

    @abstractmethod
    def index(self, mesh):
        """Indexes a mesh for further processing.

        :param mesh: an input mesh
        :type mesh: MeshType (must be present attributes `vertices`, `faces`, and `edges`)
        """
        pass


class EuclideanSphere(NeighbourhoodFunc):
    """Select all faces with at least one vertex within
    a specified radius from a specified point."""
    def __init__(self, centroid, radius_base, n_vertices, geodesic_patches, radius_scale_mode):
        self.centroid = centroid
        self.radius_base = radius_base
        self.n_vertices = n_vertices
        self.geodesic_patches = geodesic_patches
        self.radius_scale_mode = radius_scale_mode
        self.radius_scale = 1.
        self.mesh = None
        self.tree = None

    def index(self, mesh):
        self.mesh = mesh
        self.tree = cKDTree(mesh.vertices, leafsize=1000)
        if self.radius_scale_mode == 'from_edge_len':
            self.radius_scale = self.mesh.edges_unique_length.mean() if self.radius_scale_mode else 1
        elif self.radius_scale_mode == 'no_scale':
            pass

    def get_nbhood(self):
        # select vertices falling within euclidean sphere
        try:
            _, mesh_vertex_indexes = self.tree.query(
                self.centroid, k=self.n_vertices, distance_upper_bound=self.radius_base * self.radius_scale)
        except RuntimeError:
            raise DataGenerationException('Querying in very large meshes failed')

        mesh_vertex_indexes = np.array(mesh_vertex_indexes)
        # get all faces that share vertices with selected vertices
        mesh_vertex_indexes = mesh_vertex_indexes[mesh_vertex_indexes < len(self.mesh.vertices)]
        if len(mesh_vertex_indexes) == 0:
            raise DataGenerationException('No mesh vertices captured within tolerable distance')
        mesh_face_indexes = self.mesh.vertex_faces[mesh_vertex_indexes]
        mesh_face_indexes = np.unique(mesh_face_indexes[mesh_face_indexes > -1])
        # add all vertices that sit on adjacent faces
        mesh_vertex_indexes = np.unique(self.mesh.faces[mesh_face_indexes])
        # close selected faces wrt to selected vertices
        # (add faces where all selected vertices have been added)
        mesh_face_indexes, = np.where(np.all(
            np.any([self.mesh.faces == index for index in mesh_vertex_indexes], axis=0),
            axis=1))

        # copy vertices, reindex faces
        nb = reindex_zerobased(self.mesh, mesh_vertex_indexes, mesh_face_indexes)

        # get the connected component with maximal area
        if self.geodesic_patches:
            sub_meshes = nb.split(only_watertight=False)
            if len(sub_meshes) > 1:
                areas = np.array([sub_mesh.area for sub_mesh in sub_meshes])
                sub_mesh = sub_meshes[areas.argmax()]

                # get indices of verts and faces
                nb_vertex_indexes, nb_face_indexes = compute_relative_indexes(nb, sub_mesh)

                # get down to sub_mesh, copy vertices, reindex faces
                nb = reindex_zerobased(nb, nb_vertex_indexes, nb_face_indexes)

                # do nested indexing
                mesh_vertex_indexes, mesh_face_indexes = mesh_vertex_indexes[nb_vertex_indexes], \
                                                         mesh_face_indexes[nb_face_indexes]

        return nb, mesh_vertex_indexes, mesh_face_indexes, self.radius_scale

    @classmethod
    def from_config(cls, config):
        return cls(config['centroid'], config['radius_base'], config['n_vertices'],
                   config['geodesic_patches'], config['radius_scale_mode'])


class RandomEuclideanSphere(EuclideanSphere):
    def __init__(self, centroid, radius_base, n_vertices, geodesic_patches, radius_scale_mode, radius_delta,
                 max_patches_per_mesh, centroid_mode):
        super().__init__(centroid, radius_base, n_vertices, geodesic_patches, radius_scale_mode)
        self.radius_delta = radius_delta
        self.max_patches_per_mesh = max_patches_per_mesh
        self.n_patches_per_mesh = 0
        self.current_patch_idx = 0
        self.centroids_cache = []
        self.centroid_mode = centroid_mode

    def index(self, mesh):
        super(RandomEuclideanSphere, self).index(mesh)

        if self.centroid_mode == 'poisson_disk':
            mesh_vertices = np.array(mesh.vertices, order='C')
            mesh_normals = np.array(mesh.vertex_normals, order='C')
            mesh_faces = np.array(mesh.faces)

            self.centroids_cache, _ = pcu.sample_mesh_poisson_disk(
                mesh_vertices, mesh_faces, mesh_normals,
                -1, radius=2. * self.radius_base, use_geodesic_distance=True)
            self.centroids_cache = np.atleast_2d(self.centroids_cache)

            if len(self.centroids_cache) > self.max_patches_per_mesh:
                centroid_indexes = np.random.choice(
                    len(self.centroids_cache), size=self.max_patches_per_mesh, replace=False)
                self.centroids_cache = self.centroids_cache[centroid_indexes, :]

            self.n_patches_per_mesh = len(self.centroids_cache)

        elif self.centroid_mode == 'random_vertex':
            centroid_indexes = np.random.choice(
                len(self.mesh.vertices), size=self.max_patches_per_mesh, replace=False)
            self.centroids_cache = self.mesh.vertices[centroid_indexes]
            self.n_patches_per_mesh = len(self.centroids_cache)

        else:
            raise ValueError('Unknown patches specification: "{}"'.format(self.centroid))

    def get_nbhood(self):
        if self.current_patch_idx >= self.n_patches_per_mesh:
            raise StopIteration
        self.centroid = self.centroids_cache[self.current_patch_idx]
        self.radius_base = np.random.uniform(
            self.radius_base - self.radius_delta,
            self.radius_base + self.radius_delta)
        self.current_patch_idx += 1
        return super(RandomEuclideanSphere, self).get_nbhood()

    @classmethod
    def from_config(cls, config):
        return cls(config['centroid'], config['radius_base'],
                   config['n_vertices'], config['geodesic_patches'],
                   config['radius_scale_mode'], config['radius_delta'],
                   config['max_patches_per_mesh'], config['centroid_mode'])

#
# # the function for patch generator: breadth-first search
#
# def find_and_add(sets, desired_number_of_points, adjacency_graph):
#     counter = len(sets)  # counter for number of vertices added to the patch;
#     # sets is the list of vertices included to the patch
#     for verts in sets:
#         for vs in adjacency_graph.neighbors(verts):
#             if vs not in sets:
#                 sets.append(vs)
#                 counter += 1
#         #                 print(counter)
#         if counter >= desired_number_of_points:
#             break  # stop when the patch has more than 1024 vertices
#
#
#
#
# def geodesic_meshvertex_patches_from_item(item, n_points=1024):
#     """Sample patches from mesh, using vertices as points
#     in the point cloud,
#
#     :param item: MergedABCItem
#     :param n_points: number of points to output per patch (guaranteed)
#     :return:
#     """
#     yml = yaml.load(item.feat)
#     mesh = trimesh_load(item.obj)
#
#     sharp_idx = []
#     short_idx = []
#     for i in yml['curves']:
#         if len(i['vert_indices']) < 5:  # this is for filtering based on short curves:
#             # append all the vertices which are in the curves with less than 5 vertices
#             short_idx.append(np.array(i['vert_indices']) - 1)  # you need to substract 1 from vertex index,
#             # since it starts with 1
#         if i.get('sharp') is True:
#             sharp_idx.append(np.array(i['vert_indices']) - 1)  # append all the vertices which are marked as sharp
#     if len(sharp_idx) > 0:
#         sharp_idx = np.unique(np.concatenate(sharp_idx))
#     if len(short_idx) > 0:
#         short_idx = np.unique(np.concatenate(short_idx))
#
#     surfaces = []
#     for i in yml['surfaces']:
#         if 'vert_indices' in i.keys():
#             surfaces.append(np.array(i['face_indices']) - 1)
#
#
#     sharp_indicator = np.zeros((len(mesh.vertices),))
#     sharp_indicator[sharp_idx] = 1
#
#     # and faces read previously
#     adjacency_graph = mesh.vertex_adjacency_graph
#
#     # select starting vertices to grow patches from,
#     # while iterating over them use BFS to generate patches
#     # TODO: why not sample densely / specify no. of patches to sample?
#     for j in np.linspace(0, len(mesh.vertices), 7, dtype='int')[:-1]:
#         set_of_verts = [j]
#         find_and_add(sets=set_of_verts, desired_number_of_points=n_points,
#                      adjacency_graph=adjacency_graph)  # BFS function
#
#         # TODO: what does this code do?
#         a = sharp_indicator[np.array(set_of_verts)[-100:]]
#         b = np.isin(np.array(set_of_verts)[-100:], np.array(set_of_verts)[-100:] - 1)
#         if (a[b].sum() > 3):
#             #                 print('here! border!',j)
#             continue
#
#         set_of_verts = np.unique(np.array(set_of_verts))  # the resulting list of vertices in the patch
#         # TODO why discard short lines?
#         if np.isin(set_of_verts, short_idx).any():  # discard a patch if there are short lines
#             continue
#
#         patch_vertices = mesh.vertices[set_of_verts]
#         patch_sharp = sharp_indicator[set_of_verts]
#         patch_normals = mesh.vertex_normals[set_of_verts]
#
#         if patch_sharp.sum() != 0:
#             sharp_rate.append(1)
#         else:
#             sharp_rate.append(0)
#
#         surfaces_numbers = []
#         if patch_vertices.shape[0] >= n_points:
#             # select those vertices, which are not sharp in order to use them for counting surfaces (sharp vertices
#             # are counted twice, since they are on the border between two surfaces, hence they are discarded)
#             appropriate_verts = set_of_verts[:n_points][
#                 patch_sharp[:n_points].astype(int) == 0]
#
#             for surf_idx, surf_faces in enumerate(surfaces):
#                 surf_verts = np.unique(mesh.faces[surf_faces].ravel())
#
#                 if len(np.where(np.isin(appropriate_verts, surf_verts))[0]) > 0:
#                     surface_ratio = sharp_indicator[np.unique(np.array(surf_verts))].sum() / \
#                                     len(np.unique(np.array(surf_verts)))
#
#                     if surface_ratio > 0.6:
#                         break
#
#                     surfaces_numbers.append(surf_idx)  # write indices of surfaces which are present in the patch
#
#             if surface_ratio > 0.6:
#                 continue
#
#             surface_rate.append(np.unique(np.array(surfaces_numbers)))
#             patch_vertices = patch_vertices[:n_points]
#             points.append(patch_vertices)
#             patch_vertices_normalized = patch_vertices - patch_vertices.mean(axis=0)
#             patch_vertices_normalized = patch_vertices_normalized / np.linalg.norm(patch_vertices_normalized,
#                                                                                    ord=2, axis=1).max()
#             points_normalized.append(patch_vertices_normalized)
#             patch_normals = patch_normals[:n_points]
#             normals.append(patch_normals)
#             labels.append(patch_sharp[:n_points])
#
#
#     return points, labels, normals, sharp_rate
#     points = []  # for storing initial coordinates of points
#     points_normalized = []  # for storing normalized coordinates of points
#     labels = []  # for storing 0-1 labels for non-sharp/sharp points
#     normals = []
#     surface_rate = []  # for counting how many surfaces are there in the patch
#     sharp_rate = []  # for indicator whether the patch contains sharp vertices at all
#     times = []  # for times (useless)
#     p_names = []  # for names of the patches in the format "initial_mesh_name_N", where N is the starting vertex index
#


NBHOOD_BY_TYPE = {
    # 'geodesic_bfs': geodesic_meshvertex_patches_from_item,
    'euclidean_sphere': EuclideanSphere,
    'random_euclidean_sphere': RandomEuclideanSphere,
}


