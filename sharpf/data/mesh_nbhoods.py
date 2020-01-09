from abc import ABC, abstractmethod

import numpy as np
import trimesh
from scipy.spatial import KDTree


class NeighbourhoodFunc(ABC):
    """Implements obtaining neighbourhoods from meshes.
    Given a mesh and a vertex, extracts its sub-mesh, i.e.
    a subset of vertices and edges, that correspond to
    a neighbourhood of some type."""

    @abstractmethod
    def get_nbhood(self):
        """Extracts a mesh neighbourhood.

        :param mesh: an input mesh
        :type mesh: MeshType (must be present attributes `vertices`, `faces`, and `edges`)

        :param radius: a radius
        :type radius: float
        TODO maybe ring radius?

        :param centroid: index of a vertex in the mesh
        TODO maybe xyz?

        :param kwargs: optional parameters (see descendant classes)

        :returns: neighbourhood: mesh whose faces are within a a
        :rtype: MeshType (must be present attributes `vertices`, `faces`, and `edges`)
        """
        pass

    @classmethod
    def from_config(cls, config):
        pass


class EuclideanSphere(NeighbourhoodFunc):
    """Select all faces with at least one vertex within
    a specified radius from a specified point."""
    def __init__(self, centroid, radius, n_vertices):
        self.centroid = centroid
        self.radius = radius
        self.n_vertices = n_vertices
        self.mesh = None
        self.tree = None

    def index(self, mesh):
        self.mesh = mesh
        self.tree = KDTree(mesh.vertices, leafsize=100)

    def get_nbhood(self, geodesic_patches=False):
        # select vertices falling within euclidean sphere
        radius_scaler = self.mesh.edges_unique_length.mean()
        _, vert_indices = self.tree.query(
            self.centroid, k=self.n_vertices, distance_upper_bound=self.radius * radius_scaler)

        # get all faces that share vertices with selected vertices
        vert_indices = vert_indices[vert_indices < len(self.mesh.vertices)]
        adj_face_indexes = self.mesh.vertex_faces[vert_indices]
        adj_face_indexes = np.unique(adj_face_indexes[adj_face_indexes > -1])

        # get all vertices that sit on adjacent faces
        adj_vert_indices = self.mesh.faces[adj_face_indexes]
        adj_vert_indices = np.unique(adj_vert_indices)

        # copy vertices, reindex faces
        selected_vertices = self.mesh.vertices[adj_vert_indices]
        selected_faces = np.array(self.mesh.faces[adj_face_indexes])
        for reindex, index in zip(np.arange(len(selected_vertices)), adj_vert_indices):
            selected_faces[np.where(selected_faces == index)] = reindex

        # push the selected stuff into a trimesh
        neighbourhood = trimesh.base.Trimesh(
            vertices=selected_vertices,
            faces=selected_faces,
            process=False,
            validate=False)
        
        # get the connected component with maximal area
        if geodesic_patches:
            if len(neighbourhood.split(only_watertight=False)) > 1:
                areas = []
                for submesh in neighbourhood.split(only_watertight=False):
                    areas.append(submesh.area)
                neighbourhood = neighbourhood.split(only_watertight=False)[np.array(areas).argmax()]
                
                geodesic_mask = []
                for v in selected_vertices:
                    geodesic_mask.append(np.isin(neighbourhood.vertices, v).all(-1).any())
                adj_vert_indices = adj_vert_indices[geodesic_mask]
                adj_face_indexes = self.mesh.vertex_faces[adj_vert_indices]
                adj_face_indexes = np.unique(adj_face_indexes[adj_face_indexes > -1])       
        
        return neighbourhood, adj_vert_indices, self.mesh.faces[adj_face_indexes], radius_scaler

    @classmethod
    def from_config(cls, config):
        return cls(config['centroid'], config['radius'], config['n_vertices'])


class RandomEuclideanSphere(EuclideanSphere):
    def __init__(self, centroid, radius, n_vertices, radius_delta):
        super().__init__(centroid, radius, n_vertices)
        self.radius_delta = radius_delta

    def get_nbhood(self, geodesic_patches=False):
        centroid_idx = np.random.choice(len(self.mesh.vertices))
        print(centroid_idx)
        self.centroid = self.mesh.vertices[centroid_idx]
        self.radius = np.random.uniform(
            self.radius - self.radius_delta,
            self.radius + self.radius_delta)
        return super(RandomEuclideanSphere, self).get_nbhood(geodesic_patches)

    @classmethod
    def from_config(cls, config):
        return cls(config['centroid'], config['radius'],
                   config['n_vertices'], config['radius_delta'])

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


