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
        self.tree = KDTree(mesh.vertices)

    def get_nbhood(self):
        # select vertices falling within euclidean sphere
        _, vert_indices = self.tree.query(
            self.centroid, k=self.n_vertices, distance_upper_bound=self.radius)

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

        return neighbourhood, adj_vert_indices, self.mesh.faces[adj_face_indexes]

    @classmethod
    def from_config(cls, config):
        return cls(config['centroid'], config['radius'], config['n_vertices'])


class RandomEuclideanSphere(EuclideanSphere):
    def __init__(self, centroid, radius, n_vertices, radius_delta):
        super().__init__(centroid, radius, n_vertices)
        self.radius_delta = radius_delta

    def get_nbhood(self):
        centroid_idx = np.random.choice(len(self.mesh.vertices))
        self.centroid = self.mesh.vertices[centroid_idx]
        self.radius = np.random.uniform(
            self.radius - self.radius_delta,
            self.radius + self.radius_delta)
        return super(RandomEuclideanSphere, self).get_nbhood()

    @classmethod
    def from_config(cls, config):
        return cls(config['centroid'], config['radius'],
                   config['n_vertices'], config['radius_delta'])


NBHOOD_BY_TYPE = {
    # 'geodesic_bfs': geodesic_meshvertex_patches_from_item,
    'euclidean_sphere': EuclideanSphere,
    'random_euclidean_sphere': RandomEuclideanSphere,
}


