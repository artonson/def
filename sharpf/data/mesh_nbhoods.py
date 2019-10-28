from abc import ABC
from typing import Callable

from scipy.spatial import KDTree


class NeighbourhoodFunc(ABC, Callable):
    """Implements obtaining neighbourhoods from meshes.
    Given a mesh and a vertex, extracts its sub-mesh, i.e.
    a subset of vertices and edges, that correspond to
    a neighbourhood of some type."""
    def __call__(self, mesh, centroid, radius, **kwargs):
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
        raise NotImplementedError()


class EuclideanSphere(NeighbourhoodFunc):
    """Select all faces with at least one vertex within
    a specified radius from a specified point."""
    def __init__(self, mesh):
        self.mesh = mesh
        self.tree = KDTree(mesh.vertices)
        self.face_dict = None

    def __call__(self, centroid, radius, **kwargs):
        # TODO
        n_vert = 400

        _, vert_indices = self.tree.query(centroid, k=n_vert, distance_upper_bound=radius)

        vertices = set()
        faces_indices = []
        for vert_index in vert_indices:
            vert_faces = shape['face_dict'][vert_index]
            for face in vert_faces:
                # if any vertex from a face is inside the sphere pick this face
                if len(np.intersect1d(shape['faces'][face], list(vert_indices))) > 0:
                    faces_indices.append(face)
                    vertices.update(shape['faces'][face])
        vertices = list(vertices)
        #     print(vertices)
        reindex = dict(zip(vertices, range(len(vertices))))
        faces = []

        #     print(faces_indices)
        for face in np.array(shape['faces'])[faces_indices]:
            faces.append([reindex[vert] for vert in face])

        neighbourhood = trimesh.base.Trimesh(vertices=shape['vertices'][vertices],
                                             faces=faces,
                                             process=False,
                                             validate=False, )

        return mesh, vertices, faces_indices

        return neighbourhood


NBHOOD_BY_TYPE = {
    'geodesic_bfs': geodesic_meshvertex_patches_from_item,
    'euclidean_sphere': EuclideanSphere,
}

