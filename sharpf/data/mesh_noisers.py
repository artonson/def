from abc import ABC, abstractmethod

import trimesh
import numpy as np
import copy


class NoiserFunc(ABC):
    """Implements obtaining neighbourhoods from meshes.
    Given a mesh and a vertex, extracts its sub-mesh, i.e.
    a subset of vertices and edges, that correspond to
    a neighbourhood of some type."""
    @abstractmethod
    def make_noise(self, mesh):
        """Noises a mesh.
        - Find a pair of adjacent faces where the angle between them is not equal to 0
        - Add a new edge between uncommon vertices of adjacent faces (intersecting the shared edge)
        - Remove the shared edge
        - Remove the processed faces from the list

        :param mesh: an input mesh
        :type mesh: MeshType (must be present attributes `vertices`, `faces`, and `edges`)

        :returns: noisy mesh:
        :rtype: MeshType (must be present attributes `vertices`, `faces`, and `edges`)
        """
        pass

    @classmethod
    def from_config(cls, config):
        pass


class AddEdgesNoise(NoiserFunc):
    """Add and Remove edges"""
    def __init__(self, face_angle, n_sample_faces, nondegen_tri_height):
        super(AddEdgesNoise, self).__init__()
        self.face_angle = face_angle
        self.n_sample_faces = n_sample_faces
        self.nondegen_tri_height = nondegen_tri_height
        self.mesh = None

    def make_noise(self, mesh):
        self.mesh = mesh
        print("# of input vertices: {}".format(self.mesh.vertices.shape))
        print("# of input faces: {}".format(self.mesh.faces.shape))
        print("face angle: {}".format(self.face_angle))

        # Get pairs of adjacent faces, shared edges, and unshared vertices
        adj_faces = self.mesh.face_adjacency
        adj_unshared_verts = self.mesh.face_adjacency_unshared
        adj_edges = self.mesh.face_adjacency_edges

        # Select only pairs that the angle b/w faces > 0.17 radius
        idx = (self.mesh.face_adjacency_angles > self.face_angle) # 10 degree
        sample_faces = adj_faces[idx]
        sample_unshared_verts = adj_unshared_verts[idx]
        sample_edges = adj_edges[idx]

        # From those selected pairs, we only take 10 of them for editing
        if len(sample_faces) >= self.n_sample_faces:
            sample_idx = np.random.choice(len(sample_faces), \
                                        size=self.n_sample_faces, replace=False)
            sample_faces = sample_faces[sample_idx]
            sample_unshared_verts = sample_unshared_verts[sample_idx]
            sample_edges = sample_edges[sample_idx]
            
        if len(sample_faces) == 0:
            sample_idx = np.random.choice(len(adj_faces), \
                                        size=self.n_sample_faces, replace=False)
            sample_faces = adj_faces[sample_idx]
            sample_edges = adj_edges[sample_idx]
            sample_unshared_verts = adj_unshared_verts[sample_idx]

        old_faces = self.mesh.copy().faces

        # Add and remove edges
        added_face = []
        processed_faces = []
        f_count = 0
        for adj in sample_faces:
        #    print("processing face with verts: {}".format(ptch.faces[adj]))
            if not np.isin(processed_faces,adj).any():
                dif = sample_unshared_verts[f_count]
                same = sample_edges[f_count]

                # Check if the new triangle is degenerated 
                # (Degenerate angles will be returned as zero)
                new_face_1 = np.array([self.mesh.vertices[dif[0]], \
                                    self.mesh.vertices[dif[1]], \
                                    self.mesh.vertices[same[0]]]).reshape((1,3,3))
                new_face_2 = np.array([self.mesh.vertices[dif[0]], \
                                    self.mesh.vertices[dif[1]], \
                                    self.mesh.vertices[same[1]]]).reshape((1,3,3))

                if np.any(trimesh.triangles.angles(new_face_1)) and \
                np.any(trimesh.triangles.angles(new_face_2)):
                    added_face.append(np.append(dif,same[0])) # add new face
                    added_face.append(np.append(dif,same[1])) # add new face
                    # print("new faces with verts: {}".format(added_face))
                else:
                    print("REJECTED!! Degenerate Triangles")

                processed_faces.append(adj)
                # print("processed_faces {}".format(processed_faces))
            else:
                print("One of faces is already processed and {} is skipped.".format(f_count))
            f_count = f_count + 1

        old_faces = np.asarray(old_faces)
        mask = np.ones(len(old_faces), dtype=bool)
        mask[np.asarray(processed_faces).reshape(-1)] = False
        masked_faces = old_faces[mask,...]
        new_faces = np.append(masked_faces,np.asarray(added_face),axis=0)
        # Construct noisy mesh
        noisy_mesh = trimesh.base.Trimesh(vertices = self.mesh.vertices, \
                                        faces = new_faces, \
                                        process = False)
        # Fix faces flip
        trimesh.repair.fix_winding(noisy_mesh)
        assert (trimesh.triangles.nondegenerate(noisy_mesh.triangles, height=self.nondegen_tri_height).all()), \
               "Mesh contains degenerate triangle(s)"
        
        print("Sample faces: {}".format(len(sample_faces)))
        print("# of vertics: {}".format(noisy_mesh.vertices.shape))
        print("# of faces: {}".format(noisy_mesh.faces.shape))
        return noisy_mesh

    @classmethod
    def from_config(cls, config):
        return cls(config['face_angle'], config['n_sample_faces'],
                   config['nondegen_tri_height'])


class NoNoise(NoiserFunc):
    def make_noise(self): return points

    @classmethod
    def from_config(cls, config): return cls()


MESH_NOISE_BY_TYPE = {
    'no_noise': NoNoise,
    'add_edges_noise': AddEdgesNoise,
}

