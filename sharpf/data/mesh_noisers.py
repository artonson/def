from abc import ABC, abstractmethod

import trimesh as trm
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
    def __init__(self, face_angle, n_sample_adj_faces, nondegen_tri_height, n_noisy_patches):
        super(AddEdgesNoise, self).__init__()
        self.face_angle = face_angle
        self.n_sample_adj_faces = n_sample_adj_faces
        self.nondegen_tri_height = nondegen_tri_height
        self.n_noisy_patches = n_noisy_patches
        self.mesh = None
        self._idx = None

    def _flip_edges(self):
        # Get pairs of adjacent faces, shared edges, and unshared vertices
        adj_faces = self.mesh.face_adjacency[self._idx]
        adj_unshared_verts = self.mesh.face_adjacency_unshared[self._idx]
        adj_edges = self.mesh.face_adjacency_edges[self._idx]
        
        new_patches_list = []
        num_iter=5
        it = 0
        num_rej = 0
        num_added = 0
        
        if len(adj_faces) >= self.n_sample_adj_faces:
            # Loop for num_iter OR when required n_noisy_patches reaches
            while it < num_iter and len(new_patches_list) < self.n_noisy_patches:
                it += 1
                for i in range(self.n_noisy_patches):
                    if len(new_patches_list) < self.n_noisy_patches:
                        # Randomly choose 'N' pairs of adj. faces 
                        sample_idx = np.random.choice(len(adj_faces), size=self.n_sample_adj_faces, replace=False)
                        sample_adj_faces = adj_faces[sample_idx]
                        sample_unshared_verts = adj_unshared_verts[sample_idx]
                        sample_edges = adj_edges[sample_idx]

                        old_faces = self.mesh.copy().faces

                        # Add and remove edges
                        new_adding_faces = []
                        processed_faces = []
                        # f_count = 0

                        for f_count, adj in enumerate(sample_adj_faces):
                            # Check if any of faces in adj. pair is already processed
                            if not np.isin(processed_faces,adj).any():
                                dif = sample_unshared_verts[f_count]
                                same = sample_edges[f_count]

                                # Create two new triangles
                                # Remove their old shared edge and add a new edge b/w unshare vertices
                                new_face_1 = np.array([self.mesh.vertices[dif[0]], \
                                                    self.mesh.vertices[dif[1]], \
                                                    self.mesh.vertices[same[0]]]).reshape((1,3,3))
                                new_face_2 = np.array([self.mesh.vertices[dif[0]], \
                                                    self.mesh.vertices[dif[1]], \
                                                    self.mesh.vertices[same[1]]]).reshape((1,3,3))
                                
                                # Check if the new triangles are degenerated
                                if trm.triangles.nondegenerate(new_face_1, height=self.nondegen_tri_height).all() and \
                                    trm.triangles.nondegenerate(new_face_2, height=self.nondegen_tri_height).all():
                                    new_adding_faces.append(np.append(dif,same[0])) # add new face
                                    new_adding_faces.append(np.append(dif,same[1])) # add new face
                                    processed_faces.append(adj)
    #                                 print("new faces with verts: {}".format(new_adding_faces))
                                else:
                                    print("REJECTED!! Degenerate Triangles")

                            # f_count = f_count + 1

                        old_faces = np.asarray(old_faces)
                        mask = np.ones(len(old_faces), dtype=bool)
                        mask[np.asarray(processed_faces).reshape(-1)] = False
                        masked_faces = old_faces[mask,...]
                        new_faces = np.append(masked_faces,np.asarray(new_adding_faces),axis=0)
                        # Construct noisy mesh
                        noisy_mesh = trm.base.Trimesh(vertices = self.mesh.vertices, \
                                                        faces = new_faces, \
                                                        process = False)

                        # Fix faces flip
                        trm.repair.fix_winding(noisy_mesh)
                        if trm.triangles.nondegenerate(noisy_mesh.triangles, height=self.nondegen_tri_height).all():
                            # Add only patches with 50 % new adding faces
                            if len(new_adding_faces) > (self.n_sample_adj_faces*1.2):
                                print("patch {}/{} is ADDED with {}/{} new added faces.".format(i+1, self.n_noisy_patches, len(new_adding_faces), \
                                                                                                self.n_sample_adj_faces*2))
                                new_patches_list.append(noisy_mesh)
                                num_added += 1
                        else:
            #                 print("Mesh contains degenerate triangle(s) and REJECTED")
                            print("patch {}/{} is REJECTED".format(i+1, self.n_noisy_patches))
                            num_rej += 1
                        print("-----------------------------------")
            print("After {} tries: {} ADDED and {} REJECTED.".format(it, num_added, num_rej))
        
        return new_patches_list

    def make_copies(self, mesh):
        self.mesh = mesh
        # Select only pairs that the angle b/w faces < face_angle radius
        self._idx = (self.mesh.face_adjacency_angles < self.face_angle) # 5 degree = 0.087 rad.
        # new_patches_list = self._flip_edges()
        return self._flip_edges()

    def make_noise(self, mesh):
        self.mesh = mesh
        # Select only pairs that the angle b/w faces > 0.17 radius
        self._idx = (self.mesh.face_adjacency_angles > self.face_angle) # 10 degree
        # new_patches_list = self._flip_edges()
        return self._flip_edges()

    @classmethod
    def from_config(cls, config):
        return cls(config['face_angle'], config['n_sample_adj_faces'],
                   config['nondegen_tri_height'], config['n_noisy_patches'])


class NoNoise(NoiserFunc):
    def make_noise(self): return points

    @classmethod
    def from_config(cls, config): return cls()


MESH_NOISE_BY_TYPE = {
    'no_noise': NoNoise,
    'add_edges_noise': AddEdgesNoise,
}

