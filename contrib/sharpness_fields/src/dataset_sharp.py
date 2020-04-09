import numpy as np
import torch
import torch.utils.data
from scipy.spatial import cKDTree
import random


class PointCloudDataset(torch.utils.data.Dataset):

    def __init__(self, 
                xyz_list, 
                k=48, 
                max_neighbors=100, 
                r_factor_range = (2.0, 9.0),
                training = False):

        self.xyz_list = xyz_list
        self.k = k
        self.max_neighbors = max_neighbors
        self.r_factor_range = r_factor_range
        self.training = training

        self.kdtrees = []
        self.vertexCounts = []
        self.points = []
        self.normals = []
        self.tangents = []
        self.sharps = []
        self.median_separations = []

        for path in xyz_list:

            data = np.loadtxt(path, dtype=np.float32)
            
            sharps_available = (data.shape[-1] % 7 == 0)

            if sharps_available:
                data = data.reshape(-1, 7)
                self.sharps.append(data[:, 6])
            else:
                data = data.reshape(-1, 6)
                self.sharps.append(np.zeros_like(data[:,0]))

            self.points.append(data[:, 0:3])
            normals = data[:, 3:6]
            self.normals.append(normals)

            y_axis = np.array([0,1,0], dtype=np.float32)
            z_axis = np.array([0,0,1], dtype=np.float32)

            tangents = np.cross(normals, y_axis)
            badIndices = np.linalg.norm(tangents, axis=1) < 0.001
            tangents[badIndices] = np.cross(normals[badIndices], z_axis)

            tangents = tangents / np.linalg.norm(tangents, axis=1)[:, np.newaxis]

            self.tangents.append(tangents)
                
            self.vertexCounts.append(data.shape[0])

            #compute median distance between nearest neighbors (delta in the paper)

            kdtree = cKDTree(self.points[-1])
            self.kdtrees.append(kdtree)

            dists, indices = kdtree.query(self.points[-1], k=2, n_jobs=-1)
            #indices shape is (num_vertices, 2)

            neighborPairs1 = self.points[-1][indices[:,0], :]
            neighborPairs2 = self.points[-1][indices[:,1], :]

            separations = neighborPairs1 - neighborPairs2
            separations = np.linalg.norm(separations, axis=-1)
            medianSeparation = np.median(separations)
            self.median_separations.append(medianSeparation)

        self.cumVertexCount = np.cumsum(self.vertexCounts)

        print(self.cumVertexCount )

    def __len__(self):
        return self.cumVertexCount[-1]


    def __getitem__(self, item):

        while item < 0:
            item += len(self)

        modelIndex = np.argmax(self.cumVertexCount > item)
        vertIndex = item if modelIndex == 0 else item - self.cumVertexCount[modelIndex-1]

        kdtree = self.kdtrees[modelIndex]
        median_separation = self.median_separations[modelIndex]
        r_factor = random.uniform(*self.r_factor_range)
        r = median_separation * r_factor
        points = self.points[modelIndex]
        normals = self.normals[modelIndex]
        tangents = self.tangents[modelIndex]
        sharps = self.sharps[modelIndex]

        centerPoint = points[vertIndex]
        centerNormal = normals[vertIndex]

        normalFlipped = self.training and bool(random.getrandbits(1))

        if normalFlipped:
            centerNormal = -centerNormal

        centerTangent = tangents[vertIndex]
        centerBitangent = np.cross(centerNormal, centerTangent)
        centerBitangent /= np.linalg.norm(centerBitangent)
        centerSharp = sharps[vertIndex]

        if self.training:
            randomAngle = np.random.uniform(0.0, 2 * np.pi)
            centerTangent = np.cos(randomAngle) * centerTangent + np.sin(randomAngle) * centerBitangent
            centerTangent /= np.linalg.norm(centerTangent)
            centerBitangent = np.cross(centerNormal, centerTangent)
            centerBitangent /= np.linalg.norm(centerBitangent)


        indices = kdtree.query_ball_point(centerPoint, r=r)
        indices = np.array(indices)

        num_neighbors = len(indices)

        if num_neighbors > self.max_neighbors:
            indices = np.random.choice(indices, size=self.max_neighbors)
            num_neighbors = self.max_neighbors

        neighbors = points[indices]
        neighbor_normals = normals[indices]

        if normalFlipped:
            neighbor_normals = -neighbor_normals

        neighbors.resize((self.max_neighbors, 3))
        neighbor_normals.resize((self.max_neighbors, 3))

        return torch.from_numpy(neighbors), \
                torch.from_numpy(neighbor_normals), \
                torch.tensor([num_neighbors], dtype=torch.int32), \
                torch.from_numpy(centerPoint), torch.from_numpy(centerNormal), \
                torch.from_numpy(centerTangent), torch.from_numpy(centerBitangent), \
                torch.tensor([r], dtype=torch.float32), \
                torch.tensor([centerSharp], dtype=torch.float32)
       