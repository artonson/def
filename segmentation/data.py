import os
import h5py
import torch
import numpy as np
import matplotlib.pyplot as plt
from torch.utils.data import DataLoader
from torch.utils.data import Dataset
import segmentation_models_pytorch as smp
from scipy.stats import special_ortho_group

class dataset(Dataset):
    def __init__(self, mode='train'):
        with h5py.File('/home/artonson/tmp/abc/patches/abc_0020_4832_4983.hdf5', 'r') as f:
            points = np.array(f['points'])
            distances = np.array(f['distances'])
            directions = np.array(f['directions'])
        if mode == 'train':
            self.points = points[:int(len(points)*0.2)]
            self.distances = distances[:int(len(distances)*0.2)]
            self.directions = directions[:int(len(directions)*0.2)]
        elif mode == 'val':
            self.points = points[int(len(points)*0.5):]
            self.distances = distances[int(len(distances)*0.5):]
            self.directions = directions[int(len(directions)*0.5):]

    def __len__(self):
        return len(self.points)

    def __getitem__(self, idx):
        R = special_ortho_group.rvs(3)
        curr_points = np.dot((self.points[idx] - np.mean(self.points[idx])), R)[:,:-1]
        x,y = np.meshgrid(np.linspace(min(curr_points[:,0]), max(curr_points[:,0]), 33), np.linspace(min(curr_points[:,1]), max(curr_points[:,1]), 33))
        polygons = []
        polygon_inds = []
        for i in range(len(x)-1):
            for j in range(len(x[i])-1):
                x1, y1, x2, y2 = x[i][j], y[i][j], x[i][j+1], y[i+1][j]
                tmp = []
                tmp_ind = []
                for ind, p in enumerate(curr_points):
                    if (p[0] <= x2 and p[0] >= x1 and p[1] <= y2 and p[1] >= y1):
                        tmp.append(p)
tmp_ind.append(ind)
                polygons.append(tmp)
                polygon_inds.append(tmp_ind)

        i = 0
        j = 0
        depth = []
        points_4040 = []
        distance = []
        for el in polygon_inds:
            if len(el) > 0:
                ind_z = np.argmax(curr_points[:, -1][el])
                z = curr_points[:, -1][el[ind_z]]
                points_4040.append([curr_points[:, 0][el[ind_z]], curr_points[:, 1][el[ind_z]]])
                dist = self.distances[idx][el[ind_z]]
            else:
                z = 0
                dist = 3.5

            depth.append(z)
            distance.append(dist)

        distance = np.array(distance)
        distance /= max(distance)
        distance -= 1
        distance = abs(distance)
        distance = torch.FloatTensor(np.array([distance]).reshape((1, 32, 32)))

        depth = np.array(depth)
        depth -= min(depth)
        depth /= (max(depth) - min(depth))
        depth = torch.FloatTensor(depth.reshape((1, 32, 32)))
        depth = torch.cat([depth, depth, depth], axis=0)

        return depth, distance
