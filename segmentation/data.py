import os
import h5py
import torch
import numpy as np
from torch.utils.data import Dataset

class dataset(Dataset):
    def __init__(self, mode='train', shape=(32, 32), download=False):
        with h5py.File('./abc_0020_4832_4983.hdf5', 'r') as f:
            points = np.array(f['points'])
            distances = np.array(f['distances'])
            directions = np.array(f['directions'])
        self.orig_len = len(points)
        if mode == 'train':
            self.start = 0
            self.points = points[:int(len(points)*0.9)]
            self.distances = distances[:int(len(distances)*0.9)]
            self.directions = directions[:int(len(directions)*0.9)]
        elif mode == 'val':
            self.start = int(len(points)*0.9)
            self.points = points[int(len(points)*0.9):]
            self.distances = distances[int(len(distances)*0.9):]
            self.directions = directions[int(len(directions)*0.9):]

        self.shape = shape
        if download:
            self.download()

        self.depths = []
        self.distances_new = []
        for idx in range(len(self.points)):
            #print('{}/{}'.format(idx, len(self.points)))
            i = idx + self.start
            self.depths.append(torch.load('./depth/depth_{}'.format(i)))
            self.depths.append(torch.load('./depth_2/depth_{}'.format(i)))

            self.distances_new.append(torch.load('./depth/dist_{}'.format(i)))
            self.distances_new.append(torch.load('./depth_2/dist_{}'.format(i)))

        self.mean = 0
        for dist in self.distances_new:
            dist[np.isnan(dist)] = 0.0
            self.mean += torch.sum(dist)
        self.mean /= len(self.distances_new)

    def download(self):
        for idx in range(self.start+len(self.points)):
            print('saved {}/{}'.format(idx, len(self.points)))
            R = special_ortho_group.rvs(3)
            rotated_points = np.dot((self.points[idx] - np.mean(self.points[idx])), R)
            rotated_points[:,-1] += np.max(rotated_points[:,-1])
            curr_points = rotated_points[:,:-1]
            x,y = np.meshgrid(np.linspace(min(curr_points[:,0]), max(curr_points[:,0]), self.shape[0]+1), np.linspace(min(curr_points[:,1]), max(curr_points[:,1]), self.shape[-1]+1))
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
            points_3232 = []
            distances_3232 = []
            distance = []
            count = 0
            for el in polygon_inds:
                if len(el) > 0:
                    count += 1
                    ind_z = np.argmin(rotated_points[:, -1][el])
                    z = rotated_points[:, -1][el[ind_z]]
                    points_3232.append([curr_points[:, 0][el[ind_z]], curr_points[:, 1][el[ind_z]], z])
                    distances_3232.append(self.distances[idx][el[ind_z]])
                    dist = self.distances[idx][el[ind_z]]
                else:
                    z = np.nan
                    dist = np.nan
                depth.append(z)
                distance.append(dist)
                print('points saved {}'.format(count))
            points_3232 = np.array(points_3232)
            distances_3232 = np.array(distances_3232)


            distance = np.array(distance)
            distance[distance > 1.0] = 1.0
            if np.nanmax(distance) > 0.0:
                distance /= np.nanmax(distance)
            distance -= 1
            distance = abs(distance)

            depth = np.array(depth)
            depth -= np.nanmin(depth)
            if (np.nanmax(depth) - np.nanmin(depth)) > 0.0:
                depth /= (np.nanmax(depth) - np.nanmin(depth))
            torch.save(torch.FloatTensor(depth), './depth_2/depth_{}'.format(idx+self.start))
            torch.save(torch.FloatTensor(distance), './depth_2/dist_{}'.format(idx+self.start))

    def __len__(self):
        return len(self.points)*2
        
    def __getitem__(self, idx):
        depth = torch.reshape(self.depths[idx], (1,self.shape[0],self.shape[-1]))
        mask = np.array(~np.isnan(depth))

        depth[np.isnan(depth)] = 0
        mask = torch.reshape(torch.FloatTensor(mask), (1,self.shape[0],self.shape[-1]))
        distance = self.distances_new[idx]#torch.reshape(self.distances_new[idx], (1,self.shape[0],self.shape[-1]))
        distance[np.isnan(distance)] = 0.0
        distance[distance > 0.6] = True
        #distance[(distance < 0.6) & (distance > 0.2)] = 0.5
        distance[distance < 0.6] = False
        target = np.any(distance.numpy(), axis=0)*1

        depth = torch.cat([depth, depth, depth, mask], dim=0)
        #depth = torch.cat([depth, depth, depth], dim=0) 
        return depth, torch.FloatTensor([target])

