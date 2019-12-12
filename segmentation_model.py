import os
import time
import sys
import h5py
import torch
import numpy as np
import matplotlib.pyplot as plt
from torch.utils.data import DataLoader
from torch.utils.data import Dataset
from torch.nn import functional as F
import segmentation_models_pytorch as smp
from seg_models_lib.segmentation_models_pytorch.unet import Classifier, Unet
from seg_models_lib.segmentation_models_pytorch.utils.losses import BCEDiceLoss
from seg_models_lib.segmentation_models_pytorch.utils.train import ValidEpoch, TrainEpoch
from scipy.stats import special_ortho_group
import trimesh
from scipy.spatial import KDTree

import glob
from tqdm import tqdm
import itertools
from collections import defaultdict
import pandas as pd
import sys
from pathlib import Path
import pywavefront

#sys.path.append('..')

#from notebooks.io import write_ply
from sampling import get_config, process_vertices, fibonacci_sphere_sampling
from view import from_pose
from math import radians, degrees
from trimesh.ray.ray_pyembree import RayMeshIntersector
import math

import logging
logger = logging.getLogger("pywavefront")
logger.disabled=True

import view

sharp_discretization = 0.1
distance_upper_bound = 3.5

def myfunc(n, d):
    return np.linspace(d, 1 - d, n)
def myfunc_2(a, b, adj_graph):
    neighbors = np.array(list(adj_graph.neighbors(a)))[:, None]
    #neighbors = np.array(list(map(lambda x: [a, x], adj_graph.neighbors(a))))[None, ...]

    b_ = np.array(b)[None, ...]
    ind = np.any(neighbors == b_, axis=1)

    return [a, neighbors[ind][0][0]]

def resample_sharp_edges(mesh_patch, features):
    #print('resampling sharp edges...')

    adj_graph = mesh_patch.vertex_adjacency_graph # neighbors
    sharp_features = list(filter(lambda a: a if a['sharp'] is True else None, features['curves']))
    sharp_points = []
    # sharp points: mesh_patch.vertices[el['vert_indices']].shape = (N, 2), combinations.shape = (M, 2)

    for i, sharp_feat in enumerate(sharp_features):
        #print(i, '/', len(sharp_features))

        # there may be no sharp edges, but single vertices may be present -- add them
        sharp_points.append(mesh_patch.vertices[sharp_feat['vert_indices']])
        sharp_vert_indices = sharp_features[i]['vert_indices']  # = curve['vert_indices']

        sharp_edges = np.array(list(map(myfunc_2, sharp_vert_indices, [sharp_vert_indices]*len(sharp_vert_indices), [adj_graph]*len(sharp_vert_indices))))
        sharp_edges = sharp_edges.reshape(sharp_edges.shape[0], 2)

        # edges1 = mesh_patch.edges[:, None, :]
        # edges2 = combinations[None, ...]
        # sharp_edges = mesh_patch.edges[(edges1 == edges2).all(axis=2).any(axis=1)].reshape(-1, 2)

        first, second = mesh_patch.vertices[sharp_edges[:, 0]], mesh_patch.vertices[sharp_edges[:, 1]]
        n_points_per_edge = np.linalg.norm(first - second, axis=1) / sharp_discretization
        d_points_per_edge = 1. / n_points_per_edge
        for n, v1, v2 in zip(n_points_per_edge, first, second):
                t = np.linspace(d_points_per_edge, 1 - d_points_per_edge, n)
                sharp_points.append(np.outer(t, v1) + np.outer(1 - t, v2))

        # t = np.array(list(map(myfunc, n_points_per_edge, np.full(len(n_points_per_edge), d_points_per_edge))))
        # sharp_points.append(
        #     (np.outer(t.reshape(-1), first.reshape(-1)) + np.outer(1 - t.reshape(-1), second.reshape(-1))).reshape(-1, 3))


    if len(sharp_points) > 0:
        sharp_points = np.concatenate(sharp_points)
    return sharp_points
    #     sharp_edges = np.array([
    #         (mesh_patch.vertices[pair_i_j[0]], mesh_patch.vertices[pair_i_j[1]]) for pair_i_j in itertools.combinations(np.unique(sharp_vert_indices), 2)
    #         if (mesh_patch.edges == pair_i_j).all(axis=1).any(axis=0)
    #     ])
    #     time_2 = time.clock()
    #     print('#2', time_2 - start_time)
    #     if len(sharp_edges) > 0:
    #         first, second = sharp_edges[:, 0], sharp_edges[:, 1]#mesh_patch.vertices[sharp_edges[:, 0]], mesh_patch.vertices[sharp_edges[:, 1]]
    #         n_points_per_edge = np.linalg.norm(first - second, axis=1) / sharp_discretization
    #         d_points_per_edge = 1. / n_points_per_edge
    #         for n, v1, v2 in zip(n_points_per_edge, first, second):
    #             t = np.linspace(d_points_per_edge, 1 - d_points_per_edge, n)
    #             sharp_points_el.append(np.outer(t, v1) + np.outer(1 - t, v2))
    #     time_3 = time.clock()
    #     print('#3', time_3 - start_time)
    # print(sharp_points)


    #tmp = list([_resample_sharp_edges(curve, i, features, mesh_patch) for i, curve in enumerate(features['curves'])])
    # if len(tmp) > 0:
    #     return np.concatenate(np.concatenate(list(tmp)))
    # else:
    #     return []


def annotate(sharp_points, mesh_patch, features_patch, points, distance_upper_bound, distance_scaler=1, **kwargs):
    print('annotating...')
    # if patch is without sharp features
    directions = []
    if all(not curve['sharp'] for curve in features_patch['curves']):
        distances = np.ones_like(points[:, 0]) * distance_upper_bound
        directions = np.zeros_like(points)
        return distances, directions

    # if all patch sharp features are nearby the patch edge
    if np.linalg.norm(sharp_points - mesh_patch.vertices.mean(0), axis=1).min() / np.linalg.norm(
            mesh_patch.vertices - mesh_patch.vertices.mean(0), axis=1).max() >= 0.9:
        distances = np.ones_like(points[:, 0]) * distance_upper_bound
        directions = np.zeros_like(points)
        return distances, directions

    # compute distances from each input point to the sharp points
    print('computing distances')
    print(sharp_points.shape)
    tree = KDTree(sharp_points, leafsize=100)
    distances, vert_indices = tree.query(points, distance_upper_bound=distance_upper_bound * distance_scaler)
    distances = distances / distance_scaler

    far_from_sharp = distances == np.inf  # boolean mask marking objects far away from sharp curves
    distances[far_from_sharp] = distance_upper_bound

    # compute directions for points close to sharp curves
    # directions = np.zeros_like(points)
    # directions[~far_from_sharp] = sharp_points[vert_indices[~far_from_sharp]] - points[~far_from_sharp]
    # directions[~far_from_sharp] /= np.linalg.norm(directions[~far_from_sharp], axis=1, keepdims=True)
    # # fix NaNs
    # nan_inds = np.unique(np.where(np.isnan(directions))[0])
    # non_nan_inds = np.setdiff1d(np.arange(len(directions)), nan_inds)
    # tree = KDTree(points, leafsize=100)
    # _, non_nan_indices = tree.query(points[nan_inds])
    # directions[nan_inds] = directions[non_nan_inds][non_nan_indices]

    #print('end annotating')
    return distances, directions


def ray_cast_mesh(mesh, labels, image_height=512, image_width=512, num_angles=4, z_shift=-3, ortho_scale=3.0):
    #print('ray cast mesh')
    mesh_grid = np.mgrid[0:image_height, 0:image_width].reshape(2, image_height*image_width).T # screen coordinates

    # to [0, 1]
    aspect=image_width/image_height
    rays_origins=(mesh_grid/np.array([[image_height, image_width]]))

    # to [-1, 1] + aspect transform
    rays_origins[:,0] = (-2*rays_origins[:,0]+1)*ortho_scale/2
    rays_origins[:,1] = (-2*rays_origins[:,1]+1)*ortho_scale/2*aspect

    rays_origins = np.concatenate([rays_origins, np.zeros_like(rays_origins[:,[0]])], axis=1)
    ray_directions = np.tile(np.array([0,0,-1]), (rays_origins.shape[0], 1))

    angles = np.linspace(0, np.pi, num_angles)
    renders = []
    distances = []
    poses = []

    mesh_normalized = mesh.copy()
    mesh_normalized.vertices -= np.mean(mesh_normalized.vertices)
    mesh_normalized.vertices /= np.linalg.norm(mesh_normalized.vertices, axis=1).max()
    mesh_normalized.vertices *= 10
    # model a dense sample of points lying on sharp edges
    sharp_points = resample_sharp_edges(mesh_normalized, labels)
    del mesh_normalized
    i = 0
    for (x, y, z) in itertools.combinations_with_replacement(angles, 3):
        i += 1
        # equivalent
        transform = view.from_pose([x, y, z], [0,0,0])
        # moving the object
        mesh_ = mesh.copy()
        mesh_.apply_transform(transform)
        mesh_.apply_translation([0, 0, z_shift])

        mesh_normalized = mesh_.copy()
        mean = np.mean(mesh_normalized.vertices)
        mesh_normalized.vertices -= mean
        scale = np.linalg.norm(mesh_normalized.vertices, axis=1).max()
        mesh_normalized.vertices /= scale
        mesh_normalized.vertices *= 10

        ray_mesh = RayMeshIntersector(mesh_)
        torch.save(ray_mesh, '/home/gbobrovskih/abc/{}'.format(i))
        index_triangles, index_ray, values = ray_mesh.intersects_id(ray_origins=rays_origins, ray_directions=ray_directions, \
                                                                multiple_hits=False, return_locations=True)

        render = np.zeros((image_height, image_width))
        render[mesh_grid[index_ray][:, 0], mesh_grid[index_ray][:, 1]] = values[:,2]

        point_cloud = np.copy(values)
        point_cloud -= mean
        point_cloud /= scale
        point_cloud *= 10

        annotation = annotate(sharp_points, mesh_normalized, labels, point_cloud, distance_upper_bound=3.2)#mesh.edges_unique_length.mean()*10)

        dist = np.zeros((image_height, image_width))
        dist[mesh_grid[index_ray][:, 0], mesh_grid[index_ray][:, 1]] = annotation[0]
        #torch.save(render, '/home/gbobrovskih/abc/{}'.format(i))
        renders.append(render)
        distances.append(dist)
        poses.append({'rotation': (x, y, z), 'location': (0, 0, z_shift)})
        #print('next depth')
        del transform, mesh_, mesh_normalized
    #print('end ray cast mesh')
    return renders, distances, poses

class dataset(Dataset):
    def __init__(self, mode='train', download=False):
        if download:
            print('downloading')
            self.download()
        depths_id = os.listdir('/home/gbobrovskih/abc/depths')
        dists_id = os.listdir('/home/gbobrovskih/abc/dists')
        self.depths = []
        self.dists = []
        if mode == 'train':
            depths_id = depths_id[:int(len(shapes_ids)*0.9)]
            dists_id = dists_id[:int(len(shapes_ids) * 0.9)]
        elif mode == 'val':
            depths_id = depths_id[int(len(shapes_ids) * 0.9):]
            dists_id = dists_id[int(len(shapes_ids) * 0.9):]

        for i in range(len(depths_id)):
            depths.append(torch.load('/home/gbobrovskih/abc/depths/{}'.format(depths_id[i])))
            dists.append(torch.load('/home/gbobrovskih/abc/dists/{}'.format(dists_id[i])))

    def download(self, dir_in='/home/gbobrovskih/abc', dir_out='/home/gbobrovskih/abc/'):
        sys.setrecursionlimit(100000)
        shapes_ids = os.listdir(dir_in)
          
        data_path = Path('/home/gbobrovskih/abc')
        count = 0
        for shape_id in tqdm(shapes_ids[::-1]):
            print('id {}'.format(shape_id))
            print('{}/{}'.format(count, len(shapes_ids)))
            count += 1

            if not os.path.exists(data_path/shape_id):
                continue
            if len(os.listdir(data_path/shape_id)) == 0:
                continue
            mesh_file = glob.glob((data_path/shape_id/'*.obj').as_posix())[0]

            labels_file = glob.glob((data_path/shape_id/'*features*.yml').as_posix())[0]
            labels = get_config(labels_file)
            scene = pywavefront.Wavefront(mesh_file, collect_faces=True)
            vertices = process_vertices(np.array(scene.vertices))
            faces = scene.mesh_list[0].faces
            edges = set()
            for face in faces:
                for edge in itertools.combinations(face, 2):
                    edges.add(edge)

            mesh = trimesh.base.Trimesh(vertices=vertices,
                                faces=faces,
                                process=False,
                                validate=False,)

            shapes = {'vertices': vertices, 'faces': faces, 'labels': labels, 'edges': edges, 'mesh': mesh}

            renders, distances, poses = ray_cast_mesh(mesh, labels, ortho_scale=2.5)

            for i in range(len(renders)):
                torch.save(torch.FloatTensor(renders[i]), dir_out + 'depths/render_id{}_{}'.format(i, shape_id))
                torch.save(torch.FloatTensor(distances[i]), dir_out + 'dists/distance_id{}_{}'.format(i, shape_id))

    def normalize(self, data):
        norm_data = np.copy(data)
        norm_data -= np.mean(norm_data)
        norm_data /= np.linalg.norm(norm_data, axis=1).max()
        return norm_data

    def __getitem__(self, idx):
        depth = self.normalize(self.depth[idx])
        dist = self.normalize(self.dists[idx])
        return depth, dist

class dummy_dataset(Dataset):
    def __init__(self, mode='train', shape=(32, 32), task='segmentation', download=False, dirr='./depth/' ):
        self.count = 0
        with h5py.File('./abc_0020_4832_4983.hdf5', 'r') as f:
            points = np.array(f['points'])
            distances = np.array(f['distances'])
            directions = np.array(f['directions'])
        self.orig_len = len(points)
        if mode == 'train':
            #self.start = int(len(points)*0.1)
            self.points = points#[int(len(points)*0.1):]
            self.distances = distances#[int(len(distances)*0.1):]
            self.directions = directions#[int(len(directions)*0.1):]
        elif mode == 'val':
            #self.start = 0#int(len(points)*0.9)
            self.points = points#[0:int(len(points)*0.1)]
            self.distances = distances#[0:int(len(distances)*0.1)]
            self.directions = directions#[0:int(len(directions)*0.1)]
        self.start = int(len(points)*0.1)
        self.mode = mode
        self.task = task
        self.shape = shape
        self.dirr = dirr
        if download:
            self.download()
        
        self.depths = []
        self.distances_new = []
        for idx in range(len(self.points)):
            #print('{}/{}'.format(idx, len(self.points)))
            i = idx + self.start
            for j in ['', '_1', '_2', '_3']:
                tmp = torch.load('./depth{}/dist_{}'.format(j, i))
                tmp[np.isnan(tmp)] = 0.0
                tmp[tmp >= 0.7] = True
                tmp[tmp < 0.7] = False
                target = np.any(tmp.numpy(), axis=0)
                if ((target == False) and (self.count < 1500)):
                    self.distances_new.append(torch.load('./depth{}/dist_{}'.format(j, i)))
                    self.depths.append(torch.load('./depth{}/depth_{}'.format(j, i)))
                    self.count += 1
                elif (target == True):
                    self.distances_new.append(torch.load('./depth{}/dist_{}'.format(j, i)))
                    self.depths.append(torch.load('./depth{}/depth_{}'.format(j, i)))
        self.mean = 0
        if mode == 'train':
            self.distances_new = self.distances_new[int(len(self.distances_new)*0.1):]
            self.depths = self.depths[int(len(self.depths)*0.1):]
        else:
            self.distances_new = self.distances_new[:int(len(self.distances_new)*0.1)]
            self.depths = self.depths[:int(len(self.depths)*0.1)]
        #print('{}/{}'.format(len(self.depths),len(self.points)*3))
        #for dist in self.distances_new:
        #    dist[np.isnan(dist)] = 0.0
        #    self.mean += torch.sum(dist)
        #self.mean /= len(self.distances_new)

    def download(self):
        R = special_ortho_group.rvs(3)
        print(R)
        np.savetxt('{}/rotation_matrix_{}.txt'.format(self.dirr, self.mode), R)
        for idx in range(len(self.points)):
            print('saved {}/{}'.format(idx, len(self.points)))
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
                    points_3232.append([np.nan, np.nan, z])

                depth.append(z)
                distance.append(dist)

            print('points saved {}'.format(count))
            points_3232 = np.array(points_3232)
            distances_3232 = np.array(distances_3232)
            distance = np.array(distance)
            
            # normalization
            distance[distance > 1.0] = 1.0
            if np.nanmax(distance) > 0.0:
                distance /= np.nanmax(distance)
            distance -= 1
            distance = abs(distance)

            torch.save(torch.FloatTensor(points_3232), '{}/depth_{}'.format(self.dirr, idx+self.start))
            torch.save(torch.FloatTensor(distance), '{}/dist_{}'.format(self.dirr, idx+self.start))
        
    def __len__(self):
        return len(self.depths)

    def __getitem__(self, idx):
        depth = self.depths[idx][:, -1]
        torch.save(self.depths[idx], '{}_pc'.format(idx))
        # normalization
        depth = np.array(depth)
        depth -= np.nanmin(depth)
        if (np.nanmax(depth) - np.nanmin(depth)) > 0.0:
            depth /= (np.nanmax(depth) - np.nanmin(depth))
        
        depth = torch.reshape(torch.FloatTensor(depth), (1,self.shape[0],self.shape[-1]))
        mask = np.array(~np.isnan(depth))
        mask = torch.reshape(torch.FloatTensor(mask), (1,self.shape[0],self.shape[-1]))
        depth[np.isnan(depth)] = 0.0

        if self.task == 'segmentation':
            data = torch.reshape(self.distances_new[idx], (1,self.shape[0],self.shape[-1]))
            distance = torch.empty_like(data).copy_(data)
            distance[np.isnan(distance)] = 0.0
            distance *= mask
            distance += mask
            #dist1 = torch.FloatTensor(np.copy(distance))#[(distance >= 1.0) & (distance < 1.7)]
            dist1 = np.array((distance >= 1.0) & (distance < 1.7)).astype(float)
            #dist2 = torch.FloatTensor(np.copy(distance))
            dist2 = np.array(distance > 1.7).astype(float)
            target = torch.cat([torch.FloatTensor(dist1), torch.FloatTensor(dist2)], dim=0)
     
        elif self.task == 'classification':
            distance = self.distances_new[idx]
            distance[np.isnan(distance)] = 0.0
            distance[distance >= 0.7] = True
            distance[distance < 0.7] = False
            target = np.any(distance.numpy(), axis=0)*1 
            target = [target]
         
        depth = torch.cat([depth, depth, depth, mask], dim=0) 
        
        return depth, torch.FloatTensor(target)
    
class loss_(torch.nn.BCEWithLogitsLoss):
    def __name__():
        return 'bcewithlogits'

class loss_1(torch.nn.BCELoss):
    def __name__():
        return 'bce'

class ce_loss(torch.nn.CrossEntropyLoss):
    def __name__():
        return 'ce_loss'

if __name__ == '__main__':
    ENCODER = 'vgg16'
    ENCODER_WEIGHTS = None
    DEVICE = 'cuda'
    CLASSES = ['sharp', 'not_sharp']
    ACTIVATION = 'sigmoid'
    model = Unet(#Classifier(
    encoder_name=ENCODER, 
    encoder_weights=ENCODER_WEIGHTS, 
    classes=len(CLASSES), 
    activation=ACTIVATION,
    )
  
    #loss = smp.utils.losses.BCEDiceLoss(eps=1., mean)#dice_loss()#loss_(reduction='none')
    metrics = [
        smp.utils.metrics.IoUMetric(eps=1.),
        smp.utils.metrics.FscoreMetric(eps=1.),
    ]

    optimizer = torch.optim.Adam(model.parameters(), 0.001)#torch.optim.SGD(model.parameters(), 1e-2, momentum=0.7) 
    print('creating a train dataset')
    train_dataset = dataset(mode='train', download=True)
    #train_dataset = dummy_dataset(mode='train', shape=(32, 32), task='segmentation', download=False, dirr='./depth_3')
    train_loader = DataLoader(train_dataset, batch_size=32, shuffle=True)
    loss = BCEDiceLoss(eps=0.5, activation='sigmoid')

    val_dataset = dataset(mode='val')
    val_loader = DataLoader(val_dataset, batch_size=1, shuffle=False)
    train_epoch = TrainEpoch(
        model, 
        loss=loss, 
        metrics=[], 
        optimizer=optimizer,
        device=DEVICE,
        verbose=True,
    )
    val_epoch = ValidEpoch(
        model, 
        loss=loss, 
        metrics=[], 
        device=DEVICE,
        verbose=True,
    )
    PATH = './seg_model_{}_epoch_{}'

    model.load_state_dict(torch.load('./seg_model_{}_epoch_20'.format(ENCODER)))
    for i in range(0, 350):
        print('\nEpoch: {}'.format(i))
        train_logs = train_epoch.run(train_loader)
        print('train lenght', len(train_loader))
        torch.save(model.state_dict(), PATH.format(ENCODER, i))
        val_logs = val_epoch.run(val_loader)
        #for i, batch in enumerate(train_loader):
        #    model.eval()
        #    out = model.predict(batch[0].to(DEVICE))
#            mask = batch[0][:, -1, :, :].to(DEVICE)
#            print(torch.mean(torch.mul(loss(out, batch[1].to(DEVICE)), mask)))
        #    np.savetxt('{}_in_train{}.txt'.format(ENCODER, i), batch[0].detach().cpu().numpy()[0][0])
        #    np.savetxt('{}_gt_train{}.txt'.format(ENCODER, i), batch[1].detach().cpu().numpy()[0][0])
        #    np.savetxt('{}_train{}.txt'.format(ENCODER, i), out.detach().cpu().numpy()[0][0])
        #    if i > 5:
        #        break
        count = 0
        for i, batch in enumerate(val_loader):
            if i < 275:
                continue
            model.eval()
            out = model.predict(batch[0].to(DEVICE))

#            mask = batch[0][:, -1, :, :].to(DEVICE)
#            print(torch.mean(torch.mul(loss(out, batch[1].to(DEVICE)), mask)))
#            print(np.unique(batch[1].detach().cpu().numpy()[0][0]), np.unique(batch[1].detach().cpu().numpy()[0][1]))
            torch.save(batch[-1].detach().cpu().numpy()[0], '{}_in{}.txt'.format(ENCODER, i))
            np.savetxt('{}_gt_0{}.txt'.format(ENCODER, i), batch[1].detach().cpu().numpy()[0][0])
            np.savetxt('{}_gt_1{}.txt'.format(ENCODER, i), batch[1].detach().cpu().numpy()[0][1])
            np.savetxt('{}_test0{}.txt'.format(ENCODER, i), out.detach().cpu().numpy()[0][0])
            np.savetxt('{}_test1{}.txt'.format(ENCODER, i), out.detach().cpu().numpy()[0][1])
            if count < 10:
                count += 1
            else:
                break

    #for i_batch, sample_batched in enumerate(DataLoader(dataset())):
    #    print(i_batch, sample_batched)
    #    break

