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
from pyaabb import pyaabb
import trimesh
from scipy.spatial import KDTree
import annoy

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

# influences on amount of new sharp points added to sharp features
sharp_discretization = 0.1

# create bounding boxes for sharp lines
def create_boxes(lines):
    corners = []
    eps = 1e-8
    for l in lines:
        minc = np.array([
            min(l[0][0], l[1][0])-eps,
            min(l[0][1], l[1][1])-eps,
            min(l[0][2], l[1][2])-eps
            ])
        maxc = np.array([
            max(l[0][0], l[1][0])+eps,
            max(l[0][1], l[1][1])+eps,
            max(l[0][2], l[1][2])+eps
        ])
        corners.append([minc, maxc])

    return corners

# distance function for aabb point closest segment method
def dist_f(p, i, lines):
    p_list = []
    p_list.append(lines[i][0])
    p_list.append(lines[i][1])

    # print(lines[i][0])
    # print(lines[i][1])
    # print(p)
    start_4 = time.clock()
    a = lines[i][0]
    n = lines[i][1]-a
    l = np.linalg.norm(n)
    n /= l
    v = (p-a) - np.dot(p-a, n)*n

    d = np.linalg.norm(v)

    pl = p - v
    t = np.dot(pl-a, n)/l

    assert(np.linalg.norm(((1-t)*a + t*lines[i][1])-pl) < 1e-10)
    end_4 = time.clock()

    if t >= 0 and t <= 1:
        return d**2, pl

    start_4 = time.clock()
    d_list = np.array([np.linalg.norm(p-p_list[0]), np.linalg.norm(p-p_list[1])])
    d = np.min(d_list)
    closest = p_list[np.argmin(d_list)]
    end_4 = time.clock()

    return d**2, closest

# function to find neighbors for vertices points in mesh
def myfunc_2(a, b, adj_graph):
    neighbors = np.array(list(adj_graph.neighbors(a)))[:, None]
    b_ = np.array(b)[None, ...]
    ind = np.any(neighbors == b_, axis=1)

    return [a, neighbors[ind][0][0]]

def resample_sharp_edges(mesh, labels):
    sharp_features = list(filter(lambda a: a if a['sharp'] is True else None, labels['curves']))
    adj_graph = mesh.vertex_adjacency_graph  # neighbors

    curves = []

    for i in range(len(sharp_features)):
        # add sharp lines from mesh
        sharp_vert_indices = sharp_features[i]['vert_indices']
        mesh_vert = mesh.vertices[sharp_vert_indices]
        line_points = []
        for j in range(len(mesh_vert) - 1):
            x1, y1, z1 = mesh_vert[j]
            x2, y2, z2 = mesh_vert[j + 1]
            line_points.append([np.array([x1, y1, z1]),
                                np.array([x2, y2, z2])])
        curves.append(line_points)

        # add more points
        sharp_edges = np.array(list(map(myfunc_2, sharp_vert_indices, [sharp_vert_indices] * len(sharp_vert_indices),
                                        [adj_graph] * len(sharp_vert_indices))))

        sharp_edges = sharp_edges.reshape(sharp_edges.shape[0], 2)

        first, second = mesh.vertices[sharp_edges[:, 0]], mesh.vertices[sharp_edges[:, 1]]

        n_points_per_edge = np.linalg.norm(first - second, axis=1) / sharp_discretization
        d_points_per_edge = 1. / n_points_per_edge
        for n, v1, v2 in zip(n_points_per_edge, first, second):
            t = np.linspace(d_points_per_edge, 1 - d_points_per_edge, n)
            p = np.outer(t, v1) + np.outer(1 - t, v2)
            if len(p) > 0:
                for j in range(len(new_points)):
                    curves.append([np.array(p[j]), np.array(p[j + 1])])

    if len(curves) > 0:
        return np.concatenate(curves)

    return curves

# computing distances from points to sharp lines
def annotate(sharp_points, mesh_patch, features_patch, points, distance_scaler=1, **kwargs):
    distance_upper_bound = mesh_patch.edges_unique_length.mean() * 10
    if (len(sharp_points) == 0) or all(not curve['sharp'] for curve in features_patch['curves']):
        distances = np.ones_like(points[:, 0]) * distance_upper_bound
        return distances

    if np.linalg.norm(sharp_points.reshape(-1, 3) - mesh_patch.vertices.mean(0), axis=1).min() / np.linalg.norm(
            mesh_patch.vertices - mesh_patch.vertices.mean(0), axis=1).max() >= 0.9:
        distances = np.ones_like(points[:, 0]) * distance_upper_bound
        return distances

    # tree = KDTree(sharp_points, leafsize=100)
    # distances, vert_indices = tree.query(points, distance_upper_bound=distance_upper_bound * distance_scaler)

    # annoy_index = annoy.AnnoyIndex(sharp_points.shape[1], metric='euclidean')
    # for i, x in enumerate(sharp_points):
    #     annoy_index.add_item(i, x.tolist())
    # annoy_index.build(10)
    # distances = []
    # for i, x in enumerate(points):
    #     distances.append(annoy_index.get_nns_by_vector(x, 1, 100, include_distances=True)[1])
    # distances = np.array(distances).reshape(-1)#, vert_indeces = np.array(distances_and_vertindeces)[:, 0].reshape(-1, 1), np.array(distances_and_vertindeces)[:, 1]

    corners = create_boxes(sharp_points)
    method = pyaabb.AABB()
    method.build(corners)
    dist_lambda = lambda p, i: dist_f(p, i, sharp_points)
    distances = np.zeros(points.shape[0])

    for i, p in enumerate(points):
        nn = method.nearest_point(p.astype('float32'), dist_lambda) # nearest_facet, nearest_point, sq_dist
        distances[i] = (nn[-1] / np.sqrt(nn[-1])) # distance returned by nearest_point is squared

    distances /= distance_scaler
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

    return distances

def ray_cast_mesh(mesh, labels, ortho_scale=2.0, image_sz=(512, 512), num_angles=4, z_shift=-3):
    image_height, image_width = image_sz

    angles = np.linspace(0, np.pi, num_angles)
    mesh_grid = np.mgrid[0:image_height, 0:image_width].reshape(2, image_height * image_width).T  # screen coordinates

    # to [0, 1]
    aspect = image_width / image_height
    rays_origins = (mesh_grid / np.array([[image_height, image_width]]))

    # to [-1, 1] + aspect transform
    rays_origins[:, 0] = (-2 * rays_origins[:, 0] + 1) * ortho_scale / 2
    rays_origins[:, 1] = (-2 * rays_origins[:, 1] + 1) * ortho_scale / 2 * aspect

    rays_origins = np.concatenate([rays_origins, np.zeros_like(rays_origins[:, [0]])], axis=1)
    ray_directions = np.tile(np.array([0, 0, -1]), (rays_origins.shape[0], 1))
    curves = resample_sharp_edges(mesh, labels)
    poses = []
    renders = []
    distances = []
    annotations = []
    sharp_curves = []
    pcs = []

    i = 0
    for (x, y, z) in itertools.combinations_with_replacement(angles, 3):
        i += 1

        transform = view.from_pose([x, y, z], [0, 0, 0])

        # transform mesh and get normalized one
        mesh_ = mesh.copy()
        mesh_.apply_transform(transform)
        mesh_.apply_translation([0, 0, z_shift])
        mesh_normalized = mesh_.copy()
        mean = mesh_normalized.vertices.mean()
        mesh_normalized.vertices -= mean
        scale = mesh_normalized.vertices.std()  # np.linalg.norm(mesh_normalized.vertices, axis=1).max()
        mesh_normalized.vertices /= scale
        mesh_normalized.vertices *= 10

        # rotate sharp lines
        curve_points = curves.reshape(-1, 3)
        curve_points = curve_points.dot(transform.numpy().T[:3, :3]) + [0, 0, z_shift]
        curve_points -= mean
        curve_points /= scale
        curve_points *= 10
        curves_trans = curve_points.reshape(-1, 2, 3)

        # get point cloud
        ray_mesh = RayMeshIntersector(mesh_)
        index_triangles, index_ray, point_cloud = ray_mesh.intersects_id(ray_origins=rays_origins,
                                                                         ray_directions=ray_directions, \
                                                                         multiple_hits=False, return_locations=True)
        point_cloud -= mean
        point_cloud /= scale
        point_cloud *= 10

        start_4 = time.clock()
        annotation = annotate(curves_trans, mesh_normalized, labels, point_cloud, distance_upper_bound=3.0)
        end_4 = time.clock()
        print('annotation time', end_4 - start_4)

        render = np.zeros((image_height, image_width))
        render[mesh_grid[index_ray][:, 0], mesh_grid[index_ray][:, 1]] = point_cloud[:, 2]

        dist = np.zeros((image_height, image_width))
        dist[mesh_grid[index_ray][:, 0], mesh_grid[index_ray][:, 1]] = annotation

        renders.append(render)
        distances.append(dist)
        annotations.append(annotation)
        poses.append({'rotation': (x, y, z), 'location': (0, 0, z_shift)})
        #         pcs.append(point_cloud)
        #         sharp_curves.append(curves_trans)
        del transform, mesh_, mesh_normalized

    return renders, distances, annotations  # , pcs, sharp_curves

# ABC dataset
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
            depths_id = depths_id[:int(len(depths_id)*0.1)]
            dists_id = dists_id[:int(len(dists_id) * 0.1)]
        elif mode == 'val':
            depths_id = depths_id[int(len(depths_id) * 0.99):]
            dists_id = dists_id[int(len(dists_id) * 0.99):]

        for i in range(len(depths_id)):
            print('{}/{}'.format(i, len(depths_id)))
            self.depths.append(torch.load('/home/gbobrovskih/abc/depths/{}'.format(depths_id[i])))
            self.dists.append(torch.load('/home/gbobrovskih/abc/dists/{}'.format(dists_id[i])))

    def download(self, dir_in='/home/gbobrovskih/abc', dir_out='/home/gbobrovskih/abc/'):
        # sys.setrecursionlimit(100000) for annoy
        shapes_ids = os.listdir(dir_in)
          
        data_path = Path('/home/gbobrovskih/abc')
        curr_id = 0
        for shape_id in tqdm(shapes_ids):
            print('id {}'.format(shape_id))
            print('{}/{}'.format(curr_id, len(shapes_ids)))
            curr_id += 1

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
            # edges = set()
            # for face in faces:
            #     for edge in itertools.combinations(face, 2):
            #         edges.add(edge)

            mesh = trimesh.base.Trimesh(vertices=vertices,
                                faces=faces,
                                process=False,
                                validate=False,)

            #shapes = {'vertices': vertices, 'faces': faces, 'labels': labels, 'edges': edges, 'mesh': mesh}
            start_1 = time.clock()
            renders, distances, poses = ray_cast_mesh(mesh, labels, ortho_scale=2.0)
            end_1 = time.clock()
            print('ray_cast_mesh ', end_1 - start_1)
            for j in range(len(renders)):
                torch.save(torch.FloatTensor(renders[j]), dir_out + 'depths/render_id{}_{}'.format(j, shape_id))
                torch.save(torch.FloatTensor(distances[j]), dir_out + 'dists/distance_id{}_{}'.format(j, shape_id))

    def normalize(self, data):
        norm_data = np.copy(data)
        norm_data -= np.mean(norm_data)
        norm_data /= np.linalg.norm(norm_data, axis=1).max()
        return norm_data

    def __len__(self):
        return len(self.dists)

    def __getitem__(self, idx):
        depth = self.normalize(self.depths[idx])
        mask = torch.FloatTensor(np.array(depth != 0).astype(float).reshape(1, depth.shape[0], depth.shape[1]))
        depth = torch.FloatTensor(depth.reshape(1, depth.shape[0], depth.shape[1]))
        depth = torch.cat([depth, depth, depth, mask], dim=0)
        dist = self.normalize(self.dists[idx])
        return depth, torch.FloatTensor(dist)


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


class loss_logits(torch.nn.BCEWithLogitsLoss):
    def __name__():
        return 'bcewithlogits'

class loss_bce(torch.nn.BCELoss):
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

    val_dataset = dataset(mode='val', download=True)
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

    #model.load_state_dict(torch.load('./seg_model_{}_epoch_20'.format(ENCODER)))
    for i in range(0, 350):

        for i, batch in enumerate(val_loader):
            if i < 275:
                continue
            model.eval()
            print(batch[0].shape)
            out = model.predict(batch[0].to(DEVICE))


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

