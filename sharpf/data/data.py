#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
@Author: Yue Wang
@Contact: yuewangx@mit.edu
@File: data.py
@Time: 2018/10/13 6:21 PM
"""

import os
import glob
import h5py
import numpy as np
from scipy.spatial import KDTree
from torch.utils.data import Dataset


def load_data(data_path, partition, data_label, target_label):
    all_data = []
    all_label = []
    for h5_name in glob.glob(os.path.join(data_path, partition, '*.hdf5')):
        f = h5py.File(h5_name, 'r')
        data = f[data_label][:].astype('float32')
        if target_label == 'directions':
            label = np.concatenate(
                (f['distances'][:].astype('float32')[:, :, None], f[target_label][:].astype('float32'), ), 
                axis=-1
            )
        else:
            label = f[target_label][:].astype('float32')
        f.close()
        data = data - data.mean(1)[:, None, :]

        all_data.append(data)
        all_label.append(label)
    all_data = np.concatenate(all_data, axis=0)
    all_label = np.concatenate(all_label, axis=0)

    patch_nan_inds = np.unique(np.where(np.isnan(all_label))[0])
    for i in patch_nan_inds:
        nan_inds = np.unique(np.where(np.isnan(all_label[i]))[0])
        non_nan_inds = np.setdiff1d(np.arange(len(all_label[i])), nan_inds)
        tree = KDTree(all_data[i, non_nan_inds], leafsize=100)
        _, non_nan_indices = tree.query(all_data[i, nan_inds])
        all_label[i, nan_inds] = all_label[i][non_nan_inds][non_nan_indices]
    return all_data, all_label


def rotate_pointcloud(pointcloud, label=None):
    rotation_angle = np.random.uniform() * 2 * np.pi
    cosval = np.cos(rotation_angle)
    sinval = np.sin(rotation_angle)
    rotation_matrix = np.array([[cosval, 0, sinval],
                                [0, 1, 0],
                                [-sinval, 0, cosval]])
    rotated_data = np.dot(pointcloud, rotation_matrix)
    if label is not None:
        rotated_label = np.dot(label, rotation_matrix)
        return rotated_data, rotated_label
    else:
        return rotated_data


def scale_pointcloud(pointcloud):
    scale = np.random.uniform(0.5, 2)
    scaled_data = pointcloud * scale
    return scaled_data


class ABCData(Dataset):
    def __init__(self, data_path, partition, data_label, target_label):
        self.data, self.label = load_data(data_path, partition, data_label, target_label)
        self.partition = partition
        self.target_label = target_label
        print("{} out of {} items are sharp".format(
            np.sum(self.label), len(self.label)
        ))

    def __getitem__(self, item):
        pointcloud = self.data[item]
        label = self.label[item]

        if np.isscalar(label):
            points_labels = pointcloud
            #if self.partition == 'train':
            #    points_labels = rotate_pointcloud(points_labels)
            #    points_labels = scale_pointcloud(points_labels)
            #    np.random.shuffle(points_labels)
            #print (points_labels, label)

            return points_labels, label

        else:
            points_labels = np.hstack([pointcloud, np.atleast_2d(label).reshape(-1, 1)])

            if self.partition == 'train':
                if self.target_label == 'directions':
                    points_labels[:,:3], points_labels[:,4:] = rotate_pointcloud(points_labels[:,:3], points_labels[:,4:])
                else:
                    points_labels[:,:3] = rotate_pointcloud(points_labels[:,:3])
                points_labels[:,:3] = scale_pointcloud(points_labels[:,:3])
                np.random.shuffle(points_labels)

            return points_labels[:, :3], points_labels[:, 3:]

    def __len__(self):
        return self.data.shape[0]


