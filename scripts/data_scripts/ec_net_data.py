#!/usr/bin/env python3

import argparse
from itertools import chain
import json
import os
import sys
import numpy as np
import h5py
import yaml
import trimesh
from joblib import Parallel, delayed

__dir__ = os.path.normpath(
    os.path.join(
        os.path.dirname(os.path.realpath(__file__)), '..', '..')
)
sys.path[1:1] = [__dir__]

from sharpf.utils.abc_utils.abc.abc_data import ABCModality, ABCChunk, ABC_7Z_FILEMASK, MergedABCItem
from sharpf.utils.abc_utils.mesh.io import trimesh_load
from sharpf.utils.abc_utils.abc.feature_utils import get_curves_extents


def find_intersec_vec(x, y):
    y_all = np.concatenate(y)
    y_all_in = np.isin(y_all, x)
    splits = np.cumsum([0] + [len(lst) for lst in y])
    y_in = np.logical_or.reduceat(y_all_in, splits[:-1])
    return [True if isin else False for lst, isin in zip(y, y_in)]

def scale_mesh(mesh, features, shape_fabrication_extent=10, resolution_3d=0.125,
               short_curve_quantile=0.25, n_points_per_short_curve=8):
    # compute standard size spatial extent
    mesh_extent = np.max(mesh.bounding_box.extents)
    mesh = mesh.apply_scale(shape_fabrication_extent / mesh_extent)

    # compute lengths of curves
    sharp_curves_lengths = get_curves_extents(mesh, features)

    least_len = np.quantile(sharp_curves_lengths, short_curve_quantile)
    least_len_mm = resolution_3d * n_points_per_short_curve

    mesh = mesh.apply_scale(least_len_mm / least_len)

    return mesh

# @delayed
def update_data(index, item_ids, orig_vert_indices, orig_face_indexes, out_dir, a_dir):
    edge_points = np.zeros((2000, 3))
    edge = np.zeros((120, 6))
    face = np.zeros((800, 9))
    current_chunk_id = item_ids[index].decode().split('_')[0][:4]
    with ABCChunk([a_dir + 'abc_{chunk}_obj_v00.7z'.format(chunk=current_chunk_id),
                   a_dir + 'abc_{chunk}_feat_v00.7z'.format(chunk=current_chunk_id)]) as chunk:
            
            item = item_ids[index]
            patch_verts = orig_vert_indices[index]
            patch_faces = orig_face_indexes[index]
            file_in_chunk = chunk.get(item.decode())
            feat = yaml.load(file_in_chunk.feat, Loader=yaml.Loader)
            mesh = trimesh_load(file_in_chunk.obj)
            
            mesh = scale_mesh(mesh, feat)
            
            submesh = mesh.submesh(patch_faces.reshape(1,-1))[0]
            candidate_faces = submesh.vertices[submesh.faces].reshape(-1,9)
            
            if len(candidate_faces) >= 800:
                face = candidate_faces[:800]
            else:
                face = np.tile(candidate_faces, (800 // len(candidate_faces) + 1,1))[:800]

            curves_verts = []
            for curve in feat['curves']:
                if curve['sharp']:
                    curves_verts.append(curve['vert_indices'])
            
            mask_curves = find_intersec_vec(patch_verts, curves_verts)
            
            if np.sum(mask_curves) == 0:
                max_coord = submesh.vertices.max() * 100
                placeholder_edge = [max_coord, max_coord, max_coord,
                                    max_coord * 2, max_coord * 2, max_coord * 2]
                edge = np.tile(placeholder_edge, (120,1))
                placeholder_points = np.linspace(max_coord,max_coord * 2,2000).reshape(-1,1)
                edge_points = np.tile(placeholder_points, (1,3))
            else:
#                 present_curves = np.unique(np.concatenate(np.array(curves_verts, dtype='object')[mask_curves])-1)
                present_curves = np.array(curves_verts, dtype='object')[mask_curves]
                candidate_edge_inds = []
                for present_curve in present_curves:
                    for j in range(len(present_curve)-1):
                        candidate_edge_inds.append([present_curve[j], present_curve[j+1]])
                        
                candidate_edge_inds = np.array(candidate_edge_inds)
#                 candidate_edges = mesh.vertices[mesh.edges[np.isin(mesh.edges, present_curves).sum(-1) == 2]].reshape(-1,6)
                candidate_edges = mesh.vertices[candidate_edge_inds].reshape(-1,6)
                if len(candidate_edges) >= 120:
                    edge = candidate_edges[:120]
                else:
                    edge = np.tile(candidate_edges, (120 // len(candidate_edges) + 1,1))[:120]
                
                points = []
                for t in np.linspace(0, 1, 2000 // len(candidate_edges) + 1):
                    points.append(candidate_edges[:,:3] * t + candidate_edges[:,3:] * (1 - t))
                edge_points = np.array(points).reshape(-1,3)[:2000]
            
            np.save(out_dir + 'edge_points_{index}.npy'.format(index=index), edge_points)
            np.save(out_dir + 'edge_{index}.npy'.format(index=index), edge)
            np.save(out_dir + 'face_{index}.npy'.format(index=index), face)


def process_meshes(options):
    """Filter the shapes using a number of filters, saving intermediate results."""
    with h5py.File(options.input_dir+'.hdf5', 'r') as f:
        item_ids = f['item_id'][:]
        orig_vert_indices = f['orig_vert_indices'][:]
        orig_face_indexes = f['orig_face_indexes'][:]
    
#     mask = np.load(options.input_dir+'.npy')
    indices = np.arange(len(item_ids))

    parallel = Parallel(n_jobs=options.n_jobs, backend='multiprocessing')
    delayed_iterable = (delayed(update_data)(index, item_ids, orig_vert_indices, orig_face_indexes, options.output_dir, options.abc_dir)
                        for index in indices)
    parallel(delayed_iterable)


def parse_args():
    parser = argparse.ArgumentParser()

    parser.add_argument('-j', '--jobs', dest='n_jobs',
                        type=int, default=40, help='CPU jobs to use in parallel [default: 40].')
    parser.add_argument('-i', '--input-dir', dest='input_dir',
                        required=True, help='input dir with ABC dataset.')
    parser.add_argument('-a', '--abc-dir', dest='abc_dir',
                        required=True, help='input dir with ABC dataset.')
    parser.add_argument('-o', '--output-dir', dest='output_dir',
                        required=True, help='output dir.')

    return parser.parse_args()


if __name__ == '__main__':
    options = parse_args()
    process_meshes(options)
