#!/usr/bin/env python3

import argparse
import json
import os
import sys
from tqdm import tqdm
from multiprocessing import Manager, Process, Pool
import itertools


import igl
import numpy as np
from scipy import io as sio

__dir__ = os.path.normpath(
    os.path.join(
        os.path.dirname(os.path.realpath(__file__)), '..', '..', '..', '..'))

from sharpf.utils.py_utils.os import change_ext

sys.path[1:1] = [__dir__]

from sharpf.utils.abc_utils.abc.abc_data import ABCModality, ABCChunk, ABC_7Z_FILEMASK
import sharpf.utils.abc_utils.abc.feature_utils as feat
from sharpf.utils.abc_utils.mesh.io import trimesh_load

# iterate over items in chunk
#   normalize mesh [-0.5, 0.5]
#   for each pc of pienet do
#       normalize pc (?)
#       for each point in pc do
#           compute distance from point to mesh
#   compute max point to mesh distance
#   decide on mesh_id is pc_id

# program takes 
#   matlab folder as argument
#   chunks folder as argument
#   returns ('matfile', 'id', 'chunkfile', 'id')

def load_chunk(args):
    abc_item, item_id, items_list = args[0], args[1], args[2]
    mesh, _, _ = trimesh_load(abc_item.obj)
    items_list.append((mesh, item_id))
   

def compute_distances(args):
    pc_id, points, abc_items = args[0], args[1], args[2]
    distances = []
    for mesh, item_id in abc_items:

        shape_extent = 1.0  # max(x) - min(x), span from -0.5 to +0.5 cube, so extent = 2
        # compute standard size spatial extent
        mesh_extent = np.max(mesh.bounding_box.extents)
        mesh = mesh.apply_scale(shape_extent / mesh_extent)

        mesh_extent = mesh.bounding_box.extents
        mesh_bounds = mesh.bounding_box.bounds
        translation = mesh_bounds[0] + mesh_extent / 2
        mesh.apply_translation(-translation)

        distance_sq, mesh_face_indexes, _ = igl.point_mesh_squared_distance(
            points,
            mesh.vertices,
            mesh.faces
        )
        if distance_sq.max() < 1e-4:
            distances.append((distance_sq.max(), pc_id, item_id, ))
    return distances


def find_closest(options):
    pienet_pclouds = {}
    for mat_file in ['101.mat', '102.mat']:
        data = sio.loadmat(os.path.join(options.pienet_data_dir, mat_file))
        pienet_pclouds[mat_file] = [data["Training_data"][x,0]['down_sample_point'][0,0] 
                                        for x in range(data["Training_data"].shape[0])]
    out_list = []


    with open(options.output_filename, "w") as f:
        f.write(";".join(["matfile", "pc_id", "chunk_num", "mesh_id", "dist"]) + "\n")

    manager = Manager()
    processes = []
    for chunk in options.chunk_ids:
        obj_filename = os.path.join(
            options.data_dir,
            ABC_7Z_FILEMASK.format(
                chunk=chunk.zfill(4),
                modality=ABCModality.OBJ.value,
                version='00'))
        feat_filename = os.path.join(
            options.data_dir,
            ABC_7Z_FILEMASK.format(
                chunk=chunk.zfill(4),
                modality=ABCModality.FEAT.value,
                version='00'))
        abc_items = manager.list()
        with ABCChunk([obj_filename, feat_filename]) as abc_data:
            pool = Pool(40)
            N = len(abc_data)
            results = list(tqdm(pool.imap(
                load_chunk, 
                zip(itertools.islice(abc_data, N), range(N), itertools.repeat(abc_items))
            ), total=N))
            pool.close()
            pool.join()
        
        # THE TEST #
        other_items = []
        with ABCChunk([obj_filename, feat_filename]) as abc_data:
            N = 40
            for args in tqdm(zip(itertools.islice(abc_data, N), range(N), itertools.repeat(other_items)), total=N):
                load_chunk(args)
      
        import pdb; pdb.set_trace()
        # THE TEST #
        with open(options.output_filename, "a") as f:
            for matfile, clouds in pienet_pclouds.items():
                pool_args = []
                for pc_id, points in enumerate(clouds):
                    pool_args.append((pc_id, points, abc_items))

                other_pool = Pool(40)
                results = list(tqdm(other_pool.imap(compute_distances, pool_args), total=len(pool_args)))
                other_pool.close()
                other_pool.join()
                
                for item in results:
                    for line in sorted(item, key=lambda tup: tup[0])[:5]:
                        f.write(";".join([matfile, str(line[1]), str(chunk), str(line[2]), str(line[0])]) + "\n")
            


def parse_args():
    parser = argparse.ArgumentParser(
        formatter_class=argparse.ArgumentDefaultsHelpFormatter)

    parser.add_argument('-d', '--data-dir', dest='data_dir',
                        required=True, help='input dir with ABC dataset.')
    parser.add_argument('-ch_ids', '--chunk-ids', nargs='+', dest='chunk_ids',
                        required=True, help='chunk identifiers')
    parser.add_argument('-md', '--pienet-data-dir', dest='pienet_data_dir',
                        required=True, help='pienet data dir')
    parser.add_argument('-o', '--output-filename', dest='output_filename',
                        required=True, help='output filename.')

    return parser.parse_args()


if __name__ == '__main__':
    options = parse_args()
    find_closest(options)
