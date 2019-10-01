#!/usr/bin/env python3

import argparse
import os
import sys
from io import BytesIO

import trimesh
from joblib import Parallel, delayed
import numpy as np
import yaml

__dir__ = os.path.normpath(
    os.path.join(
        os.path.dirname(os.path.realpath(__file__)), '..', '..')
)
sys.path[1:1] = [__dir__]

from sharpf.data.abc_data import ABCModality, ABCChunk, ABC_7Z_FILEMASK, MergedABCItem


# the function for patch generator: breadth-first search

def find_and_add(sets, desired_number_of_points, adjacency_graph):
    counter = len(sets)  # counter for number of vertices added to the patch;
    # sets is the list of vertices included to the patch
    for verts in sets:
        for vs in adjacency_graph.neighbors(verts):
            if vs not in sets:
                sets.append(vs)
                counter += 1
        #                 print(counter)
        if counter >= desired_number_of_points:
            break  # stop when the patch has more than 1024 vertices


def trimesh_load(io: BytesIO):
    """Read the mesh: since trimesh messes the indices, this has to be done manually."""

    vertices, faces = [], []

    for line in io:
        values = line.strip().split()
        if not values: continue
        if values[0] == 'v':
            vertices.append(np.array(values[1:4], dtype='float'))
        elif values[0] == 'f':
            faces.append(np.array([values[1].split('//')[0], values[2].split('//')[0], values[3].split('//')[0]],
                                  dtype='int'))

    vertices = np.array(vertices)
    faces = np.array(faces) - 1

    mesh = trimesh.base.Trimesh(vertices=vertices, faces=faces, process=False)  # create a mesh from the vertices
    return mesh


def geodesic_meshvertex_patches_from_item(item, n_points=1024):
    """Sample patches from mesh, using vertices as points
    in the point cloud,

    :param item: MergedABCItem
    :param n_points: number of points to output per patch (guaranteed)
    :return:
    """
    yml = yaml.load(item.feat)
    mesh = trimesh_load(item.obj)

    sharp_idx = []
    short_idx = []
    for i in yml['curves']:
        if len(i['vert_indices']) < 5:  # this is for filtering based on short curves:
                                        # append all the vertices which are in the curves with less than 5 vertices
            short_idx.append(np.array(i['vert_indices']) - 1)  # you need to substract 1 from vertex index,
                                                               # since it starts with 1
        if i.get('sharp') is True:
            sharp_idx.append(np.array(i['vert_indices']) - 1)  # append all the vertices which are marked as sharp
    if len(sharp_idx) > 0:
        sharp_idx = np.unique(np.concatenate(sharp_idx))
    if len(short_idx) > 0:
        short_idx = np.unique(np.concatenate(short_idx))

    surfaces = []
    for i in yml['surfaces']:
        if 'vert_indices' in i.keys():
            surfaces.append(np.array(i['face_indices']) - 1)


    sharp_indicator = np.zeros((len(mesh.vertices),))
    sharp_indicator[sharp_idx] = 1

    # and faces read previously
    adjacency_graph = mesh.vertex_adjacency_graph

    # select starting vertices to grow patches from,
    # while iterating over them use BFS to generate patches
    # TODO: why not sample densely / specify no. of patches to sample?
    for j in np.linspace(0, len(mesh.vertices), 7, dtype='int')[:-1]:
        set_of_verts = [j]
        find_and_add(sets=set_of_verts, desired_number_of_points=n_points,
                     adjacency_graph=adjacency_graph)  # BFS function

        # TODO: what does this code do?
        a = sharp_indicator[np.array(set_of_verts)[-100:]]
        b = np.isin(np.array(set_of_verts)[-100:], np.array(set_of_verts)[-100:] - 1)
        if (a[b].sum() > 3):
            #                 print('here! border!',j)
            continue

        set_of_verts = np.unique(np.array(set_of_verts))  # the resulting list of vertices in the patch
        # TODO why discard short lines?
        if np.isin(set_of_verts, short_idx).any():  # discard a patch if there are short lines
            continue

        patch_vertices = mesh.vertices[set_of_verts]
        patch_sharp = sharp_indicator[set_of_verts]
        patch_normals = mesh.vertex_normals[set_of_verts]

        if patch_sharp.sum() != 0:
            sharp_rate.append(1)
        else:
            sharp_rate.append(0)

        surfaces_numbers = []
        if patch_vertices.shape[0] >= n_points:
            # select those vertices, which are not sharp in order to use them for counting surfaces (sharp vertices
            # are counted twice, since they are on the border between two surfaces, hence they are discarded)
            appropriate_verts = set_of_verts[:n_points][
                patch_sharp[:n_points].astype(int) == 0]

            for surf_idx, surf_faces in enumerate(surfaces):
                surf_verts = np.unique(mesh.faces[surf_faces].ravel())

                if len(np.where(np.isin(appropriate_verts, surf_verts))[0]) > 0:
                    surface_ratio = sharp_indicator[np.unique(np.array(surf_verts))].sum() / \
                                    len(np.unique(np.array(surf_verts)))

                    if surface_ratio > 0.6:
                        break

                    surfaces_numbers.append(surf_idx)  # write indices of surfaces which are present in the patch

            if surface_ratio > 0.6:
                continue

            surface_rate.append(np.unique(np.array(surfaces_numbers)))
            patch_vertices = patch_vertices[:n_points]
            points.append(patch_vertices)
            patch_vertices_normalized = patch_vertices - patch_vertices.mean(axis=0)
            patch_vertices_normalized = patch_vertices_normalized / np.linalg.norm(patch_vertices_normalized,
                                                                                   ord=2, axis=1).max()
            points_normalized.append(patch_vertices_normalized)
            patch_normals = patch_normals[:n_points]
            normals.append(patch_normals)
            labels.append(patch_sharp[:n_points])


    return points, labels, normals, sharp_rate
    points = []  # for storing initial coordinates of points
    points_normalized = []  # for storing normalized coordinates of points
    labels = []  # for storing 0-1 labels for non-sharp/sharp points
    normals = []
    surface_rate = []  # for counting how many surfaces are there in the patch
    sharp_rate = []  # for indicator whether the patch contains sharp vertices at all
    times = []  # for times (useless)
    p_names = []  # for names of the patches in the format "initial_mesh_name_N", where N is the starting vertex index


def euclidean_ball_patches_from_item():
    pass


PATCH_FUNC_BY_TYPE = {
    'geodesic_meshvertex': geodesic_meshvertex_patches_from_item,
    'euclidean_ball': euclidean_ball_patches_from_item,
}


@delayed
def generate_patches(meshes_filename, feats_filename, data_slice, n_points, patch_type, noise_type, noise_amount):
    points = []  # for storing initial coordinates of points
    points_normalized = []  # for storing normalized coordinates of points
    labels = []  # for storing 0-1 labels for non-sharp/sharp points
    normals = []
    surface_rate = []  # for counting how many surfaces are there in the patch
    sharp_rate = []  # for indicator whether the patch contains sharp vertices at all
    times = []  # for times (useless)
    p_names = []  # for names of the patches in the format "initial_mesh_name_N", where N is the starting vertex index

    patch_func = PATCH_FUNC_BY_TYPE[patch_type]

    slice_start, slice_end = data_slice
    with ABCChunk([meshes_filename, feats_filename]) as data_holder:
        for item in data_holder[slice_start:slice_end]:
            patches_points, patches_labels, patches_normals, patches_has_sharpness = patch_func(item, n_points)




    times = np.array(times)
    p_names = np.array(p_names)
    points = np.array(points)
    points_normalized = np.array(points_normalized)
    labels = np.array(labels).reshape(-1, 1024, 1)
    normals = np.array(normals)
    sharp_rate = np.array(sharp_rate)
    return times, p_names, points, points_normalized, labels, sharp_rate, surface_rate, normals


def make_patches(options):
    """Filter the shapes using a number of filters, saving intermediate results."""

    # create the data source iterator
    # abc_data = ABCData(options.data_dir,
    #                    modalities=[ABCModality.FEAT.value, ABCModality.OBJ.value],
    #                    chunks=[options.chunk],
    #                    shape_representation='trimesh')

    obj_filename = os.path.join(
        options.input_dir,
        ABC_7Z_FILEMASK.format(
            chunk=options.chunk.zfill(4),
            modality=ABCModality.OBJ.value,
            version='00'
        )
    )
    feat_filename = os.path.join(
        options.input_dir,
        ABC_7Z_FILEMASK.format(
            chunk=options.chunk.zfill(4),
            modality=ABCModality.FEAT.value,
            version='00'
        )
    )
    abc_data = ABCChunk([obj_filename, feat_filename])

    processes_to_spawn = 10 * options.n_jobs
    chunk_size = len(abc_data) // processes_to_spawn
    abc_data_slices = [(start, start + chunk_size)
                       for start in range(0, len(abc_data), chunk_size)]

    # run the filtering job in parallel
    parallel = Parallel(n_jobs=options.n_jobs)
    delayed_iterable = (generate_patches(obj_filename, feat_filename,
                                         data_slice,
                                         options.num_points,
                                         options.patch_type,
                                         options.noise_type,
                                         options.noise_amount)
                        for data_slice in abc_data_slices)
    output_by_slice = parallel(delayed_iterable)

    # write out the results
    output_filename = os.path.join(options.output_dir,'{}.txt'.format(options.chunk.zfill(4)))
    with open(output_filename, 'w') as output_file:
        output_file.write('\n'.join([
            '{} {} {}'.format(pathname_by_modality['obj'], archive_pathname_by_modality['obj'], item_id)
            for pathname_by_modality, archive_pathname_by_modality, item_id, is_ok in chain(*output_by_slice) if is_ok
        ]))


def parse_args():
    parser = argparse.ArgumentParser()

    parser.add_argument('-j', '--jobs', dest='n_jobs',
                        type=int, default=4, help='CPU jobs to use in parallel [default: 4].')
    parser.add_argument('-i', '--input-dir', dest='input_dir',
                        required=True, help='input dir with ABC dataset.')
    parser.add_argument('-c', '--chunk', required=True, help='ABC chunk id to process.')
    parser.add_argument('-o', '--output-dir', dest='output_dir',
                        required=True, help='output dir.')
    parser.add_argument('-g', '--filter-config', dest='filter_config',
                        required=True, help='filter configuration file.')
    parser.add_argument('-n', '--num-points', dest='num_points', type=int,
                        required=True, help='number of points to ')

    return parser.parse_args()


if __name__ == '__main__':
    options = parse_args()
    make_patches(options)
