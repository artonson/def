#!/usr/bin/env python3

import argparse
import glob
import multiprocessing as mp
import os
import shutil

from h5py import File as HDF5File
import tqdm
import numpy as np

from sharpf.utils.py_utils.console import eprint


class BufferedHDF5Writer(object):
    def __init__(self, output_dir=None, output_files=None, n_meshes_file=float('+inf'),
                 verbose=False):
        self.output_dir = output_dir
        self.output_files = output_files
        self.file_id = 0
        self.n_meshes_file = n_meshes_file
        self.data = []
        self.verbose = verbose

    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        if self.data:
            self._flush()

    def append(self, data):
        self.data.append(data)
        if -1 != self.n_meshes_file and len(self.data) >= self.n_meshes_file:
            self._flush()
            self.file_id += 1
            self.data = []

    def _flush(self):
        """Create the next file (as indicated by self.file_id"""
        if self.output_files:
            filename = self.output_files[self.file_id]
        else:
            filename = os.path.join(self.output_dir, '{}.hdf5'.format(self.file_id))
        # print([geometry['point_cloud'].shape for geometry in self.data])
        point_cloud = np.stack([geometry['point_cloud'] for geometry in self.data])
        normals = np.stack([geometry['point_cloud'] for geometry in self.data])
        # print(self.data[0]['point_cloud'].shape)
        # print(point_cloud)
        # print(point_cloud.shape, normals.shape)
        with HDF5File(filename, mode='w') as hdf5_file:
            hdf5_file.create_dataset('data', shape=point_cloud.shape, dtype=np.float64)
            hdf5_file['data'][...] = point_cloud
            hdf5_file.create_dataset('label', shape=normals.shape, dtype=np.float64)
            hdf5_file['label'][...] = normals
        if self.verbose:
            print('Saved {}'.format(filename))


def manual_obj_reader(filename):
    from collections import namedtuple
    TriMeshEmulator = namedtuple('TriMeshEmulator', ['vertices', 'vertex_normals'])

    with open(filename) as text_file:
        vertices, vertex_normals = [], []
        for line in text_file:
            values = line.strip().split()
            if not values:
                continue
            if values[0] == 'v':
                vertices.append(values[1:4])
            elif values[0] == 'vn':
                vertex_normals.append(values[1:4])
            else:
                continue

        return TriMeshEmulator(
            vertices=np.array(vertices, dtype=np.float32),
            vertex_normals=np.array(vertex_normals, dtype=np.float32),
        )


def compute_mesh_geometry(obj_filename):
    def _compute_point_cloud(mesh, indexes=None):
        if None is indexes:
            # indexes = np.random.choice(len(mesh.vertices), 3000)
            raise NotImplementedError('sampling point clouds from mesh currently not implemented')

        point_cloud = np.array(mesh.vertices[indexes])
        # normalize point clouds
        point_cloud -= point_cloud.mean()
        point_cloud /= np.linalg.norm(point_cloud, ord=2, axis=1).max()
        # if point_cloud.shape[0] != 512:
        #     raise ValueError('Skipping 512')
        return point_cloud

    def _compute_normals(mesh, ind=None):
        normals = np.array(mesh.vertex_normals[ind])
        # check for NaNs, if true, skip mesh
        if np.any(np.isnan(normals)):
            raise ValueError('NaNs in mesh vertex normals; skipping')
        return normals

    # mesh = trimesh.load(obj_filename)
    mesh = manual_obj_reader(obj_filename)
    # indexes = np.random.choice(len(mesh.vertices), 3000)
    indexes = np.arange(len(mesh.vertices))
    point_cloud = _compute_point_cloud(mesh, indexes)
    normals = _compute_normals(mesh, indexes)
    return {
        'point_cloud': point_cloud,
        'normals': normals,
    }


def _compute_mesh_geometry_wrapper(task):
    filename, verbose = task
    if verbose:
        eprint('Processing {}'.format(filename))
    try:
        return compute_mesh_geometry(filename)
    except Exception as e:
        eprint('Could not process {}: {}'.format(filename, e))
    return None


def main(options):
    check_path = options.hdf5_output if None is not options.hdf5_output else options.hdf5_output_dir
    if not options.overwrite:
        if os.path.exists(check_path):
            eprint('\'{}\' exists and no --overwrite specified; exiting'.format(check_path))
            return
        else:
            os.makedirs(options.hdf5_output_dir)
    else:  # options.overwrite is True
        shutil.rmtree(check_path, ignore_errors=True)
        if None is not options.hdf5_output_dir:
            os.makedirs(options.hdf5_output_dir)

    input_obj_files = [options.obj_input] if None is not options.obj_input \
        else glob.glob(os.path.join(options.obj_input_dir, '*.obj'))
    output_hdf5_files = [options.hdf5_output] if None is not options.hdf5_output else None
    hdf5_writer_params = {
        'output_dir': options.hdf5_output_dir,
        'output_files': output_hdf5_files,
        'n_meshes_file': options.n_meshes_file,
        'verbose': options.verbose,
    }

    with BufferedHDF5Writer(**hdf5_writer_params) as hdf_writer:
        _data_iter = ((filename, options.verbose)
                      for filename in input_obj_files)
        pool = mp.Pool(processes=options.n_jobs)
        for mesh_geometry in tqdm.tqdm(
                pool.imap_unordered(_compute_mesh_geometry_wrapper, _data_iter),
                total=len(input_obj_files)):
            if None is not mesh_geometry:
                hdf_writer.append(mesh_geometry)
        pool.close()
        pool.join()


def parse_options():
    parser = argparse.ArgumentParser(description='Convert the OBJ-based point cloud dataset to HDF5 format.')

    parser.add_argument('-w', '--overwrite', dest='overwrite', action='store_true',
                        default=False, help='overwrite existing files.')

    group = parser.add_mutually_exclusive_group(required=True)
    group.add_argument('-i', '--input', dest='obj_input', help='OBJ input file.')
    group.add_argument('--input-dir', dest='obj_input_dir', help='directory of OBJ input files.')

    group = parser.add_mutually_exclusive_group(required=True)
    group.add_argument('-o', '--output', dest='hdf5_output', help='a single HDF5 output file.')
    group.add_argument('--output-dir', dest='hdf5_output_dir', help='directory with HDF5 output files.')

    parser.add_argument('-j', '--jobs', dest='n_jobs', default=mp.cpu_count(), type=int,
                        help='allow n_jobs jobs at once; infinite jobs with no arg.')

    parser.add_argument('-m', '--num-meshes-per-file', dest='n_meshes_file', default=-1, type=int,
                        help='how many meshes to put in a single HDF5 file (all by default).')

    parser.add_argument('-v', '--verbose', dest='verbose', action='store_true',
                        default=False, help='overwrite existing files.')

    args = parser.parse_args()

    if None is not args.obj_input_dir and args.hdf5_output:
        raise ValueError('Could not output a whole directory into a single file \'{}\''.format(args.hdf5_output))
    if None is not args.hdf5_output and os.path.isdir(args.hdf5_output):
        eprint('\'{}\' is an existing  dir specified by -o switch; '
               'selecting it as output-dir'.format(args.hdf5_output))
        args.hdf5_output_dir = args.hdf5_output
        args.hdf5_output = None
    return args


if __name__ == '__main__':
    options = parse_options()
    main(options)
