import os
import re
import h5py
import argparse
import numpy as np
from tqdm import tqdm 
import plyfile
from scipy.spatial import KDTree
import sys
sys.setrecursionlimit(10000)


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('-i', '--input_dir', help='input directory with splitted files')
    parser.add_argument('--input_file', help='input file with reference')
    parser.add_argument('--input_label', help='input label with reference')
    parser.add_argument('-o', '--output_file', help='hdf5 file to merge files to')
    parser.add_argument('--input_format', help='input files format')
    parser.add_argument('--label', help='label in hdf5 file to put data to (default: pred)')
    args = parser.parse_args()
    return args


def atoi(text):
    return int(text) if text.isdigit() else text   


def string_with_numbers_comparator(filename):
    return [atoi(c) for c in re.split('(\d+)', filename)]


def normalize_point_cloud(_input):
    if len(_input.shape)==2:
        axis = 0
    elif len(_input.shape)==3:
        axis = 1
    centroid = np.mean(_input, axis=axis, keepdims=True)
    _input = _input - centroid
    furthest_distance = np.amax(np.sqrt(np.sum(_input ** 2, axis=-1)),axis=axis,keepdims=True)
    _input = _input / furthest_distance
    return _input, centroid, furthest_distance


if __name__=='__main__':
    print('start')
    
    args = parse_args()
    
    with h5py.File(args.input_file, 'r') as f:
        points = f[args.input_label][:]
        
#     points -= points.mean(1)[:,None,:]
#     points /= np.linalg.norm(points, axis=-1).max(-1)[:,None,None]
    
    input_files = sorted(os.listdir(args.input_dir), key=string_with_numbers_comparator)

    data = np.zeros((len(points), ), dtype="object")
    
    print('start loop')

    for filename in tqdm(input_files):

        if not filename.endswith(args.input_format):
            continue

        file_path = "{input_dir}/{filename}".format(
            input_dir=args.input_dir,
            filename=filename
        )
        
        
#         points_ups = np.loadtxt(file_path[:-8]+'.xyz')
#         mean_ups = points_ups.mean(0)[None,:]
#         scale_ups = np.linalg.norm(points_ups, axis=-1).max(-1)
        
        input_number = int(filename.split('_')[-2])
        data_i = plyfile.PlyData.read(file_path)
        pred_points = np.stack([data_i.elements[0].data['x'],data_i.elements[0].data['y'],data_i.elements[0].data['z']], axis=1)
        
#         pred_scaled = pred_points - mean_ups
#         pred_scaled /= scale_ups

        points_i, _, _ = normalize_point_cloud(points[input_number].reshape(-1,3))
        
        prediction = np.zeros(len(points_i))
        tree = KDTree(points_i)
        distances, vert_indices = tree.query(pred_points, k=1)
        prediction[vert_indices] = 1
        
        data[input_number] = prediction.astype(np.float32)

    data = np.array(data)

    label = args.label if None is not args.label else 'pred'
    
    print('create file')
    with h5py.File(args.output_file, 'w') as out_file:
        dataset = out_file.create_dataset(label, shape=(len(data), ), dtype=h5py.special_dtype(vlen=np.float32))
        for i in range(len(data)):
            dataset[i] = data[i]
#         out_file.create_dataset(label, shape=data.shape, data=data, dtype=np.float32)
        
    print('finish')

