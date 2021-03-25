import os
import re
import h5py
import argparse
import numpy as np
from tqdm import tqdm 
import plyfile
from scipy.spatial import KDTree


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


def image_to_points(image):
    image_height, image_width = 64, 64
    resolution_3d = 0.075
    screen_aspect_ratio = 1

    rays_screen_coords = np.mgrid[0:image_height, 0:image_width].reshape(
            2, image_height * image_width).T.astype(np.float)

    rays_origins = (rays_screen_coords / np.array([[image_height, image_width]]))   # [h, w, 2], in [0, 1]
    
    factor = image_height / 2 * resolution_3d
    rays_origins[:, 0] = (-2 * rays_origins[:, 0] + 1) * factor  # to [-1, 1] + aspect transform
    rays_origins[:, 1] = (-2 * rays_origins[:, 1] + 1) * factor * screen_aspect_ratio
    
    rays_origins = np.concatenate([
        rays_origins,
        np.zeros_like(rays_origins[:, [0]])
    ], axis=1)

    i = np.where(image.ravel() != 0)[0]
    points = np.zeros((len(i), 3))
    points[:, 0] = rays_origins[i, 0]
    points[:, 1] = rays_origins[i, 1]
    points[:, 2] = image.ravel()[i]
    return points


if __name__=='__main__':
    print('start')
    
    args = parse_args()
    
    with h5py.File(args.input_file, 'r') as f:
        images = f[args.input_label][:]
       
    points = []
    for im in images:
        points.append(image_to_points(im))
    
    points = np.array(points)
    
    
#     points -= points.mean(1)[:,None,:]
#     points /= np.linalg.norm(points, axis=-1).max(-1)[:,None,None]
    
    input_files = sorted(os.listdir(args.input_dir), key=string_with_numbers_comparator)

    data = np.zeros((len(points), 4096))
    
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
        
        points_i, mean, scale = normalize_point_cloud(points[input_number])
        
        data_i = plyfile.PlyData.read(file_path)
        pred_points = np.stack([data_i.elements[0].data['x'],data_i.elements[0].data['y'],data_i.elements[0].data['z']], axis=1)
        
#         pred_scaled = pred_points - mean
#         pred_scaled /= scale
        
        prediction = np.zeros((4096))
        tree = KDTree(points_i)
        distances, vert_indices = tree.query(pred_points, k=1)
        prediction[vert_indices] = 1
        
        data[input_number] = prediction.astype(np.float32)

    data = np.array(data)

    label = args.label if None is not args.label else 'pred'
    
    print('create file')
    with h5py.File(args.output_file, 'w') as out_file:
        out_file.create_dataset(label, shape=data.shape, data=data, dtype=np.float32)
        
    print('finish')

