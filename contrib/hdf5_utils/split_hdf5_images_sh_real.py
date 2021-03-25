import os
import argparse
import h5py
import numpy as np
from tqdm import tqdm


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('filename', help='hdf5 file to split')
    parser.add_argument('--label', help='label in hdf5 file to get data from (default: all)')
    parser.add_argument('--output_dir', help='output directory for splitted files')
    parser.add_argument('--output_format', default='xyz', help='output format for splitted files (default: xyz)')
    parser.add_argument('--use_normals', default=False, help='use normals')
    args = parser.parse_args()
    return args


def image_to_points(image):
    image_height, image_width = image.shape
    resolution_3d = 0.02
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
    return points, i

    
def split_hdf5(filename, label, output_dir, output_format="xyz", use_normals=False):
    
    with h5py.File(filename, 'r') as f:
        data = list(f[label])
        points = []
        nonzero = []
        for i, data_i in tqdm(enumerate(data)):
            tmp, tmp_nz = image_to_points(data_i)
            points.append(tmp)
            nonzero.append(tmp_nz)
        data = points
        if use_normals:
            norms = list(f['normals_estimation_100'])
            normals = []
            for i in range(len(data)):
#                 normals.append(np.concatenate([data[i], norms[i][nonzero[i]]], axis = -1))
                normals.append(np.concatenate([data[i], norms[i].reshape(-1,3)], axis = -1))
#             out = []
#             for i in range(len(data)):
#                 out.append(np.concatenate([data[i], normals[i]], axis = -1))
#             data = np.concatenate([data, normals], axis = -1)
#             print(data.shape)
            data = normals
    
    file_basename = os.path.splitext(os.path.basename(filename))[0]
   
    for i, data_i in tqdm(enumerate(data)):
        
#         points_i = image_to_points(data_i)
        
        output_name = "{output_dir}/{file_basename}_{label}_{i}.{output_format}".format(
            output_dir=output_dir,
            file_basename=file_basename,
            label=label,
            i=i,
            output_format=output_format
        )
        
        np.savetxt(output_name, data_i, delimiter=' ')
        
        
if __name__=='__main__':
    args = parse_args()
    output_format = args.output_format
    if output_format.startswith('.'):
        output_format = output_format[1:]
        
    if not os.path.exists(args.output_dir):
        os.makedirs(args.output_dir)
    
    split_hdf5(args.filename, args.label, args.output_dir, output_format, args.use_normals)

