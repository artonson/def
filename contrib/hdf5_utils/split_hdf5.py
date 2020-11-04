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
    parser.add_argument('--use_normals', default=False, action='store_true', help='use normals')
    parser.add_argument('--reshape_nx3', default=False, action='store_true', help='if set, reshapes each item into (n, 3) array')
    args = parser.parse_args()
    return args

    
def split_hdf5(filename, label, output_dir, output_format="xyz", use_normals=False, reshape_nx3=False):
    
    with h5py.File(filename, 'r') as f:
        data = list(f[label])
        if use_normals:
            normals = list(f['normals'])
            data = np.concatenate([data, normals], axis=-1)
            print(data.shape)
            data = list(data)
    
    file_basename = os.path.splitext(os.path.basename(filename))[0]
   
    for i, data_i in tqdm(enumerate(data)):

        data_i = np.array(data_i)
        if reshape_nx3:
            data_i = data_i.reshape((-1, 3))
        
        output_name = "{output_dir}/{file_basename}_{label}_{i}.{output_format}".format(
            output_dir=output_dir,
            file_basename=file_basename,
            label=label,
            i=i,
            output_format=output_format
        )
        
        np.savetxt(output_name, data_i, delimiter=' ')
        
        
if __name__ == '__main__':
    args = parse_args()
    
    output_format = args.output_format
    if output_format.startswith('.'):
        output_format = output_format[1:]
        
    if not os.path.exists(args.output_dir):
        os.makedirs(args.output_dir)
    
    split_hdf5(args.filename, args.label, args.output_dir, output_format, args.use_normals, args.reshape_nx3)
