import os
import argparse
import numpy as np
import h5py

def main(options):
    
    g = h5py.File(options.input_file, "r")
    num_patches = g['points'].shape[0]
    print("number of patches ", num_patches)

    f = h5py.File(options.output_file, "w")
    dt = h5py.vlen_dtype(np.dtype('float32'))
    points = f.create_dataset('points', (n_patches,), dtype=dt)
    distances = f.create_dataset('distances', (n_patches,), dtype=dt)
    for idx, item in enumerate(g['points']):
        points[idx] = item
    for idx, item in enumerate(g['distances']):
        distances[idx] = item
    
    for key in g.keys():
        if key == "points" or key == "distances":
            continue
        else:
            dset = f.create_dataset('key', data=g[key])
    
def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('-i', '--input_file', dest='input_file',
                        type=str, required=False, help='input hdf5 file')
    parser.add_argument('-o', '--output_file', dest='output_file',
                        type=str, required=True, help='output hdf5 file')
    return parser.parse_args()

if __name__ == '__main__':
    options = parse_args()
    main(options)
