#!/usr/bin/env python3

import os
import argparse
import numpy as np
import trimesh.transformations as tt
import scipy.io as sio

from fuse_images import load_ground_truth

def main(options):
    pienet_args = []
    for item in os.listdir(options.input_path):
        path = f'{options.input_path}/{item}'
        fname = item.split(".")[0]
        views, view_alignments = load_ground_truth(path)
        points_views = [view.to_points() for view in views]

        all_points = np.concatenate([
            tt.transform_points(view.depth, t)
            for view, t in zip(points_views, view_alignments)
        ])

        all_distances = np.concatenate([
            view.signal for view in points_views
        ])
        
        matlab_dict = {'Training_data': []}
        matlab_dict['Training_data'].append(
            {
                'down_sample_point': all_points,
                'PC_8096_edge_points_label_bin': all_distances, 
            }
        )
        sio.savemat(f'{options.output_path}/{fname}.mat', matlab_dict)
        pienet_args.append((f'{options.output_path}/{fname}.mat', all_distances.shape[0]))

    with open(options.args_file, "w") as f:
        for fpath, pnum in pienet_args:
            f.write(f'{fpath} {pnum}\n')

def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('-ip', '--input_path', dest='input_path', type=str, required=True,
                        help='path to fused images')
    parser.add_argument('-op', '--output_path', dest='output_path', type=str, required=True,
                        help='path to store pienet point clouds')
    parser.add_argument('-af', '--args_file', dest='args_file', type=str, required=True,
                        help='where to store args for pienet inference')
    
    return parser.parse_args()


if __name__ == '__main__':
    options = parse_args()
    main(options)