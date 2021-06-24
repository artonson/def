import os
import argparse

def main(options):
    hdf5_pathes = []
    with open(options.hdf5_list, "r") as f:
        for line in f:
            path = line.strip().split(" ")[1]
            if os.path.exists(path):
                hdf5_pathes.append(path)
    with open(options.hdf5_list_out, "w") as f:
        for path in hdf5_pathes:
            path_out = path.split("/")
            path_out[-1] = "train_0.hdf5"
            f.write(path + " " + "/".join(path_out) + "\n")

def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('-hl', '--hdf5_list', dest='hdf5_list',
                        type=str, required=True, help='file with hdf6 pathes')
    parser.add_argument('-hlo', '--hdf5_list_out', dest='hdf5_list_out',
                        type=str, required=True, help='file with hdf5 pathes succeeded')
    return parser.parse_args()