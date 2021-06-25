import os
import argparse
import pathlib

def main(options):
    hdf5_dict = {
        "low_res_whole.json": {"0.0": []},
        "med_res_whole.json": {"0.0": []},
        "high_res_whole.json": {
            "0.0": [],
            "0.005": [],
            "0.02": [],
            "0.08": [],
        },
    }
    with open(options.hdf5_list) as f:
        for line in f:
            line = line.strip()
            if "low_res_whole.json" in line:
                hdf5_dict["low_res_whole.json"]["0.0"].append(line)
            elif "med_res_whole.json" in line:
                hdf5_dict["med_res_whole.json"]["0.0"].append(line)
            elif "high_res_whole.json" in line:
                hdf5_dict["high_res_whole.json"]["0.0"].append(line)
            elif "high_res_whole_0.005.json" in line:
                hdf5_dict["high_res_whole.json"]["0.005"].append(line)
            elif "high_res_whole_0.02.json" in line:
                hdf5_dict["high_res_whole.json"]["0.02"].append(line)
            elif "high_res_whole_0.08.json" in line:
                hdf5_dict["high_res_whole.json"]["0.08"].append(line)
    output_file = []
    for resol in hdf5_dict.keys():
        for noise in hdf5_dict[resol].keys():
            path = pathlib.Path(options.prefix_path) / resol / noise
            try:
                path.mkdir(parents=True, exist_ok=False)
            except FileExistsError:
                pass
            for idx, hdf5_file in enumerate(hdf5_dict[resol][noise]):
                train_file = path / f'train_{idx}.hdf5'
                os.symlink(hdf5_file, train_file)
                output_file.append((hdf5_file, train_file))
    with open(options.hdf5_output_list, "w"):
        for line in output_file:
            f.write(f'{line[0]} {line[1]}\n')
    
def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('-hl', '--hdf5_list', dest='hdf5_list',
                        type=str, required=False, help='list with hdf5 pathes')
    parser.add_argument('-hol', '--hdf5_output_list', dest='hdf5_output_list',
                        type=str, required=True, help='list with new hdf5 pathes with corresponding train_X.hdf5 path')
    parser.add_argument('-pref', '--prefix_path', dest='prefix_path',
                        type=str, required=True, help='path prefix to store symlinks')
    return parser.parse_args()

if __name__ == '__main__':
    options = parse_args()
    main(options)
