import os
import re
import h5py
import argparse
import numpy as np
from tqdm import tqdm 


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('-i', '--input_dir', help='input directory with splitted files')
    parser.add_argument('-o', '--output_file', help='hdf5 file to merge files to')
    parser.add_argument('--input_format', help='input files format')
    args = parser.parse_args()
    return args


def atoi(text):
    return int(text) if text.isdigit() else text   


def string_with_numbers_comparator(filename):
    return [atoi(c) for c in re.split('(\d+)', filename)]


if __name__=='__main__':
    
    args = parse_args()

    input_files = sorted(os.listdir(args.input_dir), key=string_with_numbers_comparator)

    data = []
    for filename in tqdm(input_files):

        if not filename.endswith(args.input_format):
            continue

        file_path = "{input_dir}/{filename}".format(
            input_dir=args.input_dir,
            filename=filename
        )

        data_i = np.loadtxt(file_path)
        data.append(data_i)

    data = np.array(data)

    with h5py.File(args.output_file, 'w') as out_file:
        out_file.create_dataset('pred', shape=data.shape, data=data, dtype=np.float32)

