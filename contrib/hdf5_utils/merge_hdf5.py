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
    parser.add_argument('--label', help='label in hdf5 file to put data to (default: pred)')
    parser.add_argument('--varlen', default=False, help='variable length')
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
        data.append(data_i.astype(np.float32))

    data = np.array(data)

    label = args.label if None is not args.label else 'pred'
    if args.varlen:
        with h5py.File(args.output_file, 'w') as out_file:
            dataset = out_file.create_dataset(label, shape=(len(data), ), dtype=h5py.special_dtype(vlen=np.float32))
            for i in range(len(data)):
                dataset[i] = data[i].ravel()
    else:
        with h5py.File(args.output_file, 'w') as out_file:
            out_file.create_dataset(label, shape=data.shape, data=data, dtype=np.float32)