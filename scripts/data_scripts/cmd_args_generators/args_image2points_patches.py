from glob import glob
import os
import pathlib
import argparse


def main(options):
    all_gt = glob(options.input_dir_pattern, recursive=True)
    out_path = pathlib.Path(options.output_dir)
    with open(options.output_file, 'w') as f:
        for file_path in all_gt:
            if "high_res" in str(file_path):
                sht = 0.02
            elif "med_res" in str(file_path):
                sht = 0.05
            elif "low_res" in str(file_path):
                sht = 0.125
            else:
                sht = 0.125
            plib_path = pathlib.Path(file_path)
            inp_file_dir = plib_path.parents[0]
            rel_path = plib_path.parents[1].relative_to(options.input_rel_path)
            patches_location = (out_path / rel_path)
            try:
                patches_location.mkdir(parents=True, exist_ok=False)
            except FileExistsError:
                pass
            patches_filepath = f'{patches_location}/patches.hdf5'
            f.write(f'{inp_file_dir} {patches_filepath} {sht}\n')

def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('-i', '--input_dir_pattern', dest='input_dir_pattern',
                        type=str, required=True, help='how to find all input dirs')
    parser.add_argument('-irel', '--input_rel_path', dest='input_rel_path',
                        type=str, required=True, help='all inputs common prefix')
    parser.add_argument('-od', '--output_dir', dest='output_dir',
                        type=str, required=True, help='where to store patches')
    parser.add_argument('-of', '--output_file', dest='output_file',
                        type=str, required=True, help='arguments file')
    return parser.parse_args()

if __name__ == '__main__':
    options = parse_args()
    main(options)
