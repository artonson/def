import os

def main(options):
    
    hdf5_pathes = []
    with open(options.hdf5_list, "r") as f:
        for line in f:
            path = line.strip().split(" ")[1]
            if os.path.exists(path):
                hdf5_pathes.append(path)
    with open(options.hdf5_list_out, "w") as f:
        for path in hdf5_pathes:
            f.write(path + "\n")
    
    


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('-i', '--input_dir', dest='input_dir',
                        type=str, required=False, help='where to find all input dirs')
    parser.add_argument('-hl', '--hdf5_list', dest='hdf5_list',
                        type=str, required=True, help='file with hdf5 pathes')
    parser.add_argument('-hlo', '--hdf5_list_out', dest='hdf5_list_out',
                        type=str, required=True, help='file with hdf5 pathes succeeded')
    parser.add_argument('-od', '--output_dir', dest='output_dir',
                        type=str, required=False, help='where to store patches')
    return parser.parse_args()

if __name__ == '__main__':
    options = parse_args()
    main(options)
