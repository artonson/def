import os
import pathlib

FUSION_PARAMETERS = "~/sharpf_data/images2points_fusion.txt"
OUTPUT_FILE = ""


def main():
    fused_paths = []
    with open(FUSION_PARAMETERS, "r") as f:
        for line in f:
            line = line.rstrip()
            path = line.split(" ")[2]
            fused_paths.append(path)

    mv_args = []
    for path in fused_paths:
        path_obj = pathlib.Path(path)
        hdf5_src = [path_obj / x for x in os.listdir(path_obj) if ".hdf5" in x]
        hdf5_dst = [path_obj / x.replace("patches", path_obj.name) for x in os.listdir(path_obj) if ".hdf5" in x]
        mv_args.extend(*zip(hdf5_src, hdf5_dst))

    for pair in mv_args:
        print(pair, "\n")
        
if __name__ == "__main__":
    main()