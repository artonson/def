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
    for path in fused_paths:
        path_obj = pathlib,Path(path)
        print(path_obj.name)
        print([x for x in os.listdir(path_obj) if ".hdf5" in x])
        print("\n")

if __name__ == "__main__":
    main()