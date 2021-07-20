import os
import pathlib

FUSION_PARAMETERS = "/trinity/home/e.bogomolov/sharpf_data/images2points_fusion.txt"
OUTPUT_FILE = "/trinity/home/e.bogomolov/sharpf_data/images2points_metrics.txt"


def move_predicts():
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
        mv_args.extend(zip(hdf5_src, hdf5_dst))

    for pair in mv_args:
        if not os.path.exists(str(pair[1])):
            os.symlink(str(pair[0]), str(pair[1]))


def move_input():
    fused_paths = []
    with open(FUSION_PARAMETERS, "r") as f:
        for line in f:
            line = line.rstrip()
            path = line.split(" ")[0]
            fused_paths.append(path)

    mv_args = []
    for path in fused_paths:
        path_obj = pathlib.Path(path)
        hdf5_src = path_obj
        hdf5_dst = str(path_obj).replace("/patches", "")
        mv_args.append(hdf5_src, hdf5_dst)

    for pair in mv_args:
        if not os.path.exists(str(pair[1])):
            os.symlink(str(pair[0]), str(pair[1]))



def main():
    fused_paths = []
    with open(FUSION_PARAMETERS, "r") as f:
        for line in f:
            line = line.rstrip()
            input = line.split(" ")[0]
            predicts = line.split(" ")[2]
            fused_paths.append((input, predicts))
    
    output_file = []
    for input, predicts in fused_paths:
        hdf5_src = [x for x in os.listdir(predicts) if ".hdf5" in x]
        if "high_res" in str(hdf5_src):
            res = 0.02
        elif "med_res" in str(hdf5_src):
            res = 0.05
        elif "low_res" in str(hdf5_src):
            res = 0.125
        else:
            res = 0.125
        if hdf5_src:
            output_file.append((input.replace("/patches", ""), res))

    with open(OUTPUT_FILE, "w") as f:
        for line in output_file:
            f.write(f'{line[0]} {line[1]}\n')
    
if __name__ == "__main__":
    move_input()
    move_predicts()
    create_args()