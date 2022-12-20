import os
import pathlib
import numpy as np

ARGS_FILE = "/trinity/home/e.bogomolov/sharpf_data/images2points_metrics.txt"
BASE_PATH = "/gpfs/gpfs0/3ddl/sharp_features/images2points/data_v2_cvpr/images_whole/"
FUSION_PATH = "/gpfs/gpfs0/3ddl/sharp_features/images2points/fused/"
METHOD_NAME = "def"

FUSE_METHODS = [
    "crop__adv60__min",
    "crop__linreg",
    "min"
]

START_VAL = np.array([0,0,0,0,0])
RES_NOISE_METRICS = {
    "low_res_whole.json" : START_VAL,
    "med_res_whole.json" : START_VAL,
    "high_res_whole.json": START_VAL,
    "high_res_whole_0.02.json": START_VAL,
}

OUTPUT_PATH = "/trinity/home/e.bogomolov/sharpf_data/images2points_collected_metrics.txt"

def main():
    metrics_dir_paths = []
    with open(ARGS_FILE, "r") as f:
        for line in f:
            line = line.rstrip()
            path = pathlib.Path(line[0]).relative_to(BASE_PATH)
            path = pathlib.Path(FUSION_PATH) / path / METHOD_NAME
            metrics_dir_paths.append(path)

    for dir_path in metrics_dir_paths:
        for res_noise_key in RES_NOISE_METRICS:
            metrics_file = dir_path.glob(f"*{res_noise_key}*")[0]
            res_noise = metrics_file.relative_to(BASE_PATH).parts[0]
            with open(metrics_file, "r") as f:
                f.readline()
                metrics = f.readline().rstrip().split(",")
            RES_NOISE_METRICS[res_noise] += np.array(metrics + [1])
    
    with open(OUTPUT_PATH, "w") as f:
        f.write('RES_NOISE,MFPR,MRecall,MRMSE,Q95RMSE\n')
        for key, value in RES_NOISE_METRICS.items():
            f.write(f'{key},{ {*(value / value[-1])} }\n'.replace(" ",""))
            
if __name__ == "__main__":
    main()