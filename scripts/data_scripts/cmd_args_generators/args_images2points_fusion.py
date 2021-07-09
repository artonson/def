
from glob import glob
import os
import pathlib
import argparse


def main():
    arguments = []

    predicts_prefix = "/gpfs/data/gpfs0/3ddl/sharp_features/predictions/images2points/for_model/"
    fusion_prefix = "/gpfs/gpfs0/3ddl/sharp_features/images2points/fused"
    mapping_path = "/trinity/home/e.bogomolov/sharpf_data/images2points_patches_for_train.txt"
    output_file = "/trinity/home/e.bogomolov/sharpf_data/images2points_fusion.txt"

    with open(mapping_path, "r") as f:
        for line in f:
            line = line.strip()
            gt_path, symlink_path = line.split(" ")
            symlink_path = str(pathlib.Path(symlink_path).relative_to("/gpfs/gpfs0/3ddl/sharp_features/images2points/for_model"))
            symlink_path = symlink_path.replace("_res_whole.json", "")
            symlink_path = symlink_path.replace(".hdf5", "")

            outdir = pathlib.Path(os.path.join(fusion_prefix, symlink_path))
            try:
                outdir.mkdir(parents=True, exist_ok=False)
            except FileExistsError:
                pass

            arguments.append((
                gt_path, 
                os.path.join(predicts_prefix, symlink_path), 
                str(outdir),
            ))

    with open(output_file, "w") as f:
        for args in arguments:
            f.write(f'{args[0]} {args[1]} {args[2]} 0.9\n')



if __name__ == '__main__':
    main()
