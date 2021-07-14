
from glob import glob
import os
import pathlib
import argparse


def main():
    arguments = []

    predicts_prefix = "/gpfs/gpfs0/3ddl/sharp_features/images2points/predictions/"
    fusion_prefix = "/gpfs/gpfs0/3ddl/sharp_features/images2points/fused"
    mapping_path = "/trinity/home/e.bogomolov/sharpf_data/images2points_patches_for_train.txt"
    output_file = "/trinity/home/e.bogomolov/sharpf_data/images2points_fusion.txt"
    symlinks_prefix = "/gpfs/gpfs0/3ddl/sharp_features/images2points/for_model"
    gt_prefix = "/gpfs/gpfs0/3ddl/sharp_features/images2points/data_v2_cvpr/images_whole/"

    with open(mapping_path, "r") as f:
        for line in f:
            line = line.strip()
            gt_path, symlink_path = line.split(" ")
            symlink_path = str(pathlib.Path(symlink_path).relative_to(symlinks_prefix))
            symlink_path = symlink_path.rstrip(".hdf5")

            outdir = str(pathlib.Path(gt_path).relative_to(gt_prefix)).rstrip("patches.hdf5")
            outdir = pathlib.Path(os.path.join(fusion_prefix, outdir))
            try:
                outdir.mkdir(parents=True, exist_ok=False)
            except FileExistsError:
                pass
            
            predict_path = os.path.join(predicts_prefix, symlink_path, "predictions")
            if os.path.exists(predict_path):
                arguments.append((
                    gt_path, 
                    predict_path, 
                    str(outdir),
                ))

    with open(output_file, "w") as f:
        for args in arguments:
            f.write(f'{args[0]} {args[1]} {args[2]} 0.9\n')


if __name__ == '__main__':
    main()
