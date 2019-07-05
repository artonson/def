#!/usr/bin/env bash

#SBATCH
#SBATCH
#SBATCH
#SBATCH
#SBATCH
#SBATCH
#SBATCH


#SBATCH --partition=cpu_big
#SBATCH --cpus-per-task=20
#SBATCH --ntasks=10
#SBATCH --nodes=10
#SBATCH --time 00:05:00

module load apps/singularity-3.2.0
singularity exec \
  --bind /trinity/home/a.artemov/FloorplanVectorization:/code \
  /gpfs/gpfs0/3ddl/singularity-images/artonson_vectran_latest.sif \
  python3 /code/scripts/test.py




# python3 make_data.py [filter|patch|dataset] [OPTIONS]

make_data.py filter \
    -f face_ratio \
    --face_ratio_thr 0.1 \
    -f face_aspect_ratio \
    -i /gpfs/gpfs0/3ddl/datasets/abc \
    -o /gpfs/gpfs0/3ddl/sharp_features/data \


make_data.py patch \
    -i /gpfs/gpfs0/3ddl/datasets/abc \
    --filter /gpfs/gpfs0/3ddl/sharp_features/data \
    -o /gpfs/gpfs0/3ddl/sharp_features/data/patches \
    --num_points 1024


make_data.py dataset \
    -i /gpfs/gpfs0/3ddl/datasets/abc \
    --filter /gpfs/gpfs0/3ddl/sharp_features/data \
    -o /gpfs/gpfs0/3ddl/sharp_features/data/tensors \



