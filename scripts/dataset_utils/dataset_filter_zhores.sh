#!/usr/bin/env bash

#SBATCH
#SBATCH
#SBATCH
#SBATCH
#SBATCH
#SBATCH
#SBATCH

CPUS_PER_TASK=20

#SBATCH -J sharpf_dataset_filter
#SBATCH --partition=cpu_big
#SBATCH --cpus-per-task=${CPUS_PER_TASK}
#SBATCH --ntasks=1
#SBATCH --nodes=10
#SBATCH --time 00:05:00
#SBATCH --array=0-99

# ${SLURM_ARRAY_TASK_ID} is set by SLURM


${SLURM_ARRAY_TASK_ID} \


module load apps/singularity-3.2.0
singularity exec \
  --bind /trinity/home/a.artemov/FloorplanVectorization:/code \
  /gpfs/gpfs0/3ddl/singularity-images/artonson_vectran_latest.sif \
  python3 /code/scripts/dataset_utils/dataset_filter.py \
    -i /gpfs/gpfs0/3ddl/datasets/abc \
    -o /gpfs/gpfs0/3ddl/sharp_features/data \
    -j ${CPUS_PER_TASK}



