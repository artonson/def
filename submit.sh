#!/bin/bash -l
#SBATCH --job-name=sharpf
#SBATCH --nodes=1
#SBATCH --gres=gpu:4
#SBATCH --ntasks-per-node=1
#SBATCH --partition=gpu_small
#SBATCH --exclusive
#SBATCH --mem=300G
#SBATCH --time=0-23:59:59

# author: Ruslan Rakhimov

SIMAGE_PATH=/gpfs/gpfs0/3ddl/env/a1.sif
PROJECT_ROOT=/trinity/home/${USER}/sharp_features
args="$@"

cd ${PROJECT_ROOT}
srun singularity exec --bind /gpfs:/gpfs --nv ${SIMAGE_PATH} bash -c "
  NUM_GPUS=$(nvidia-smi --query-gpu=name --format=csv,noheader | wc -l) &&
  python train_net.py ${args} trainer.gpus=\${NUM_GPUS} trainer.distributed_backend=ddp_spawn
"
cd ${SLURM_SUBMIT_DIR}
exit 0
