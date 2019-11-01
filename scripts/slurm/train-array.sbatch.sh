#!/bin/bash

#SBATCH --job-name=vectran-train-array
#SBATCH --output=array_%A_%a.out
#SBATCH --error=array_%A_%a.err
#SBATCH --array=0-9
#SBATCH --time=01:00:00
#SBATCH --partition=gpu_big
#SBATCH --cpus-per-task=4
#SBATCH --ntasks-per-node=4

set -x

# get environment
source ../../env.sh

# set environment for this job
HOST_CODE_DIR=$(realpath $(dirname `realpath $0`)/../..)     # dirname of THIS file
HOST_DATA_DIR="/gpfs/gpfs0/3ddl/vectorization/datasets/svg_datasets"
HOST_LOG_DIR="/gpfs/gpfs0/3ddl/vectorization/logs"

CONT_CODE_DIR="/code"
CONT_DATA_DIR="/data"
CONT_LOG_DIR="/logs"

echo SLURM_ARRAY_TASK_ID=${SLURM_ARRAY_TASK_ID}


# specify training parameters
NUM_EPOCHS=10
GPU_ID=0
MODEL_SPEC=${CONT_CODE_DIR}/vectran/models/specs/
DATA_TYPE=combined
MEMORY_LIMIT=21474836480  # 22 GB
JOB_ARRAY_CONFIG=${CONT_CODE_DIR}/vectran/models/specs/

module load apps/singularity-3.2.0
singularity exec \
  --nv \
  --bind ${HOST_CODE_DIR}:${CONT_CODE_DIR} \
  --bind ${HOST_DATA_DIR}:${CONT_DATA_DIR} \
  --bind ${HOST_LOG_DIR}:${CONT_LOG_DIR} \
  ${SIMAGE_FILENAME} \
  python3 ${CONT_CODE_DIR}/scripts/train_supervised.py \
    --gpu ${GPU_ID} \
    --epochs ${NUM_EPOCHS} \
    --model-spec ${MODEL_SPEC} \
    --infer-from-spec \
    --data-root ${CONT_DATA_DIR} \
    --data-type ${DATA_TYPE} \
    --memory-constraint ${MEMORY_LIMIT} \
    --overwrite \
    --job-array-config ${JOB_ARRAY_CONFIG} \
    --job-array-task-id ${SLURM_ARRAY_TASK_ID}

