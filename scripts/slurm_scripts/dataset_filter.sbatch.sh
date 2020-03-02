#!/bin/bash

#SBATCH --job-name=sharpf_dataset_filter
#SBATCH --output=array_%A_%a.out
#SBATCH --error=array_%A_%a.err
#SBATCH --array=40-99
#SBATCH --time=01:00:00
#SBATCH --partition=gpu_big
#SBATCH --cpus-per-task=24
#SBATCH --ntasks-per-node=1

set -x
CPUS_PER_TASK=24

SIMAGES_DIR=/gpfs/gpfs0/3ddl/singularity-images
IMAGE_NAME="artonson/sharp_features"
IMAGE_VERSION="latest"
IMAGE_NAME_TAG="${IMAGE_NAME}:${IMAGE_VERSION}"
SIMAGE_FILENAME="${SIMAGES_DIR}/$(echo ${IMAGE_NAME_TAG} | tr /: _).sif"

HOST_CODE_DIR="/trinity/home/a.artemov/repos/sharp_features"
HOST_DATA_DIR="/gpfs/gpfs0/3ddl/datasets/abc"
HOST_OUT_DIR="/gpfs/gpfs0/3ddl/sharp_features/data"
HOST_LOG_DIR="/logs"

CONT_CODE_DIR="/code"
CONT_DATA_DIR="/data"
CONT_OUT_DIR="/out"
CONT_LOG_DIR="/logs"

echo SLURM_ARRAY_TASK_ID=${SLURM_ARRAY_TASK_ID}

module load apps/singularity-3.2.0
singularity exec \
  --bind ${HOST_CODE_DIR}:${CONT_CODE_DIR} \
  --bind ${HOST_DATA_DIR}:${CONT_DATA_DIR} \
  --bind ${HOST_LOG_DIR}:${CONT_LOG_DIR} \
  --bind ${HOST_OUT_DIR}:${CONT_OUT_DIR} \
  ${SIMAGE_FILENAME} \
  python3 ${CONT_CODE_DIR}/scripts/dataset_utils/dataset_filter.py \
    --input-dir ${CONT_DATA_DIR} \
    --chunk ${SLURM_ARRAY_TASK_ID} \
    --output-dir ${CONT_OUT_DIR} \
    --jobs ${CPUS_PER_TASK} \
    --filter-config ${CONT_CODE_DIR}/scripts/dataset_utils/dataset_filter_config.json
