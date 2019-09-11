#!/usr/bin/env bash

CPUS_PER_TASK=20

#SBATCH -J sharpf_dataset_filter
#SBATCH --partition=cpu_big
#SBATCH --cpus-per-task=${CPUS_PER_TASK}
#SBATCH --ntasks=1
#SBATCH --nodes=10
#SBATCH --time 00:05:00
#SBATCH --array=0-99

# ${SLURM_ARRAY_TASK_ID} is set by SLURM

SIMAGES_DIR=/gpfs/gpfs0/3ddl/singularity-images
IMAGE_NAME="artonson/sharp_features"
IMAGE_VERSION="latest"
IMAGE_NAME_TAG="${IMAGE_NAME}:${IMAGE_VERSION}"
SIMAGE_FILENAME="${SIMAGES_DIR}/$(echo ${IMAGE_NAME_TAG} | tr /: _).sif"

HOST_CODE_DIR=$(realpath $(dirname `realpath $0`)/../..)     # dirname of THIS file's parent dir
HOST_DATA_DIR="/gpfs/gpfs0/3ddl/datasets/abc"
HOST_OUT_DIR="/gpfs/gpfs0/3ddl/sharp_features/data"
HOST_LOG_DIR="/logs"

CONT_CODE_DIR="/code"
CONT_DATA_DIR="/data"
CONT_OUT_DIR="/out"
CONT_LOG_DIR="/logs"


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
    --jobs ${CPUS_PER_TASK}

