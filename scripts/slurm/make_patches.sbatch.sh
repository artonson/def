#!/bin/bash

#SBATCH --job-name=sharpf_make_data_1024-noiseless-tsdf3.2
#SBATCH --output=array_%A_%a.out
#SBATCH --error=array_%A_%a.err
#SBATCH --array=1-20
#SBATCH --time=3-0
#SBATCH --partition=cpu_big
#SBATCH --cpus-per-task=24
#SBATCH --ntasks-per-node=1
#SBATCH --mem=40000
###SBATCH --reservation=project_1

set -x
CPUS_PER_TASK=24

SIMAGES_DIR=/gpfs/gpfs0/3ddl/singularity-images
IMAGE_NAME="artonson/sharp_features"
IMAGE_VERSION="latest"
IMAGE_NAME_TAG="${IMAGE_NAME}:${IMAGE_VERSION}"
SIMAGE_FILENAME="${SIMAGES_DIR}/$(echo ${IMAGE_NAME_TAG} | tr /: _).sif"

#DATASET="dist_field-512-noiseless-tsdf1.0"
#DATASET="dist_field-1024-noiseless-tsdf1.0"
#DATASET="dist_field-2048-noiseless-tsdf1.0"
#DATASET="dist_field-4096-noise0.1-tsdf1.0"
#DATASET="dist_field-4096-noise0.4-tsdf1.0"
#DATASET="dist_field-4096-noise0.025-tsdf1.0"
#DATASET="dist_field-4096-noise1.6-tsdf1.0"
#DATASET="dist_field-4096-noise3.2-tsdf1.0"
#DATASET="dist_field-4096-noiseless-tsdf0.4"
#DATASET="dist_field-4096-noiseless-tsdf0.8"
#DATASET="dist_field-4096-noiseless-tsdf1.0"
#DATASET="dist_field-4096-noiseless-tsdf1.6"
#DATASET="dist_field-4096-noiseless-tsdf3.2"
#DATASET="dist_field-8192-noiseless-tsdf1.0"
#DATASET="dist_field-512-noiseless-tsdf3.2"
DATASET="dist_field-1024-noiseless-tsdf3.2"
#DATASET="dist_field-2048-noiseless-tsdf3.2"
#DATASET="dist_field-4096-noiseless-tsdf3.2"
#DATASET="dist_field-8192-noiseless-tsdf3.2"

HOST_CODE_DIR="/trinity/home/a.matveev/sharp_features-master"
HOST_DATA_DIR="/gpfs/gpfs0/3ddl/datasets/abc"
HOST_OUT_DIR="/gpfs/gpfs0/3ddl/sharp_features/geodesic/${DATASET}"
HOST_LOG_DIR="/trinity/home/a.matveev/logs"
mkdir -p ${HOST_OUT_DIR} && chmod -R 777 ${HOST_OUT_DIR}

CONT_CODE_DIR="/code"
CONT_DATA_DIR="/data"
CONT_OUT_DIR="/out"
CONT_LOG_DIR="/logs"

echo SLURM_ARRAY_TASK_ID="${SLURM_ARRAY_TASK_ID}"

module load apps/singularity-3.2.0
singularity exec \
  --bind ${HOST_CODE_DIR}:${CONT_CODE_DIR} \
  --bind ${HOST_DATA_DIR}:${CONT_DATA_DIR} \
  --bind ${HOST_LOG_DIR}:${CONT_LOG_DIR} \
  --bind ${HOST_OUT_DIR}:${CONT_OUT_DIR} \
  "${SIMAGE_FILENAME}" \
  python3 ${CONT_CODE_DIR}/scripts/dataset_utils/make_data.py \
    --input-dir ${CONT_DATA_DIR} \
    --chunk "${SLURM_ARRAY_TASK_ID}" \
    --output-dir ${CONT_OUT_DIR} \
    --jobs ${CPUS_PER_TASK} \
    --dataset-config ${CONT_CODE_DIR}/sharpf/data/configs/${DATASET}.json

