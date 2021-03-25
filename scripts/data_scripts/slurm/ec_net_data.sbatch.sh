#!/bin/bash

#SBATCH --job-name=sharpf_dataset_filter
#SBATCH --output=array_%A_%a.out
#SBATCH --error=array_%A_%a.err
#SBATCH --time=48:00:00
#SBATCH --partition=cpu
#SBATCH --mem-per-cpu=4g
#SBATCH --cpus-per-task=24
#SBATCH --ntasks=1

set -x

while getopts "i:" opt
do
    case ${opt} in
        i) INPUT_FILE=$OPTARG;;
        *) usage; exit 1 ;;
    esac
done

if [[ ! ${INPUT_FILE} ]]; then
    echo "input_file is not set";
    usage
    exit 1
fi

CPUS_PER_TASK=24

SIMAGES_DIR=/gpfs/gpfs0/3ddl/singularity-images
IMAGE_NAME="artonson/sharp_features"
IMAGE_VERSION="latest"
IMAGE_NAME_TAG="${IMAGE_NAME}:${IMAGE_VERSION}"
SIMAGE_FILENAME="${SIMAGES_DIR}/$(echo ${IMAGE_NAME_TAG} | tr /: _).sif"

HOST_CODE_DIR="/trinity/home/a.matveev/sharp_features/"
HOST_DATA_DIR="/gpfs/gpfs0/3ddl/datasets/abc/"
HOST_OUT_DIR="/trinity/home/a.matveev/ec_net_data_2/${INPUT_FILE}"
HOST_LOG_DIR="/trinity/home/a.matveev/logs/"
HOST_FILE_DIR="/trinity/home/a.matveev/ec_net_data_2/"

CONT_CODE_DIR="/code/"
CONT_DATA_DIR="/data/"
CONT_FILE_DIR="/file/"
CONT_OUT_DIR="/out/"
CONT_LOG_DIR="/logs/"

echo SLURM_ARRAY_TASK_ID=${SLURM_ARRAY_TASK_ID}

module load apps/singularity-3.2.0
singularity exec \
  --bind ${HOST_CODE_DIR}:${CONT_CODE_DIR} \
  --bind ${HOST_DATA_DIR}:${CONT_DATA_DIR} \
  --bind ${HOST_FILE_DIR}:${CONT_FILE_DIR} \
  --bind ${HOST_LOG_DIR}:${CONT_LOG_DIR} \
  --bind ${HOST_OUT_DIR}:${CONT_OUT_DIR} \
  ${SIMAGE_FILENAME} \
  python3 ${CONT_CODE_DIR}/scripts/data_scripts/ec_net_data.py \
    --input-dir ${CONT_FILE_DIR}/${INPUT_FILE} \
    --abc-dir ${CONT_DATA_DIR} \
    --output-dir ${CONT_OUT_DIR} \
    --jobs ${CPUS_PER_TASK}

