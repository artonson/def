#!/bin/bash

#SBATCH --job-name=sharpf-train
#SBATCH --output=%A_%a.out
#SBATCH --error=%A_%a.err
#SBATCH --time=24:00:00
#SBATCH --partition=gpu_big
#SBATCH --gpus=1
#SBATCH --mem=8000

set -x

# get environment
source ../../env.sh

# set environment for this job
# check if script is started via SLURM or bash
# if with SLURM: there variable '$SLURM_JOB_ID' will exist
# `if [ -n $SLURM_JOB_ID ]` checks if $SLURM_JOB_ID is not an empty string
if [ -n $SLURM_JOB_ID ];  then
    echo "Script started with SLURM, SLURM_JOBID=$SLURM_JOBID"
    # check the original location through scontrol and $SLURM_JOB_ID
    # only take the last commandname since for job arrays we have multiple of them
    SCRIPT_PATH=$(scontrol show job $SLURM_JOBID | awk -F= '/Command=/{print $2}' | tail -n 1)
else
    # otherwise: started with bash. Get the real location.
    SCRIPT_PATH=$(realpath $0)
fi

# getting location of software_name 
HOST_CODE_DIR=$(realpath $(dirname ${SCRIPT_PATH})/../..)     # dirname of THIS file
HOST_DATA_DIR="/gpfs/gpfs0/3ddl/sharp_features/dist_field-8192-noiseless-tsdf3.2"
HOST_LOG_DIR="/gpfs/gpfs0/3ddl/sharp_features/logs"

CONT_CODE_DIR="/code"
CONT_DATA_DIR="/data"
CONT_LOG_DIR="/logs"

echo SLURM_ARRAY_TASK_ID=${SLURM_ARRAY_TASK_ID}


# specify training parameters
NUM_EPOCHS=10
GPU_ID=0
MODEL_SPEC=${CONT_CODE_DIR}/sharpf/models/specs/dgcnn.json
#JOB_ARRAY_CONFIG=${CONT_CODE_DIR}/vectran/models/specs/job-arrays/train-array.config

# load the module
module load apps/singularity-3.2.0

# execute the command within the container
singularity exec \
  --nv \
  --bind ${HOST_CODE_DIR}:${CONT_CODE_DIR} \
  --bind ${HOST_DATA_DIR}:${CONT_DATA_DIR} \
  --bind ${HOST_LOG_DIR}:${CONT_LOG_DIR} \
  ${SIMAGE_FILENAME} \
  python3 ${CONT_CODE_DIR}/scripts/train_sharp.py \
    --gpu ${GPU_ID} \
    --epochs ${NUM_EPOCHS} \
    --model-spec ${MODEL_SPEC} \
    --infer-from-spec \
    --data-root ${CONT_DATA_DIR} \
    --overwrite \
    --verbose

