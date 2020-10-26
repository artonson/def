#!/usr/bin/env bash

set -x
set -e

# directories configuration on the cluster
export DATA_DIR=/gpfs/gpfs0/3ddl/datasets/abc/
export INPUT_FILENAME=/gpfs/gpfs0/3ddl/sharp_features/data_v2_cvpr/points_whole/abc_0022_points_whole_item_ids.txt
export OUTPUT_BASE_DIR=/gpfs/gpfs0/3ddl/sharp_features/data_v2_cvpr
export DATATYPE=points_whole

# codebase configuration
export PROJECT_ROOT=/trinity/home/a.artemov/repos/sharp_features
export GENERATE_DATA_SCRIPT=${PROJECT_ROOT}/scripts/data_scripts/slurm/make_points_whole.sbatch.sh

# job configuration
export CONFIG_LIST="high_res_whole.json"

declare -A NOISE_LEVELS
# IMAGES IMAGES IMAGES IMAGES
# NOISE_LEVELS["high_res.json"]="0.0"
# NOISE_LEVELS[med_res]="0.2 0.1 0.05 0.025 0.0125 0.00625 0.0"
# NOISE_LEVELS[low_res]="0.5 0.25 0.125 0.0625 0.03125 0.015625 0.0"

# POINTS POINTS POINTS POINTS
NOISE_LEVELS["high_res_whole.json"]="0.0"
export NOISE_LEVELS

# data configuration
#export TRAIN_START_CHUNK="00"
#export TRAIN_END_CHUNK="01"
#export VAL_START_CHUNK="21"
#export VAL_END_CHUNK="21"
export TEST_START_CHUNK="22"
export TEST_END_CHUNK="22"


run_slurm_jobs() {

  local start_chunk=$1
  local end_chunk=$2
  local config=$3
  local input_file=$4

  for chunk in $( seq -w "${start_chunk}" "${end_chunk}" )
  do
    local output_dir=${OUTPUT_BASE_DIR}/${DATATYPE}/${config}/raw/${chunk}
    mkdir -p "${output_dir}"
    sbatch \
      --array=1-10 \
      --parsable \
      "${GENERATE_DATA_SCRIPT}" \
        -c "${chunk}" \
        -d "${DATA_DIR}" \
        -o "${output_dir}" \
        -l "${output_dir}" \
        -f "${config}" \
        -i "${input_file}" \
        -v
  done

}


for config in ${CONFIG_LIST}
do
#  if [[ -n "${TRAIN_START_CHUNK}" && -n "${TRAIN_END_CHUNK}" ]]
#  then
#    run_slurm_jobs "${TRAIN_START_CHUNK}" "${TRAIN_END_CHUNK}" "${config}" "${INPUT_FILENAME}"
#  fi
#
#  if [[ -n "${VAL_START_CHUNK}" && -n "${VAL_END_CHUNK}" ]]
#  then
#    run_slurm_jobs "${VAL_START_CHUNK}" "${VAL_END_CHUNK}" "${config}" "${INPUT_FILENAME}"
#  fi
#
  if [[ -n "${TEST_START_CHUNK}" && -n "${TEST_END_CHUNK}" ]]
  then
    run_slurm_jobs "${TEST_START_CHUNK}" "${TEST_END_CHUNK}" "${config}" "${INPUT_FILENAME}"
  fi
done
