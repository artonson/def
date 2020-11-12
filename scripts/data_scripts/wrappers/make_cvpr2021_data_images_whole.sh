#!/usr/bin/env bash

set -x
set -e

# directories configuration on the cluster
export DATA_DIR=/gpfs/gpfs0/3ddl/datasets/abc/
export INPUT_FILENAME=/gpfs/gpfs0/3ddl/sharp_features/data_v2_cvpr/printed_models.txt
export OUTPUT_BASE_DIR=/gpfs/gpfs0/3ddl/sharp_features/data_v2_cvpr
export DATATYPE=images_whole

# codebase configuration
export PROJECT_ROOT=/trinity/home/a.artemov/repos/sharp_features
export GENERATE_DATA_SCRIPT=${PROJECT_ROOT}/scripts/data_scripts/slurm/make_images_whole.sbatch.sh

# job configuration
export CONFIG_LIST="high_res_whole.json high_res_whole_0.005.json high_res_whole_0.02.json high_res_whole_0.08.json med_res_whole.json low_res_whole.json"

# data configuration
#export TRAIN_START_CHUNK="00"
#export TRAIN_END_CHUNK="01"
#export VAL_START_CHUNK="21"
#export VAL_END_CHUNK="21"
export TEST_START_CHUNK="50"
export TEST_END_CHUNK="50"


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
