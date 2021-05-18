#!/usr/bin/env bash

set -x
set -e

WAIT_JOBS=$1

echo "**************************************************" >&2
echo "cluster_make_symlinks.sh called with parameters:  " >&2
echo "  OUTPUT_BASE_DIR       = ${OUTPUT_BASE_DIR}      " >&2
echo "  DATATYPE              = ${DATATYPE}             " >&2
echo "  CONFIG_LIST           = ${CONFIG_LIST}          " >&2
echo "  TRAIN_START_CHUNK     = ${TRAIN_START_CHUNK}    " >&2
echo "  TRAIN_END_CHUNK       = ${TRAIN_END_CHUNK}      " >&2
echo "  VAL_START_CHUNK       = ${VAL_START_CHUNK}      " >&2
echo "  VAL_END_CHUNK         = ${VAL_END_CHUNK}        " >&2
echo "  TEST_START_CHUNK      = ${TEST_START_CHUNK}     " >&2
echo "  TEST_END_CHUNK        = ${TEST_END_CHUNK}       " >&2
echo "  GENERATE_DATA_SCRIPT  = ${GENERATE_DATA_SCRIPT} " >&2
echo "  SHUFFLE_SCRIPT        = ${SHUFFLE_SCRIPT}       " >&2
echo "--------------------------------------------------" >&2
echo "  WAIT_JOBS             = ${WAIT_JOBS}            " >&2
echo "**************************************************" >&2

declare -A NOISE_LEVELS
# POINTS POINTS POINTS POINTS
# NOISE_LEVELS["high_res.json"]="0.0"
# NOISE_LEVELS[med_res]="0.2 0.1 0.05 0.025 0.0125 0.00625 0.0"
# NOISE_LEVELS[low_res]="0.5 0.25 0.125 0.0625 0.03125 0.015625 0.0"

# IMAGES IMAGES IMAGES IMAGES
# NOISE_LEVELS["high_res.json"]="0.0025 0.005 0.01 0.02 0.04 0.08"
NOISE_LEVELS["high_res.json"]="0.005 0.02 0.08"
# NOISE_LEVELS["high_res.json"]="0.0"
NOISE_LEVELS["med_res.json"]="0.0"
NOISE_LEVELS["low_res.json"]="0.0"
export NOISE_LEVELS


run_slurm_jobs() {

  local start_chunk=$1
  local end_chunk=$2
  local pattern=$3
  local raw_input_dir=$4
  local dataset_output_dir=$5
  local split=$6
  local max_items_to_store=$7

  local symlinks_dir="${dataset_output_dir}/${split}_symlinks"
  local output_dir="${dataset_output_dir}/${split}"

  mkdir -p "${symlinks_dir}" "${output_dir}"

  for chunk in $( seq -w "${start_chunk}" "${end_chunk}" )
  do
      find -L \
        "${raw_input_dir}/${chunk}" \
        -type f \
        -regextype sed \
        -regex "${pattern}" \
        -print0 \
          | xargs -0 \
            cp -s -f --target-directory="${symlinks_dir}"
  done

  sbatch \
      --parsable \
      "${SHUFFLE_SCRIPT}" \
        -i "${symlinks_dir}" \
        -o "${output_dir}" \
        -m ${max_items_to_store} \
        -v

}


# simply wait for the ${WAIT_JOBS} to complete, doing nothing
[[ ! -z "${WAIT_JOBS}" ]] && \
srun \
  --dependency afterany:"${WAIT_JOBS}" \
  sleep 1 >/dev/null 2>&1


# build all datasets
for config in ${CONFIG_LIST}
do
  raw_input_dir=${OUTPUT_BASE_DIR}/${DATATYPE}/${config}/raw
 
  for noise_level in ${NOISE_LEVELS[${config}]}
  do
    dataset_output_dir=${OUTPUT_BASE_DIR}/${DATATYPE}/${config}/${noise_level}
 
    if [ "${noise_level}" != "0.0" ]
    then
      pattern=".*/abc_[[:digit:]]\+_[[:digit:]]\+_[[:digit:]]\+_${noise_level}.hdf5"
      # pattern=".*/fix_abc_[[:digit:]]\+_[[:digit:]]\+_[[:digit:]]\+_${noise_level}_0.hdf5"
    else
      pattern=".*/abc_[[:digit:]]\+_[[:digit:]]\+_[[:digit:]]\+.hdf5"
      # pattern=".*/fix_abc_[[:digit:]]\+_[[:digit:]]\+_[[:digit:]]\+_0.hdf5"
    fi
 
 
    if [[ -n "${TRAIN_START_CHUNK}" && -n "${TRAIN_END_CHUNK}" ]]
    then
      run_slurm_jobs "${TRAIN_START_CHUNK}" "${TRAIN_END_CHUNK}" "${pattern}" "${raw_input_dir}" "${dataset_output_dir}" train_arbitrary 131072
    fi
 
    if [[ -n "${VAL_START_CHUNK}" && -n "${VAL_END_CHUNK}" ]]
    then
      run_slurm_jobs "${VAL_START_CHUNK}" "${VAL_END_CHUNK}" "${pattern}" "${raw_input_dir}" "${dataset_output_dir}" val_arbitrary 65536
    fi
 
    if [[ -n "${TEST_START_CHUNK}" && -n "${TEST_END_CHUNK}" ]]
    then
      run_slurm_jobs "${TEST_START_CHUNK}" "${TEST_END_CHUNK}" "${pattern}" "${raw_input_dir}" "${dataset_output_dir}" test_arbitrary 65536
    fi
 
  done

done