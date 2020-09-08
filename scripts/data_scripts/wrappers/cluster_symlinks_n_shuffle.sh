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


run_slurm_jobs() {

  local start_chunk=$1
  local end_chunk=$2
  local pattern=$3
  local raw_input_dir=$4
  local dataset_output_dir=$5
  local split=$6

  local symlinks_dir="${dataset_output_dir}/${split}_symlinks"
  local output_dir="${dataset_output_dir}/${split}"

  mkdir -p "${symlinks_dir}" "${output_dir}"

  for chunk in $( seq -w "${start_chunk}" "${end_chunk}" )
  do
      find "${raw_input_dir}/${chunk}" \
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
        -v

}


# simply wait for the ${WAIT_JOBS} to complete, doing nothing
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
    else
      pattern=".*/abc_[[:digit:]]\+_[[:digit:]]\+_[[:digit:]]\+.hdf5"
    fi


    if [[ -n "${TRAIN_START_CHUNK}" && -n "${TRAIN_END_CHUNK}" ]]
    then
      run_slurm_jobs "${TRAIN_START_CHUNK}" "${TRAIN_END_CHUNK}" "${pattern}" "${raw_input_dir}" "${dataset_output_dir}" train
    fi

    if [[ -n "${VAL_START_CHUNK}" && -n "${VAL_END_CHUNK}" ]]
    then
      run_slurm_jobs "${VAL_START_CHUNK}" "${VAL_END_CHUNK}" "${pattern}" "${raw_input_dir}" "${dataset_output_dir}" val
    fi

    if [[ -n "${TEST_START_CHUNK}" && -n "${TEST_END_CHUNK}" ]]
    then
      run_slurm_jobs "${TEST_START_CHUNK}" "${TEST_END_CHUNK}" "${pattern}" "${raw_input_dir}" "${dataset_output_dir}" test
    fi

  done

done
