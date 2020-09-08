#!/usr/bin/env bash

set -x
set -e

WAIT_JOBS=$1

echo "**************************************************" >&2
echo "cluster_generate_data.sh called with parameters:  " >&2
echo "  DATA_DIR              = ${DATA_DIR}             " >&2
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
echo "--------------------------------------------------" >&2
echo "  WAIT_JOBS             = ${WAIT_JOBS}            " >&2
echo "**************************************************" >&2


run_slurm_jobs() {

  local start_chunk=$1
  local end_chunk=$2
  local config=$3

  for chunk in $( seq -w "${start_chunk}" "${end_chunk}" )
  do
    local output_dir=${OUTPUT_BASE_DIR}/${DATATYPE}/${config}/raw/${chunk}
    mkdir -p "${output_dir}"
    sbatch \
      --parsable \
      "${GENERATE_DATA_SCRIPT}" \
        -c "${chunk}" \
        -d "${DATA_DIR}" \
        -o "${output_dir}" \
        -l "${output_dir}" \
        -f "${config}" \
        -v
  done

}


for config in ${CONFIG_LIST}
do
  if [[ -n "${TRAIN_START_CHUNK}" && -n "${TRAIN_END_CHUNK}" ]]
  then
    run_slurm_jobs "${TRAIN_START_CHUNK}" "${TRAIN_END_CHUNK}" "${config}"
  fi

  if [[ -n "${VAL_START_CHUNK}" && -n "${VAL_END_CHUNK}" ]]
  then
    run_slurm_jobs "${VAL_START_CHUNK}" "${VAL_END_CHUNK}" "${config}"
  fi

  if [[ -n "${TEST_START_CHUNK}" && -n "${TEST_END_CHUNK}" ]]
  then
    run_slurm_jobs "${TEST_START_CHUNK}" "${TEST_END_CHUNK}" "${config}"
  fi
done

