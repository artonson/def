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


filter_sample() {

  local start_chunk=$1
  local end_chunk=$2
  local config=$3
  local task_count=$4

  for chunk in $( seq -w "${start_chunk}" "${end_chunk}" )
  do
    local output_dir=${OUTPUT_BASE_DIR}/${DATATYPE}/${config}/raw/${chunk}
    mkdir -p "${output_dir}"
    sbatch \
      --array=1-${task_count} \
      --time=24:00:00 \
      --parsable \
      "${GENERATE_DATA_SCRIPT}" \
        -c "${chunk}" \
        -d "${DATA_DIR}" \
        -o "${output_dir}" \
        -l "${output_dir}" \
        -f "${config}" \
        -v \
        -s 10
  done

}

python ${FILTER_SAMPLE_SCRIPT} \
  --verbose \
  --input ${} \
  --output ${} \
  --resolution_3d ${} \
  --distances_near_thr_factor ${} \
  --knn_radius_factor ${} \
  --min_cc_points_to_keep ${} \
  --subsample_rate ${}
