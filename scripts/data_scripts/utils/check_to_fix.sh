#!/bin/bash

set -e
set -x

ABC_DIR=/gpfs/gpfs0/3ddl/datasets/abc.unpacked
DATA_BASE_DIR=/gpfs/gpfs0/3ddl/sharp_features/data_v2_cvpr
DATA_MODALITY=points

# CONFIG_LIST="high_res.json med_res.json low_res.json"
CONFIG_LIST="med_res.json"

# data configuration
TRAIN_START_CHUNK="00"
TRAIN_END_CHUNK="01"
VAL_START_CHUNK="21"
VAL_END_CHUNK="21"
TEST_START_CHUNK="22"
TEST_END_CHUNK="22"

compose_fixlist() {

  local start_chunk=$1
  local end_chunk=$2
  local config=$3

  for chunk in $( seq -w "${start_chunk}" "${end_chunk}" )
  do

    INPUT_DIR=${DATA_BASE_DIR}/${DATA_MODALITY}/${config}/raw/${chunk}

    for f in $( find "${INPUT_DIR}" -regextype sed -regex ".*/abc_[[:digit:]]\+_[[:digit:]]\+_[[:digit:]]\+.hdf5" | sort )
    do
        ls "${INPUT_DIR}/fix_$( basename -s .hdf5 ${f} )_0.hdf5" \
            >/dev/null \
            2>&1 \
        || echo ${f} >>~/tmp/fixlist2_${config}_${chunk}.txt
    done

  done

}



for config in ${CONFIG_LIST}
do 

  if [[ -n "${TRAIN_START_CHUNK}" && -n "${TRAIN_END_CHUNK}" ]]
  then
    compose_fixlist "${TRAIN_START_CHUNK}" "${TRAIN_END_CHUNK}" "${config}"
  fi

  if [[ -n "${VAL_START_CHUNK}" && -n "${VAL_END_CHUNK}" ]]
  then
    compose_fixlist "${VAL_START_CHUNK}" "${VAL_END_CHUNK}" "${config}"
  fi

  if [[ -n "${TEST_START_CHUNK}" && -n "${TEST_END_CHUNK}" ]]
  then
    compose_fixlist "${TEST_START_CHUNK}" "${TEST_END_CHUNK}" "${config}"
  fi

done


