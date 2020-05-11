#!/usr/bin/env bash

set -e
set -x

# This script is intended to use on prometheus machine
# All paths have been configured so.

DATASETS_DIR=/mnt/neuro/artonson/sharp_features_datasets
MODALITY=points
PROJECT_ROOT="$( cd "$( dirname "${BASH_SOURCE[0]}" )"/../.. >/dev/null 2>&1 && pwd )"
OUTPUT_DIR=/home/artonson/tmp


hdf5_to_txt() {
    local DATASETS_DIR=$1
    local MODALITY=$2
    local RES=$3
    local SPLIT=$4
    local I1=$5
    local I2=$6
    local OUTPUT_BASE_DIR=$7

    for FILE_IDX in $( seq ${I1} ${I2} )
    do
        local OUTPUT_DIR="${OUTPUT_BASE_DIR}/${SPLIT}_${FILE_IDX}"
        local DATASET="${DATASETS_DIR}/${MODALITY}/dataset_config_${RES}_res_clean.json/${SPLIT}/${SPLIT}_${FILE_IDX}.hdf5"

        ./convert_hdf5_to_txts.py \
            -i ${DATASET} \
            -o ${OUTPUT_DIR} \
            -d points \
            -t distances

        local TARGZ_FILENAME="${RES}_res_${SPLIT}_${FILE_IDX}.tar.gz"
        tar -zcf ${OUTPUT_BASE_DIR}/${TARGZ_FILENAME} ${OUTPUT_DIR}

        # rm -rf ${OUTPUT_DIR}
    done
}


for RES in high med low
do
    # train files
    # hdf5_to_txt ${DATASETS_DIR} ${MODALITY} ${RES} train 0 0 ${OUTPUT_DIR}

    # val files
    hdf5_to_txt ${DATASETS_DIR} ${MODALITY} ${RES} val 0 0 ${OUTPUT_DIR}

    # test files
    hdf5_to_txt ${DATASETS_DIR} ${MODALITY} ${RES} test 0 0 ${OUTPUT_DIR}
done
