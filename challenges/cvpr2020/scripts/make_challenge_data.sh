#!/usr/bin/env bash

set -e
set -x

# This script is intended to use on prometheus machine
# All paths have been configured so.

DATASETS_DIR=/mnt/neuro/artonson/sharp_features_datasets
MODALITY=points
PROJECT_ROOT="$( cd "$( dirname "${BASH_SOURCE[0]}" )"/../.. >/dev/null 2>&1 && pwd )"
OUTPUT_DIR=/home/artonson/datasets/cvpr_abc_challenge


hdf5_to_txt() {
    local DATASETS_DIR=$1
    local MODALITY=$2
    local RES=$3
    local SPLIT=$4
    local I1=$5
    local I2=$6
    local OUTPUT_BASE_DIR=$7
    local FORMAT=$8
    local SPLIT_XY=$9
    local OUTPUT_THR=${10}

    if ${SPLIT_XY} ; then
        SPLIT_ARG="--split-xy"
    else
        SPLIT_ARG=""
    fi

    if [[ ${OUTPUT_THR} ]] ; then
        PREFIX="segmentation_"
        OUTPUT_THR_ARG="--target-thr ${OUTPUT_THR}"
    else
        PREFIX="field_"
    fi

    for FILE_IDX in $( seq ${I1} ${I2} )
    do
        local DATASET="${DATASETS_DIR}/${MODALITY}/dataset_config_${RES}_res_clean.json/${SPLIT}/${SPLIT}_${FILE_IDX}.hdf5"
        local REL_PATH="${FORMAT}/${RES}_res/${SPLIT}_${FILE_IDX}"
        local OUTPUT_DIR="${OUTPUT_BASE_DIR}/${REL_PATH}"

        ./convert_hdf5_to_txts.py \
            --input-file ${DATASET} \
            --output-dir ${OUTPUT_DIR} \
            --data-label points \
            --target-label distances \
            --output-format ${FORMAT} \
            --output-prefix ${PREFIX} \
            ${SPLIT_ARG} \
            ${OUTPUT_THR_ARG}

        if [[ ${FORMAT} == "npy" ]]; then
            if ${SPLIT_XY}; then
                mv "${OUTPUT_DIR}/${PREFIX}000000_data.npy" "${OUTPUT_DIR}/${PREFIX}${FILE_IDX}_data.npy"
                mv "${OUTPUT_DIR}/${PREFIX}000000_target.npy" "${OUTPUT_DIR}/${PREFIX}${FILE_IDX}_target.npy"
            else
                mv "${OUTPUT_DIR}/${PREFIX}000000.npy" "${OUTPUT_DIR}/${PREFIX}${FILE_IDX}.npy"
            fi
        fi

        if ${SPLIT_XY}; then
            local TARGZ_FILENAME="${OUTPUT_BASE_DIR}/${FORMAT}/${RES}_res_${SPLIT}_${FILE_IDX}_data.tar.gz"
            mkdir -p "${OUTPUT_BASE_DIR}/${REL_PATH}_data" \
             && mv ${OUTPUT_DIR}/*_data.* "${OUTPUT_BASE_DIR}/${REL_PATH}_data" \
             && tar -zcf ${TARGZ_FILENAME} -C ${OUTPUT_BASE_DIR} "${REL_PATH}_data"

            local TARGZ_FILENAME="${OUTPUT_BASE_DIR}/${FORMAT}/${RES}_res_${SPLIT}_${FILE_IDX}_target.tar.gz"
            mkdir -p "${OUTPUT_BASE_DIR}/${REL_PATH}_target" \
             && mv ${OUTPUT_DIR}/*_target.* "${OUTPUT_BASE_DIR}/${REL_PATH}_target" \
             && tar -zcf ${TARGZ_FILENAME} -C ${OUTPUT_BASE_DIR} "${REL_PATH}_target"

        else
            local TARGZ_FILENAME="${OUTPUT_BASE_DIR}/${FORMAT}/${RES}_res_${SPLIT}_${FILE_IDX}.tar.gz"
            tar \
                -zcf ${TARGZ_FILENAME} \
                -C ${OUTPUT_BASE_DIR} \
                ${REL_PATH}
        fi

        # rm -rf ${OUTPUT_DIR}
    done
}

MAX_TRAIN_IDX=0
MAX_VAL_IDX=0
MAX_TEST_IDX=0

# text files
for RES in high med low
do
    # train files
    hdf5_to_txt ${DATASETS_DIR} ${MODALITY} ${RES} train 0 ${MAX_TRAIN_IDX} ${OUTPUT_DIR} txt false
    # val files
    hdf5_to_txt ${DATASETS_DIR} ${MODALITY} ${RES} val 0 ${MAX_VAL_IDX} ${OUTPUT_DIR} txt true 
    # test files
    hdf5_to_txt ${DATASETS_DIR} ${MODALITY} ${RES} test 0 ${MAX_TEST_IDX} ${OUTPUT_DIR} txt true 
done


# NPY files
for RES in high med low
do
    # train files
    hdf5_to_txt ${DATASETS_DIR} ${MODALITY} ${RES} train 0 ${MAX_TRAIN_IDX} ${OUTPUT_DIR} npy false
    # val files
    hdf5_to_txt ${DATASETS_DIR} ${MODALITY} ${RES} val 0 ${MAX_VAL_IDX} ${OUTPUT_DIR} npy true
    # test files
    hdf5_to_txt ${DATASETS_DIR} ${MODALITY} ${RES} test 0 ${MAX_TEST_IDX} ${OUTPUT_DIR} npy true
done

