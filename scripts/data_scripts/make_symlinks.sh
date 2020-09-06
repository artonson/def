#!/bin/bash

set -e
set -x


###############################################################
###############################################################

# PROJECT_DATA_BASE_DIR=/gpfs/gpfs0/3ddl/sharp_features
# DATASET_NAME=data_v2_cvpr
# MODALITY=points
# MODALITY_BASE_DIR=${PROJECT_DATA_BASE_DIR}/${DATASET_NAME}/${MODALITY}

# TRAIN_START_CHUNK="00"
# TRAIN_END_CHUNK="01"
# VAL_START_CHUNK="21"
# VAL_END_CHUNK="21"
# TEST_START_CHUNK="22"
# TEST_END_CHUNK="22"

# RESOLUTIONS="high_res med_res low_res"
# declare -A NOISE_LEVELS
# NOISE_LEVELS[high_res]="0.08 0.04 0.02 0.01 0.005 0.0025 0.0"
# NOISE_LEVELS[med_res]="0.2 0.1 0.05 0.025 0.0125 0.00625 0.0"
# NOISE_LEVELS[low_res]="0.5 0.25 0.125 0.0625 0.03125 0.015625 0.0"




###############################################################
###############################################################

PROJECT_DATA_BASE_DIR=/gpfs/gpfs0/3ddl/sharp_features
DATASET_NAME=data_v2_cvpr
MODALITY=images
MODALITY_BASE_DIR=${PROJECT_DATA_BASE_DIR}/${DATASET_NAME}/${MODALITY}

TRAIN_START_CHUNK="00"
TRAIN_END_CHUNK="01"
VAL_START_CHUNK="21"
VAL_END_CHUNK="21"
TEST_START_CHUNK="22"
TEST_END_CHUNK="22"

RESOLUTIONS="high_res"
declare -A NOISE_LEVELS
NOISE_LEVELS[high_res]="0.0"
# NOISE_LEVELS[med_res]="0.2 0.1 0.05 0.025 0.0125 0.00625 0.0"
# NOISE_LEVELS[low_res]="0.5 0.25 0.125 0.0625 0.03125 0.015625 0.0"


for resolution in ${RESOLUTIONS}
do
    resolution_dir_raw=${MODALITY_BASE_DIR}/dataset_config_${resolution}_clean_64x64.json/raw
    echo ${resolution_dir_raw}

    # noise_levels="$( ls -1 ${resolution_dir_raw}/${chunk} | cut -d _  -f 5 | sort | uniq | cut -d . -f 1-2) 0.0"
    for noise_level in ${NOISE_LEVELS[${resolution}]}
    do
        if [ ${noise_level} != "0.0" ]
        then
             pattern=".*/abc_[[:digit:]]\+_[[:digit:]]\+_[[:digit:]]\+_${noise_level}.hdf5"
        else
             pattern=".*/abc_[[:digit:]]\+_[[:digit:]]\+_[[:digit:]]\+.hdf5"
        fi


        for chunk in $( seq -w ${TRAIN_START_CHUNK} ${TRAIN_END_CHUNK} )
        do
            sub_dataset_dir=${MODALITY_BASE_DIR}/${resolution}/${noise_level}
            mkdir -p \
                ${sub_dataset_dir}/train_symlinks \
                ${sub_dataset_dir}/train

            echo ${chunk} ${noise_level} \
              && find ${resolution_dir_raw}/${chunk} -type f -regextype sed -regex ${pattern} -print0 \
               | xargs -0 cp -s -f --target-directory=${sub_dataset_dir}/train_symlinks
        done


        for chunk in $( seq -w ${VAL_START_CHUNK} ${VAL_END_CHUNK} )
        do
            sub_dataset_dir=${MODALITY_BASE_DIR}/${resolution}/${noise_level}
            mkdir -p \
                ${sub_dataset_dir}/val_symlinks \
                ${sub_dataset_dir}/val
  
            echo ${chunk} ${noise_level} \
              && find ${resolution_dir_raw}/${chunk} -type f -regextype sed -regex ${pattern} -print0 \
               | xargs -0 cp -s -f --target-directory=${sub_dataset_dir}/val_symlinks
        done
  
  
        for chunk in $( seq -w ${TEST_START_CHUNK} ${TEST_END_CHUNK} )
        do
            sub_dataset_dir=${MODALITY_BASE_DIR}/${resolution}/${noise_level}
            mkdir -p \
                ${sub_dataset_dir}/test_symlinks \
                ${sub_dataset_dir}/test
  
            echo ${chunk} ${noise_level} \
              && find ${resolution_dir_raw}/${chunk} -type f -regextype sed -regex ${pattern} -print0 \
               | xargs -0 cp -s -f --target-directory=${sub_dataset_dir}/test_symlinks
        done
    done
done



