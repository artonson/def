#!/bin/bash

set -e
#set -x

### POINTS CONFIG
# PROJECT_DATA_BASE_DIR=/gpfs/gpfs0/3ddl/sharp_features
# DATASET_NAME=data_v2_cvpr
# MODALITY=points
# MODALITY_BASE_DIR=${PROJECT_DATA_BASE_DIR}/${DATASET_NAME}/${MODALITY}

# RESOLUTIONS="high_res med_res low_res"
# declare -A NOISE_LEVELS
# NOISE_LEVELS[high_res]="0.08 0.04 0.02 0.01 0.005 0.0025 0.0"
# NOISE_LEVELS[med_res]="0.2 0.1 0.05 0.025 0.0125 0.00625 0.0"
# NOISE_LEVELS[low_res]="0.5 0.25 0.125 0.0625 0.03125 0.015625 0.0"
# SPLITS="train val test"


### IMAGES CONFIG
PROJECT_DATA_BASE_DIR=/gpfs/gpfs0/3ddl/sharp_features
DATASET_NAME=data_v2_cvpr
MODALITY=images
MODALITY_BASE_DIR=${PROJECT_DATA_BASE_DIR}/${DATASET_NAME}/${MODALITY}

RESOLUTIONS="high_res"
declare -A NOISE_LEVELS
# NOISE_LEVELS[high_res]="0.08 0.04 0.02 0.01 0.005 0.0025 0.0"
NOISE_LEVELS[high_res]="0.0"
# NOISE_LEVELS[med_res]="0.2 0.1 0.05 0.025 0.0125 0.00625 0.0"
# NOISE_LEVELS[low_res]="0.5 0.25 0.125 0.0625 0.03125 0.015625 0.0"
SPLITS="train"
#SPLITS="train val test"

for resolution in ${RESOLUTIONS}
do

    for noise_level in ${NOISE_LEVELS[${resolution}]}
    do

        for split in ${SPLITS}
        do
            split_dir=${MODALITY_BASE_DIR}/${resolution}/${noise_level}/${split}
            echo $split_dir

            mkdir -p ${split_dir}

            #sbatch -d afterany:334969:334970 \
            sbatch \
                make_canonical_datasets.sbatch.sh \
                -i ${split_dir}_symlinks \
                -o ${split_dir} \
                -v
        done

    done
done



