#!/usr/bin/env bash

set -x
set -e

# directories configuration on the cluster
export DATA_DIR=/gpfs/gpfs0/3ddl/datasets/abc/
export OUTPUT_BASE_DIR=/gpfs/gpfs0/3ddl/sharp_features/data_v2_cvpr
export DATATYPE=points

# codebase configuration
export PROJECT_ROOT=/trinity/home/a.artemov/repos/sharp_features
export GENERATE_DATA_SCRIPT=${PROJECT_ROOT}/scripts/data_scripts/slurm/make_points.sbatch.sh
export SHUFFLE_SCRIPT=${PROJECT_ROOT}/scripts/data_scripts/slurm/make_canonical_datasets.sbatch.sh

# job configuration
export CONFIG_LIST="high_res.json"

declare -A NOISE_LEVELS
NOISE_LEVELS["high_res.json"]="0.0"
# NOISE_LEVELS[med_res]="0.2 0.1 0.05 0.025 0.0125 0.00625 0.0"
# NOISE_LEVELS[low_res]="0.5 0.25 0.125 0.0625 0.03125 0.015625 0.0"
export NOISE_LEVELS

# data configuration
export TRAIN_START_CHUNK="00"
export TRAIN_END_CHUNK="01"
export VAL_START_CHUNK="21"
export VAL_END_CHUNK="21"
export TEST_START_CHUNK="22"
export TEST_END_CHUNK="22"


JOB_IDS_1=$( ./cluster_generate_data )

JOB_IDS_2=$( ./cluster_symlinks_n_shuffle.sh ${JOB_IDS_1})
