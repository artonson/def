#!/bin/bash

set -ex

export ABC_DATA_DIR=/gpfs/gpfs0/3ddl/datasets/abc/
export PROJECT_ROOT=/trinity/home/a.artemov/repos/sharp_features2
export SCRIPT=${PROJECT_ROOT}/scripts/data_scripts/slurm/compute_dataset_statistics.sbatch.sh
export DEF_DATA_ROOT=/gpfs/gpfs0/3ddl/sharp_features

IMAGES_IN="${DEF_DATA_ROOT}/data_v2_cvpr/images/high_res.json/0.0/train/train_0.hdf5
${DEF_DATA_ROOT}/data_v2_cvpr/images/med_res.json/0.0/train/train_0.hdf5
${DEF_DATA_ROOT}/data_v2_cvpr/images/low_res.json/0.0/train/train_0.hdf5
${DEF_DATA_ROOT}/data_v3_cvpr/images/high_res.json/0.0/train/train_0.hdf5
${DEF_DATA_ROOT}/data_v3_cvpr/images/med_res.json/0.0/train/train_0.hdf5
${DEF_DATA_ROOT}/data_v3_cvpr/images/low_res.json/0.0/train/train_0.hdf5"

CONFIGS=(
  high_res_whole.json
  med_res_whole.json
  low_res_whole.json
  high_res_whole.json
  med_res_whole.json
  low_res_whole.json
)

#
#data_v3_cvpr images high
#data_v3_cvpr images med
#data_v3_cvpr images low
#
#data_v2_cvpr images high
#data_v2_cvpr images med
#data_v2_cvpr images low
#

i=0
for IN in ${IMAGES_IN}
do
  for CHUNK in 00 01
  do
    OUT="$( dirname "${IN}" )/train_0__${CHUNK}.txt"
    sbatch ${SCRIPT} \
      -a ${ABC_DATA_DIR} \
      -i "${IN}" \
      -o "${OUT}" \
      -c ${CHUNK} \
      -s images \
      -f "depthmap_datasets/${CONFIGS[$i]}" \
      -v
  done
  i=$(( i + 1 ))
done

#data_v2_cvpr points high
#data_v2_cvpr points med
#data_v2_cvpr points low

POINTS_IN="${DEF_DATA_ROOT}/data_v2_cvpr/points/high_res.json/0.0/train/train_0.hdf5
${DEF_DATA_ROOT}/data_v2_cvpr/points/med_res.json/0.0/train/train_0.hdf5
${DEF_DATA_ROOT}/data_v2_cvpr/points/low_res.json/0.0/train/train_0.hdf5"
CONFIGS=(
  high_res_whole.json
  med_res_whole.json
  low_res_whole.json
)

i=0
for IN in ${POINTS_IN}
do
  for CHUNK in 00 01
  do
    OUT="$( dirname "${IN}" )/train_0__${CHUNK}.txt"
    sbatch ${SCRIPT} \
      -a ${ABC_DATA_DIR} \
      -i "${IN}" \
      -o "${OUT}" \
      -c ${CHUNK} \
      -s points \
      -f "pointcloud_datasets/${CONFIGS[$i]}" \
      -v
  done
  i=$(( i + 1 ))
done


#
#/gpfs/data/gpfs0/3ddl/sharp_features/data_v2_cvpr/images_whole/high_res_whole.json/raw
#/gpfs/data/gpfs0/3ddl/sharp_features/data_v2_cvpr/images_whole/high_res_whole.json/raw
#data_v2_cvpr images_whole high [18 views] 50 | 51
#data_v2_cvpr images_whole high [128 views] images_128_views
#data_v2_cvpr images_whole med [18 views]
#data_v2_cvpr images_whole low [18 views]
#
#data_v2_cvpr points_whole high
#data_v2_cvpr points_whole med
#data_v2_cvpr points_whole low
#
#
