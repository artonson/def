#!/usr/bin/env bash

DATA_DIR=/gpfs/gpfs0/3ddl/datasets/abc/
OUTPUT_BASE_DIR=/gpfs/gpfs0/3ddl/sharp_features/data_v2_cvpr
PROJECT_ROOT=/trinity/home/a.artemov/repos/sharp_features
# SCRIPT_NAME=${PROJECT_ROOT}/scripts/data_scripts/slurm/make_patches.sbatch.sh
SCRIPT_NAME=${PROJECT_ROOT}/scripts/data_scripts/slurm/make_images.sbatch.sh
# DATATYPE=points
DATATYPE=images

# for chunk in $( seq -w 00 01)
for chunk in 21 22
do
  # for config in dataset_config_high_res_clean.json dataset_config_low_res_clean.json dataset_config_med_res_clean.json
#  for config in dataset_config_high_res_noisy.json dataset_config_med_res_noisy.json dataset_config_low_res_noisy.json
  for config in dataset_config_high_res_clean_64x64.json
  do
    echo "chunk = ${chunk} config = ${config}"
    OUTPUT_DIR=${OUTPUT_BASE_DIR}/${DATATYPE}/${config}/raw/${chunk}
    mkdir -p "${OUTPUT_DIR}"
#     echo "sbatch ${SCRIPT_NAME} -c "${chunk}" -d ${DATA_DIR} -o "${OUTPUT_DIR}" -l "${OUTPUT_DIR}" -f ${config} -v"
    sbatch ${SCRIPT_NAME} \
      -c "${chunk}" \
      -d ${DATA_DIR} \
      -o "${OUTPUT_DIR}" \
      -l "${OUTPUT_DIR}" \
      -f ${config} \
      -v
  done
done

