#!/usr/bin/env bash

DATA_DIR=/gpfs/gpfs0/3ddl/datasets/abc/
OUTPUT_BASE_DIR=/gpfs/gpfs0/3ddl/sharp_features/eccv_data
PROJECT_ROOT=/trinity/home/a.artemov/repos/sharp_features
SCRIPT_NAME=${PROJECT_ROOT}/scripts/slurm_scripts/make_images.sbatch.sh
DATATYPE=images

for chunk in $( seq -w 01 01)
do
  for config in dataset_config_high_res_clean_512x512.json
  do
    echo "chunk = ${chunk} config = ${config}"
    OUTPUT_DIR=${OUTPUT_BASE_DIR}/${DATATYPE}/${config}/${chunk}
    mkdir -p "${OUTPUT_DIR}"
    sbatch ${SCRIPT_NAME} \
      -c "${chunk}" \
      -d ${DATA_DIR} \
      -o "${OUTPUT_DIR}" \
      -l "${OUTPUT_DIR}" \
      -f ${config} \
      -v
  done
done

