#!/bin/bash

module load apps/singularity-3.2.0

__usage="Usage: $0 -i input_dir"
usage() { echo "$__usage" >&2; }

# Get all the required options and set the necessary variables
while getopts "i:" opt
do
    case ${opt} in
        i) INPUT_DIR_HOST=$OPTARG;;
        *) usage; exit 1 ;;
    esac
done

INPUT_DIR_CONTAINER="/input"
if [[ ! ${INPUT_DIR_HOST} ]]; then
    echo "input_dir is not set" && usage && exit 1
fi

H5_LEN_SCRIPT=/code/sharpf/utils/scripts/get_hdf5_size.py

# get image filenames from here
PROJECT_ROOT=/trinity/home/a.artemov/repos/sharp_features
source "${PROJECT_ROOT}"/env.sh

CODE_PATH_CONTAINER="/code"
CODE_PATH_HOST=${PROJECT_ROOT}

singularity exec \
  --bind ${CODE_PATH_HOST}:${CODE_PATH_CONTAINER} \
  --bind "${INPUT_DIR_HOST}":${INPUT_DIR_CONTAINER} \
  --bind /gpfs:/gpfs \
  "${SIMAGE_FILENAME}" \
      bash -c \
      "python3 ${H5_LEN_SCRIPT} \\
        --input-dir ${INPUT_DIR_CONTAINER}"
