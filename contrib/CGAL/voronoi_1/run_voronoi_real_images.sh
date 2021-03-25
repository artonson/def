#!/usr/bin/env bash

set -x
set -e

usage() { echo "Usage: ${0} -i input_file -o output_file [-R offset_radius] [-r conv_radius] [-t threshold] [-j num_jobs]" >&2; }

V_OFFSET_RADIUS=0.2
V_CONV_RADIUS=0.1
V_THRESHOLD=0.16
NUM_JOBS=1
while getopts "i:o:R:r:t:j:v:" opt
do
    case ${opt} in
        i) INPUT_HDF5_FILENAME=${OPTARG} ;;
        o) OUTPUT_HDF5_FILENAME=${OPTARG} ;;
        R) V_OFFSET_RADIUS=${OPTARG} ;;
        r) V_CONV_RADIUS=${OPTARG} ;;
        t) V_THRESHOLD=${OPTARG} ;;
        j) NUM_JOBS=${OPTARG} ;;
        v) VARLEN=${OPTARG} ;;
        *) usage; exit 1 ;;
    esac
done

# check input variables
[[ -f ${INPUT_HDF5_FILENAME} ]] || { echo "input_file not set or empty"; usage; exit 1; }
OUTPUT_DIR="$( cd "$( dirname "${OUTPUT_HDF5_FILENAME}" )" >/dev/null 2>&1 && pwd )"
[[ -d ${OUTPUT_DIR} ]] || { echo "output directory ${HOST_OUTPUT_DIR} needs to be created first"; usage; exit 1; }

# prepate directories and needed environment
INPUT_XYZ_DIR=$( mktemp -d )
OUTPUT_XYZ_DIR=$( mktemp -d )

PY_SRC_DIR=/code/hdf5_utils
SPLIT_SCRIPT=${PY_SRC_DIR}/split_hdf5_images.py
MERGE_SCRIPT=${PY_SRC_DIR}/merge_hdf5.py
BIN_DIR=/home/user/code/voronoi_1
BINARY=${BIN_DIR}/voronoi_1

DATA_LABEL="image"
TARGET_LABEL="distances"

# Run commands
python3 ${SPLIT_SCRIPT} \
    --label ${DATA_LABEL} \
    --output_dir ${INPUT_XYZ_DIR} \
    ${INPUT_HDF5_FILENAME}

find ${INPUT_XYZ_DIR} -type f -name "*.xyz" \
    | parallel \
        --jobs ${NUM_JOBS} \
            ${BINARY} \
            -f {} \
            -R ${V_OFFSET_RADIUS} \
            -r ${V_CONV_RADIUS} \
            -t ${V_THRESHOLD} \
            -o ${OUTPUT_XYZ_DIR} \
        {}

python3 ${MERGE_SCRIPT} \
    -i ${OUTPUT_XYZ_DIR} \
    -o ${OUTPUT_HDF5_FILENAME} \
    --label ${TARGET_LABEL} \
    --input_format txt \
    --varlen ${VARLEN}

rm -rf ${INPUT_XYZ_DIR} ${OUTPUT_XYZ_DIR}
