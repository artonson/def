#!/usr/bin/env bash

#set -x
set -e

# Set up variables -- we assume we're working in docker container
# where all of the aligned scans are mounted into subfolder in '/data' folder.
# Also assume '/logs' is mounted and writable.
# '/code' contains path to this repository.
INPUT_BASE_DIR=/data
INPUT_SUBFOLDER_REGEX='.*/[0-9]+(top|side)(_folder)?'
OUTPUT_POINTS_DIR=/logs/real_points
OUTPUT_IMAGES_DIR=/logs/real_images

PREPARE_SCRIPT=/code/scripts/data_scripts/prepare_real_scans.py
POINTS_SCRIPT=/code/scripts/data_scripts/prepare_real_points_dataset.py
IMAGES_SCRIPT=/code/scripts/data_scripts/prepare_real_images_dataset.py

MAX_POINT_MESH_DISTANCE=10.0
MAX_DISTANCE_TO_FEATURE=2.5

DEBUG_FLAG="--debug"
VERBOSE_FLAG="--verbose"

####################################################################
# 1. Collect information about existing scans that need conversion.
echo "Fetching scans to preprocess..."

function print_prepare() {
  local folder=$1
  echo "python ${PREPARE_SCRIPT}
 -i ${folder}
 -o ${OUTPUT_IMAGES_DIR}/$( basename "${folder}" )_preprocessed.hdf5
 ${DEBUG_FLAG}
 ${VERBOSE_FLAG}"
}

PREPARE_CMDS_FILE=all_prepare_commands.sh
find ${INPUT_BASE_DIR} \
  -type d \
  -regextype egrep \
  -regex "${INPUT_SUBFOLDER_REGEX}" \
  -print0 | \
  while IFS= read -d '' file
  do 
    print_prepare "$file" | tr -d '\n'
    echo
  done >${PREPARE_CMDS_FILE}


# 1.1. Preprocess scans.
NUM_PROCESSES=20
echo "Preprocessing scans..."
parallel -j ${NUM_PROCESSES} <${PREPARE_CMDS_FILE}

# If you don't have parallel...
#PREPARE_PREFIX=prepare_
#split ${PREPARE_CMDS_FILE} \
#  --number=l/${NUM_PROCESSES} \
#  ${PREPARE_PREFIX}


####################################################################
# 2. Collect information about existing scans that need conversion.
echo "Collecting information..."

#function preprocess_points() {
#  local folder=$1
#  echo "python ${POINTS_SCRIPT} \\
#    -i ${folder} \\
#    -o ${OUTPUT_IMAGES_DIR}/$( basename "${folder}" )_images.hdf5 \\
#    -d ${MAX_POINT_MESH_DISTANCE} \\
#    -s ${MAX_DISTANCE_TO_FEATURE} \\
#    ${DEBUG_FLAG} \\
#    ${VERBOSE_FLAG}"
#}

#
#for folder in $( find ${INPUT_BASE_DIR} -type d -regextype egrep -regex ${INPUT_SUBFOLDER_REGEX} )
#do
#  asd
#done


#
#
#
## 1. Collect information about existing scans that need conversion.
#echo "Creating images datasets..."
#
#function print_prepare() {
#  local folder=$1
#  echo "python ${PREPARE_SCRIPT}
#    -i ${folder}
#    -o ${OUTPUT_IMAGES_DIR}/$( basename "${folder}" )_images.hdf5
#    -d ${MAX_POINT_MESH_DISTANCE}
#    -s ${MAX_DISTANCE_TO_FEATURE}
#    ${DEBUG_FLAG}
#    ${VERBOSE_FLAG}"
#}
#
#
#
#
#
#for d in $( find /data -mindepth 2 -maxdepth 2 -type d ) ; do echo "python /code/scripts/data_scripts/prepare_real_points_dataset.py -i ${d} -o /logs/real_world_patches/$( basename $d )_patches.hdf5 -d 10.0 -s 2.5 --debug" ; done >/logs/real_world_patches/commands.txt
#
#
#
#
## 1. Collect information about existing scans that need conversion.
#echo "Creating images datasets..."
#INPUT_SUBFOLDER_REGEX='.*/[0-9]+(top|side)(_folder)?'
#
#mkdir -p ${OUTPUT_IMAGES_DIR}
#for folder in $( find ${INPUT_BASE_DIR} -type d -regextype egrep -regex ${INPUT_SUBFOLDER_REGEX} ) ;
#do
#  echo "python ${IMAGES_SCRIPT} -i ${folder} -o ${OUTPUT_IMAGES_DIR}/$( basename ${folder} )_images.hdf5 -d 10.0 -s 2.5 --debug"
#done >/logs/real_world_images/commands.txt
