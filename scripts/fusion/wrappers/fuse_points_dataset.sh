#!/usr/bin/env bash

#set -x
set -e

# Set up variables -- we assume we're working in docker container
# where all of the aligned scans are mounted into subfolder in '/data' folder.
# Also assume '/logs' is mounted and writable.
# '/code' contains path to this repository.
INPUT_BASE_DIR=/data/mirrored_11
INPUT_SUBFOLDER_REGEX='.*/[0-9]+(top|side)(_folder)?'
OUTPUT_BASE_DIR=/logs/sharp_features_real_data
#OUTPUT_PREPROCESSED_DIR=${OUTPUT_BASE_DIR}/preprocessed
#OUTPUT_POINTS_DIR=${OUTPUT_BASE_DIR}/points
#OUTPUT_IMAGES_DIR=${OUTPUT_BASE_DIR}/images

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
 -o ${folder}/$( basename "${folder}" )_preprocessed.hdf5
 ${DEBUG_FLAG}
 ${VERBOSE_FLAG}
 >${folder}/$( basename "${folder}" )_preprocessed.log 2>&1"
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


####################################################################
# 2. Preprocess scans.
echo "Preprocessing scans..."

NUM_PROCESSES=36
parallel -j ${NUM_PROCESSES} <${PREPARE_CMDS_FILE}

# If you don't have parallel...
#PREPARE_PREFIX=prepare_
#split ${PREPARE_CMDS_FILE} \
#  --number=l/${NUM_PROCESSES} \
#  ${PREPARE_PREFIX}


####################################################################
# 3. Collect information about existing scans that need conversion.
#echo "Preparing scans to process as point clouds..."

function preprocess_points() {
  local folder=$1
  echo "python ${POINTS_SCRIPT}
 -i ${folder}
 -o ${folder}/$( basename "${folder}" )_points.hdf5
 -d ${MAX_POINT_MESH_DISTANCE}
 -s ${MAX_DISTANCE_TO_FEATURE}
 ${DEBUG_FLAG}
 ${VERBOSE_FLAG}
 >${folder}/$( basename "${folder}" )_points.log 2>&1"

    ./fuse_images.py \
        -t /logs/images/$f \
        -p /logs/images_predictions/$( basename $f .hdf5)/predictions/ \
        -o /logs/images_fused/ \
        -j 12 \
        -f image.yml \
        -s 10.0
}

POINTS_CMDS_FILE=all_points_commands.sh
find ${INPUT_BASE_DIR} \
  -type d \
  -regextype egrep \
  -regex "${INPUT_SUBFOLDER_REGEX}" \
  -print0 | \
  while IFS= read -d '' file
  do 
    preprocess_points "$file" | tr -d '\n'
    echo
  done >${POINTS_CMDS_FILE}


####################################################################
# 4. Prepare points datasets
echo "Preprocessing scans as point clouds..."

NUM_PROCESSES=36
parallel -j ${NUM_PROCESSES} <${POINTS_CMDS_FILE}


####################################################################
# 5. Collect information about existing scans that need conversion.
#echo "Preparing scans to process as images..."

#function preprocess_images() {
#  local folder=$1
#  echo "python ${IMAGES_SCRIPT}
# -i ${folder}
# -o ${folder}/$( basename "${folder}" )_images.hdf5
# -d ${MAX_POINT_MESH_DISTANCE}
# -s ${MAX_DISTANCE_TO_FEATURE}
# --full_mesh
# ${DEBUG_FLAG}
# ${VERBOSE_FLAG}
# >${folder}/$( basename "${folder}" )_images.log 2>&1"
#}

#IMAGES_CMDS_FILE=all_images_commands.sh
#find ${INPUT_BASE_DIR} \
#  -type d \
#  -regextype egrep \
#  -regex "${INPUT_SUBFOLDER_REGEX}" \
#  -print0 | \
#  while IFS= read -d '' file
#  do 
#    preprocess_images "$file" | tr -d '\n'
#    echo
#  done >${IMAGES_CMDS_FILE}


####################################################################
# 6. Prepare images datasets
#echo "Preprocessing scans as images..."

#NUM_PROCESSES=36
#parallel --progress -j ${NUM_PROCESSES} <${IMAGES_CMDS_FILE}


set -x

for f in $( ls -1 /logs/images )
do
    ./fuse_images.py \
        -t /logs/images/$f \
        -p /logs/images_predictions/$( basename $f .hdf5)/predictions/ \
        -o /logs/images_fused/ \
        -j 12 \
        -f image.yml \
        -s 10.0

#   ../plot_depth_sharpness_grid.py \
#       -i /logs/images_fused/$( basename $f .hdf5)__predictions.hdf5 \
#       -o /logs/images_fused/$( basename $f .hdf5)__predictions.png \
#       -s 1.0 -si -r 2048 1536 --ncols 1 -f 20 240 -c 512

    #    -i /logs/images_fused/$( basename $f .hdf5)__min.hdf5
    echo "python ../plot_snapshots.py
        -i /logs/images_fused/$( basename $f .hdf5)__ground_truth.hdf5
        -o /logs/images_fused/$( basename $f .hdf5).html
        -s 10.0 -ps 0.5 -ph flat" | tr -d '\n'
    echo

done
