#!/bin/bash

set -x 

INPUT_DIR_GT=/logs/images
INPUT_DIR_PRED=/logs/images_predictions
OUTPUT_DIR=/logs/images_fused


for f in $( ls -1 ${INPUT_DIR_GT} )
do 
    ./fuse_images.py \
        -t ${INPUT_DIR_GT}/$f \
        -p ${INPUT_DIR_PRED}/$( basename $f .hdf5)/predictions/ \
        -o ${OUTPUT_DIR} \
        -j 24 \
        -f image.yml \
        -s 10.0 \
        -r 10.0

#   ../plot_depth_sharpness_grid.py \
#       -i ${OUTPUT_DIR}/$( basename $f .hdf5)__predictions.hdf5 \
#       -o ${OUTPUT_DIR}/$( basename $f .hdf5)__predictions.png \
#       -s 1.0 -si -r 2048 1536 --ncols 1 -f 20 240 -c 512

    python ../plot_snapshots.py \
        -i ${OUTPUT_DIR}/$( basename $f .hdf5)__ground_truth.hdf5 \
        -i ${OUTPUT_DIR}/$( basename $f .hdf5)__min.hdf5 \
        -o ${OUTPUT_DIR}/$( basename $f .hdf5).html \
        -s 10.0 -ps 0.5 -ph flat
#echo "" | tr -d '\n'
#    echo

done
