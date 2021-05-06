#!/bin/bash

set -x 

# /mnt/ssd/artonson/def_predictions/data_rw_siggraph/images_aligninf_partmesh_whole_testonly/amed
INPUT_DIR_GT=/data/images_aligninf_partmesh_whole/test
INPUT_DIR_PRED=/logs/images_aligninf_partmesh_whole_testonly/amed
OUTPUT_DIR=/logs/images_aligninf_partmesh_whole_testonly/amed


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

    python ../plot_depth_sharpness_grid.py \
        -i ${OUTPUT_DIR}/$( basename $f .hdf5)__predictions.hdf5 \
        -o ${OUTPUT_DIR}/$( basename $f .hdf5)__predictions.png \
        -s 10.0 -di -si -r 2048 1536 --ncols 4 -f 16 16 -c 512 -bgd -dv 0 -sv 10

    python ../plot_depth_sharpness_grid.py \
        -i ${INPUT_DIR_GT}/$f \
        -o ${OUTPUT_DIR}/$( basename $f .hdf5)__ground_truth.png \
        -s 10.0 -di -si -r 2048 1536 --ncols 4 -f 16 16 -c 512 -w -bgd -dv 0 -sv 0

    python ../plot_snapshots.py \
        -i ${OUTPUT_DIR}/$( basename $f .hdf5)__ground_truth.hdf5 \
        -i ${OUTPUT_DIR}/$( basename $f .hdf5)__min.hdf5 \
        -o ${OUTPUT_DIR}/$( basename $f .hdf5).html \
        -s 10.0 -ps 0.5 -ph flat

#echo "" | tr -d '\n'
#    echo

done
