#!/bin/bash

set -x 

# /mnt/ssd/artonson/def_predictions/data_rw_siggraph/images_aligninf_partmesh_whole_testonly/amed

#DATASETS="images_align4mm_fullmesh_whole
#images_align4mm_fullmesh_whole_testonly
#images_align4mm_partmesh_whole
#images_align4mm_partmesh_whole_testonly
#images_aligninf_fullmesh_whole
#images_aligninf_fullmesh_whole_testonly
#images_aligninf_partmesh_whole
#images_aligninf_partmesh_whole_testonly"

DATASETS="images_aligninf_partmesh_whole"

CONFIGS="real_images_bw1.yml
real_images_bw2.yml
real_images_bw4.yml
real_images_dthr1.yml
real_images_dthr4.yml
real_images_dthr16.yml
real_images_nn4.yml
real_images_nn16.yml
real_images_ratio0.04.yml
real_images_ratio0.16.yml
real_images_ratio0.32.yml
real_images_ratio0.64.yml
real_images_zthr1.yml
real_images_zthr16.yml"



for dataset in ${DATASETS}
do
    INPUT_DIR_GT=/data/${dataset}/test
    INPUT_DIR_PRED=/logs/${dataset}/amed
    OUTPUT_DIR=/logs/${dataset}/amed
    echo "${INPUT_DIR_GT} ${INPUT_DIR_PRED} ${OUTPUT_DIR}"


    for f in $( ls -1 ${INPUT_DIR_GT} )
    do
        ./fuse_images.py \
            -t ${INPUT_DIR_GT}/$f \
            -p ${INPUT_DIR_PRED}/$( basename $f .hdf5)/predictions/ \
            -o ${OUTPUT_DIR} \
            -j 36 \
            -f configs/real_images_base.yml \
            -s 10.0 \
            -r 10.0
 
        python ../plot_depth_sharpness_grid.py \
            -i ${INPUT_DIR_GT}/$f \
            -o ${OUTPUT_DIR}/$( basename $f .hdf5)__ground_truth.png \
            -s 11.0 -di -si --ncols 1 -f 8 8 -c auto -cx -w -dv 0 -sv 0 -bgd --verbose -dp -sp &
 
        python ../plot_depth_sharpness_grid.py \
            -i ${OUTPUT_DIR}/$( basename $f .hdf5)__predictions.hdf5 \
            -o ${OUTPUT_DIR}/$( basename $f .hdf5)__predictions.png \
            -s 11.0 -di -si --ncols 1 -f 8 8 -c auto -cx -dv 0 -sv 10 -bgd --verbose -sp &
 
        python ../plot_depth_sharpness_grid.py \
            -i ${OUTPUT_DIR}/$( basename $f .hdf5)__absdiff.hdf5 \
            -o ${OUTPUT_DIR}/$( basename $f .hdf5)__absdiff.png \
            -s 11.0 -di -si --ncols 1 -f 8 8 -c auto -cx -dv 0 -sv 0 -bgd --verbose -scm plasma -sp &

        python ../plot_snapshots.py \
            -i ${OUTPUT_DIR}/$( basename $f .hdf5)__ground_truth.hdf5 \
            -i ${OUTPUT_DIR}/$( basename $f .hdf5)__min.hdf5 \
            -i ${OUTPUT_DIR}/$( basename $f .hdf5)__min__absdiff.hdf5 \
            -i ${OUTPUT_DIR}/$( basename $f .hdf5)__adv60.hdf5 \
            -i ${OUTPUT_DIR}/$( basename $f .hdf5)__adv60__absdiff.hdf5 \
            -o ${OUTPUT_DIR}/$( basename $f .hdf5).html \
            -s 11.0 -ps 0.5 -ph flat &

        python ../compute_metrics.py \
            -t ${OUTPUT_DIR}/$( basename $f .hdf5)__ground_truth.hdf5 \
            -p ${OUTPUT_DIR}/$( basename $f .hdf5)__adv60.hdf5 \
            -r 0.5 -s 10.0 >${OUTPUT_DIR}/$( basename $f .hdf5)__adv60__metrics.txt &
        python ../compute_metrics.py \
            -t ${OUTPUT_DIR}/$( basename $f .hdf5)__ground_truth.hdf5 \
            -p ${OUTPUT_DIR}/$( basename $f .hdf5)__min.hdf5 \
            -r 0.5 -s 10.0 >${OUTPUT_DIR}/$( basename $f .hdf5)__min__metrics.txt &

        wait
        /usr/bin/convert \
            ${OUTPUT_DIR}/$( basename $f .hdf5)__ground_truth.png \
            ${OUTPUT_DIR}/$( basename $f .hdf5)__predictions.png \
            ${OUTPUT_DIR}/$( basename $f .hdf5)__absdiff.png \
            +append ${OUTPUT_DIR}/$( basename $f .hdf5)__result.png
 
    #echo "" | tr -d '\n'
    #    echo
 
    done


done

