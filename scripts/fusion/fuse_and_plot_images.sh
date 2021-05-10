#!/bin/bash

# set -x 

# /mnt/ssd/artonson/def_predictions/data_rw_siggraph/images_aligninf_partmesh_whole_testonly/amed

#DATASETS="images_align4mm_fullmesh_whole
#images_align4mm_fullmesh_whole_testonly
#images_align4mm_partmesh_whole
#images_align4mm_partmesh_whole_testonly
#images_aligninf_fullmesh_whole
#images_aligninf_fullmesh_whole_testonly
#images_aligninf_partmesh_whole
#images_aligninf_partmesh_whole_testonly"

#DATASETS="images_align4mm_fullmesh_whole"

#CONFIGS="real_images_bw1.yml
#real_images_bw2.yml
#real_images_bw4.yml
#real_images_dthr1.yml
#real_images_dthr4.yml
#real_images_dthr16.yml
#real_images_nn4.yml
#real_images_nn16.yml
#real_images_ratio0.04.yml
#real_images_ratio0.16.yml
#real_images_ratio0.32.yml
#real_images_ratio0.64.yml
#real_images_zthr1.yml
#real_images_zthr16.yml"


#DATASETS="images_align4mm_fullmesh_whole
#images_align4mm_fullmesh_whole_testonly"
#images_align4mm_partmesh_whole
#images_align4mm_partmesh_whole_testonly"

PRED_DATASET="images_align4mm_partmesh_whole"
GT_DATASET="images_align4mm_fullmesh_whole"

CONFIGS="real_images_base.yml"

awk_compute_mean_std() {
    in_dir=$1
    mask=$2

    tail -q -n+2 ${in_dir}/*${mask}.txt \
        | awk -F, '{sum+=$1; sumsq+=$1*$1} END {print "mRMSE-ALL=" sum/NR " +/- " sqrt(sumsq/NR - (sum/NR)*(sum/NR))}'

    tail -q -n+2 ${in_dir}/*${mask}.txt \
        | awk -F, '{sum+=$2; sumsq+=$2*$2} END {print "mq95RMSE-ALL=" sum/NR " +/- " sqrt(sumsq/NR - (sum/NR)*(sum/NR))}'

    tail -q -n+2 ${in_dir}/*${mask}.txt \
        | awk -F, '{sum+=$3; sumsq+=$3*$3} END {print "mBadPoints(0.5)-Close-Sharp=" sum/NR " +/- " sqrt(sumsq/NR - (sum/NR)*(sum/NR))}'

    tail -q -n+2 ${in_dir}/*${mask}.txt \
        | awk -F, '{sum+=$4; sumsq+=$4*$4} END {print "mBadPoints(2)-Close-Sharp=" sum/NR " +/- " sqrt(sumsq/NR - (sum/NR)*(sum/NR))}'

    tail -q -n+2 ${in_dir}/*${mask}.txt \
        | awk -F, '{sum+=$5; sumsq+=$5*$5} END {print "mIOU-Sharp=" sum/NR " +/- " sqrt(sumsq/NR - (sum/NR)*(sum/NR))}'
}

for config in ${CONFIGS}
do

    for gt_dataset in ${GT_DATASET}
    do
        INPUT_DIR_GT=/data/${gt_dataset}/test
        INPUT_DIR_PRED=/logs/${pred_dataset}/amed
        OUTPUT_DIR=/logs/${pred_dataset}/amed
        echo "${INPUT_DIR_GT} ${INPUT_DIR_PRED} ${OUTPUT_DIR}"

        output_path_config=${OUTPUT_DIR}/$( basename ${config} .yml )
        mkdir -p ${output_path_config}

        for f in $( ls -1 ${INPUT_DIR_GT} )
        do
#           ./fuse_images.py \
#               -t ${INPUT_DIR_GT}/$f \
#               -p ${INPUT_DIR_PRED}/$( basename $f .hdf5)/predictions/ \
#               -o ${output_path_config} \
#               -j 36 \
#               -f configs/${config} \
#               -s 10.0 \
#               -r 10.0
     
#           python ../plot_depth_sharpness_grid.py \
#               -i ${INPUT_DIR_GT}/$f \
#               -o ${output_path_config}/$( basename $f .hdf5)__ground_truth.png \
#               -s 11.0 -di -si --ncols 1 -f 8 8 -c auto -cx -w -dv 0 -sv 0 -bgd --verbose -dp -sp &
#    
#           python ../plot_depth_sharpness_grid.py \
#               -i ${output_path_config}/$( basename $f .hdf5)__predictions.hdf5 \
#               -o ${output_path_config}/$( basename $f .hdf5)__predictions.png \
#               -s 11.0 -di -si --ncols 1 -f 8 8 -c auto -cx -dv 0 -sv 10 -bgd --verbose -sp &
#    
#           python ../plot_depth_sharpness_grid.py \
#               -i ${output_path_config}/$( basename $f .hdf5)__absdiff.hdf5 \
#               -o ${output_path_config}/$( basename $f .hdf5)__absdiff.png \
#               -s 11.0 -di -si --ncols 1 -f 8 8 -c auto -cx -dv 0 -sv 0 -bgd --verbose -scm plasma -sp &
#
            python ../plot_snapshots.py \
                -i ${output_path_config}/$( basename $f .hdf5)__ground_truth.hdf5 \
                -i ${output_path_config}/$( basename $f .hdf5)__min.hdf5 \
                -i ${output_path_config}/$( basename $f .hdf5)__min__absdiff.hdf5 \
                -i ${output_path_config}/$( basename $f .hdf5)__adv60.hdf5 \
                -i ${output_path_config}/$( basename $f .hdf5)__adv60__absdiff.hdf5 \
                -icm plasma_r \
                -icm plasma_r \
                -icm plasma \
                -icm plasma_r \
                -icm plasma \
                -o ${output_path_config}/$( basename $f .hdf5).html \
                -s 11.0 -ps 1.25 -ph flat &

#           python ../compute_metrics.py \
#               -t ${INPUT_DIR_GT}/$f \
#               -p ${output_path_config}/$( basename $f .hdf5)__predictions.hdf5 \
#               -r 0.5 -s 10.0 -sv >${output_path_config}/$( basename $f .hdf5)__metrics__single_view.txt &
#
#           python ../compute_metrics.py \
#               -t ${output_path_config}/$( basename $f .hdf5)__ground_truth.hdf5 \
#               -p ${output_path_config}/$( basename $f .hdf5)__adv60.hdf5 \
#               -r 0.5 -s 10.0 >${output_path_config}/$( basename $f .hdf5)__adv60__metrics.txt &
#           python ../compute_metrics.py \
#               -t ${output_path_config}/$( basename $f .hdf5)__ground_truth.hdf5 \
#               -p ${output_path_config}/$( basename $f .hdf5)__min.hdf5 \
#               -r 0.5 -s 10.0 >${output_path_config}/$( basename $f .hdf5)__min__metrics.txt &

#           wait
#           /usr/bin/convert \
#               ${output_path_config}/$( basename $f .hdf5)__ground_truth.png \
#               ${output_path_config}/$( basename $f .hdf5)__predictions.png \
#               ${output_path_config}/$( basename $f .hdf5)__absdiff.png \
#               +append ${output_path_config}/$( basename $f .hdf5)__result.png

        done

        wait

#       awk_compute_mean_std ${output_path_config} __metrics__single_view \
#           >${output_path_config}/metrics__single_view.txt
#
#       awk_compute_mean_std ${output_path_config} __min__metrics \
#           >${output_path_config}/min__metrics__fusion.txt
#
#       awk_compute_mean_std ${output_path_config} __adv60__metrics \
#           >${output_path_config}/adv60__metrics__fusion.txt


    done

done

