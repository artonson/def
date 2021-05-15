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
#images_align4mm_fullmesh_whole_testonly
#images_align4mm_partmesh_whole
#images_align4mm_partmesh_whole_testonly"

DATASETS="images_align4mm_fullmesh_whole"
VIEWS_DATASET="images_align4mm_fullmesh_whole"
FUSED_DATASET="images_align4mm_fullmesh_whole"

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


#100side_folder_images__align4mm_fullmesh_whole__fused_adv60__absdiff__metrics.txt
#<instance_id>__<training_dataset>__<what>__<what_variant>__<characteristic>

for config in ${CONFIGS}
do

    for gt_dataset in ${DATASETS}
    do
        pred_dataset=${gt_dataset}

        INPUT_DIR_GT=/data/${gt_dataset}/test
        INPUT_DIR_PRED=/logs/${pred_dataset}/amed
        OUTPUT_DIR=/logs/${pred_dataset}/amed
        echo "${INPUT_DIR_GT} ${INPUT_DIR_PRED} ${OUTPUT_DIR}"

        output_path_config=${OUTPUT_DIR}/$( basename ${config} .yml )
        mkdir -p ${output_path_config}


        dataset_views__metrics="${output_path_config}/metrics__single_view.txt"
        dataset_fused_pred_min__metrics="${output_path_config}/min__metrics__fusion.txt"
        dataset_fused_pred_adv60__metrics="${output_path_config}/adv60__metrics__fusion.txt"
        dataset_fused_pred_linreg__metrics="${output_path_config}/linreg__metrics__fusion.txt"

        for f in $( ls -1 ${INPUT_DIR_GT} )
        do
            # views_gt="${INPUT_DIR_GT}/$f"
            views_gt="/data/${VIEWS_DATASET}/test/${f}"
            views_pred_dir="${INPUT_DIR_PRED}/$( basename $f .hdf5)/predictions/"

            ./fuse_images.py \
                -t ${views_gt} \
                -p ${views_pred_dir} \
                -o ${output_path_config} \
                -j 36 \
                -f configs/${config} \
                -s 10.0 \
                -r 10.0

            views_gt__grid="${output_path_config}/$( basename $f .hdf5)__ground_truth.png"
            views_pred="${output_path_config}/$( basename $f .hdf5)__predictions.hdf5"
            views_pred__grid="${output_path_config}/$( basename $f .hdf5)__predictions.png"
            views_absdiff="${output_path_config}/$( basename $f .hdf5)__absdiff.hdf5"
            views_absdiff__grid="${output_path_config}/$( basename $f .hdf5)__absdiff.png"
            views_result__grid="${output_path_config}/$( basename $f .hdf5)__result.png"
            views__metrics="${output_path_config}/$( basename $f .hdf5)__metrics__single_view.txt"
 
            # fused_gt="${output_path_config}/$( basename $f .hdf5)__ground_truth.hdf5"
            fused_gt="/logs/${FUSED_DATASET}/amed/$( basename ${config} .yml )/$( basename $f .hdf5)__ground_truth.hdf5"
 
            fused_pred_min="${output_path_config}/$( basename $f .hdf5)__min.hdf5"
            fused_pred_min_absdiff="${output_path_config}/$( basename $f .hdf5)__min__absdiff.hdf5"
            fused_pred_min__metrics="${output_path_config}/$( basename $f .hdf5)__min__metrics.txt"
 
            fused_pred_adv60="${output_path_config}/$( basename $f .hdf5)__adv60.hdf5"
            fused_pred_adv60_absdiff="${output_path_config}/$( basename $f .hdf5)__adv60__absdiff.hdf5"
            fused_pred_adv60__metrics="${output_path_config}/$( basename $f .hdf5)__adv60__metrics.txt"
 
            fused_pred_linreg="${output_path_config}/$( basename $f .hdf5)__adv60__linreg.hdf5"
            fused_pred_linreg_absdiff="${output_path_config}/$( basename $f .hdf5)__linreg__absdiff.hdf5"
            fused_pred_linreg__metrics="${output_path_config}/$( basename $f .hdf5)__linreg__metrics.txt"

            fused_snapshot="${output_path_config}/$( basename $f .hdf5).html"
 
 
            python ./fusion_analysis.py \
                -t ${fused_gt} \
                -p ${fused_pred_min} \
                -o ${fused_pred_min_absdiff}
            python ./fusion_analysis.py \
                -t ${fused_gt} \
                -p ${fused_pred_adv60} \
                -o ${fused_pred_adv60_absdiff}
            python ./fusion_analysis.py \
                -t ${fused_gt} \
                -p ${fused_pred_linreg} \
                -o ${fused_pred_linreg_absdiff}
 
            python ../plot_depth_sharpness_grid.py \
                -i ${views_gt} \
                -o ${views_gt__grid} \
                -s 11.0 -di -si --ncols 1 -f 8 8 -c auto -cx -w -dv 0 -sv 0 -bgd --verbose -dp -sp &
 
            python ../plot_depth_sharpness_grid.py \
                -i ${views_pred} \
                -o ${views_pred__grid} \
                -s 11.0 -di -si --ncols 1 -f 8 8 -c auto -cx -dv 0 -sv 10 -bgd --verbose -sp &
 
            python ../plot_depth_sharpness_grid.py \
                -i ${views_absdiff} \
                -o ${views_absdiff__grid} \
                -s 11.0 -di -si --ncols 1 -f 8 8 -c auto -cx -dv 0 -sv 0 -bgd --verbose -scm plasma -sp &
 
            python ../plot_snapshots.py \
                -i ${fused_gt} \
                -i ${fused_pred_min} \
                -i ${fused_pred_min_absdiff} \
                -i ${fused_pred_adv60} \
                -i ${fused_pred_adv60_absdiff} \
                -i ${fused_pred_linreg} \
                -i ${fused_pred_linreg_absdiff} \
                -icm plasma_r \
                -icm plasma_r \
                -icm plasma \
                -icm plasma_r \
                -icm plasma \
                -icm plasma_r \
                -icm plasma \
                -o ${fused_snapshot} \
                -s 11.0 -ps 1.25 -ph flat &
 
            python ../compute_metrics.py \
                -t ${views_gt} \
                -p ${views_pred} \
                -r 0.5 -s 10.0 -sv \
                >${views__metrics} &
 
            python ../compute_metrics.py \
                -t ${fused_gt} \
                -p ${fused_pred_adv60} \
                -r 0.5 -s 10.0 \
                >${fused_pred_adv60__metrics} &
 
            python ../compute_metrics.py \
                -t ${fused_gt} \
                -p ${fused_pred_min} \
                -r 0.5 -s 10.0 \
                >${fused_pred_min__metrics} &
 
            python ../compute_metrics.py \
                -t ${fused_gt} \
                -p ${fused_pred_linreg} \
                -r 0.5 -s 10.0 \
                >${fused_pred_linreg__metrics} &

            wait
            /usr/bin/convert \
                ${views_gt__grid} \
                ${views_pred__grid} \
                ${views_absdiff__grid} \
                +append ${views_result__grid}

        done

        wait
 
        awk_compute_mean_std ${output_path_config} __metrics__single_view \
            >${dataset_views__metrics}
 
        awk_compute_mean_std ${output_path_config} __min__metrics \
            >${dataset_fused_pred_min__metrics}
 
        awk_compute_mean_std ${output_path_config} __adv60__metrics \
            >${dataset_fused_pred_adv60__metrics}

        awk_compute_mean_std ${output_path_config} __linreg__metrics \
            >${dataset_fused_pred_linreg__metrics}

    done

done

