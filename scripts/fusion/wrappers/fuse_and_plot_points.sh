#!/bin/bash

# set -x 

# /mnt/ssd/artonson/def_predictions/data_rw_siggraph/images_aligninf_partmesh_whole_testonly/amed
FUSE_SCRIPT=/code/scripts/fusion/fuse_points.py
FUSION_ANALYSIS_SCRIPT=/code/scripts/fusion/fusion_analysis.py
PLOT_SNAPSHOTS_SCRIPT=/code/scripts/plot_snapshots.py
COMPUTE_METRICS_SCRIPT=/code/scripts/compute_metrics.py


DATASETS="points_align4mm_partmesh_whole"

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

for gt_dataset in ${DATASETS}
do
    pred_dataset=${gt_dataset}

    INPUT_DIR_GT=/data/${gt_dataset}/test
    INPUT_DIR_PRED=/logs/${pred_dataset}/med
    OUTPUT_DIR=/logs/${pred_dataset}/amed
    echo "${INPUT_DIR_GT} ${INPUT_DIR_PRED} ${OUTPUT_DIR}"

    output_path_config=${OUTPUT_DIR}/fusion
    mkdir -p ${output_path_config}

    dataset_views__metrics="${output_path_config}/metrics__patches.txt"
    dataset_fused_pred_min__metrics="${output_path_config}/min__metrics__fusion.txt"
    dataset_fused_pred_adv60__metrics="${output_path_config}/adv60__metrics__fusion.txt"
    dataset_fused_pred_linreg__metrics="${output_path_config}/linreg__metrics__fusion.txt"

    for f in $( ls -1 ${INPUT_DIR_GT} )
    do
        patches_gt="/data/${VIEWS_DATASET}/test/${f}"
        patches_pred_dir="${INPUT_DIR_PRED}/$( basename "$f" .hdf5)/predictions/"

         ${FUSE_SCRIPT} \
            -t "${patches_gt}" \
            -p "${patches_pred_dir}" \
            -o ${output_path_config} \
            -j 36 \
            -s 10.0 \
            -r 10.0

        patches_pred="${output_path_config}/$( basename "$f" .hdf5)__predictions.hdf5"
        patches__metrics="${output_path_config}/$( basename "$f" .hdf5)__metrics__patches.txt"

        fused_gt="${output_path_config}/$( basename "$f" .hdf5)__ground_truth.hdf5"

        fused_pred_min="${output_path_config}/$( basename "$f" .hdf5)__min.hdf5"
        fused_pred_min_absdiff="${output_path_config}/$( basename "$f" .hdf5)__min__absdiff.hdf5"
        fused_pred_min__metrics="${output_path_config}/$( basename "$f" .hdf5)__min__metrics.txt"

        fused_pred_adv60="${output_path_config}/$( basename "$f" .hdf5)__adv60.hdf5"
        fused_pred_adv60_absdiff="${output_path_config}/$( basename "$f" .hdf5)__adv60__absdiff.hdf5"
        fused_pred_adv60__metrics="${output_path_config}/$( basename "$f" .hdf5)__adv60__metrics.txt"

        fused_pred_linreg="${output_path_config}/$( basename "$f" .hdf5)__adv60__linreg.hdf5"
        fused_pred_linreg_absdiff="${output_path_config}/$( basename "$f" .hdf5)__linreg__absdiff.hdf5"
        fused_pred_linreg__metrics="${output_path_config}/$( basename "$f" .hdf5)__linreg__metrics.txt"

        fused_snapshot="${output_path_config}/$( basename "$f" .hdf5).html"


        python ${FUSION_ANALYSIS_SCRIPT} \
            -t "${fused_gt}" \
            -p "${fused_pred_min}" \
            -o "${fused_pred_min_absdiff}"
        python ${FUSION_ANALYSIS_SCRIPT} \
            -t "${fused_gt}" \
            -p "${fused_pred_adv60}" \
            -o "${fused_pred_adv60_absdiff}"
        python ${FUSION_ANALYSIS_SCRIPT} \
            -t "${fused_gt}" \
            -p "${fused_pred_linreg}" \
            -o "${fused_pred_linreg_absdiff}"

        python ${PLOT_SNAPSHOTS_SCRIPT} \
            -i "${fused_gt}" \
            -i "${fused_pred_min}" \
            -i "${fused_pred_min_absdiff}" \
            -i "${fused_pred_adv60}" \
            -i "${fused_pred_adv60_absdiff}" \
            -i "${fused_pred_linreg}" \
            -i "${fused_pred_linreg_absdiff}" \
            -icm plasma_r \
            -icm plasma_r \
            -icm plasma \
            -icm plasma_r \
            -icm plasma \
            -icm plasma_r \
            -icm plasma \
            -o "${fused_snapshot}" \
            -s 11.0 -ps 1.25 -ph flat &

        python ${COMPUTE_METRICS_SCRIPT} \
            -t "${patches_gt}" \
            -p "${patches_pred}" \
            -r 0.5 -s 10.0 -sv \
            >"${patches__metrics}" &

        python ${COMPUTE_METRICS_SCRIPT} \
            -t "${fused_gt}" \
            -p "${fused_pred_adv60}" \
            -r 0.5 -s 10.0 \
            >"${fused_pred_adv60__metrics}" &

        python ${COMPUTE_METRICS_SCRIPT} \
            -t "${fused_gt}" \
            -p "${fused_pred_min}" \
            -r 0.5 -s 10.0 \
            >"${fused_pred_min__metrics}" &

        python ${COMPUTE_METRICS_SCRIPT} \
            -t "${fused_gt}" \
            -p "${fused_pred_linreg}" \
            -r 0.5 -s 10.0 \
            >"${fused_pred_linreg__metrics}" &

    done

    wait

    awk_compute_mean_std ${output_path_config} __metrics__patches \
        >${dataset_views__metrics}

    awk_compute_mean_std ${output_path_config} __min__metrics \
        >${dataset_fused_pred_min__metrics}

    awk_compute_mean_std ${output_path_config} __adv60__metrics \
        >${dataset_fused_pred_adv60__metrics}

    awk_compute_mean_std ${output_path_config} __linreg__metrics \
        >${dataset_fused_pred_linreg__metrics}

done

