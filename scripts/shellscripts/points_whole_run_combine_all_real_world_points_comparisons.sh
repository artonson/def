#!/usr/bin/env bash

set -x
set -e


LOGDIR=/logs
BASE_GT_DIR=${LOGDIR}/whole_models_inference_final/gt/points
BASE_PREDICTIONS_DIR=${LOGDIR}/whole_models_inference_final/predictions/points
COMBINE_SCRIPT=/code/scripts/points_whole_combine_predictions.py
INSTANCE_LIST_FILENAME=${LOGDIR}/whole_models_inference_final/sl_scans_points.txt
BASE_OUTPUT_DIR=${LOGDIR}/whole_models_inference_final/combined/points


run_combine() {
    local model=$1
    local instance_name=$2

    gt_filename=${BASE_GT_DIR}/real_world/${instance_name}.hdf5
    pred_filename=${BASE_PREDICTIONS_DIR}/real_world/real_str/${instance_name}.hdf5
    output_dir=${BASE_OUTPUT_DIR}/real_world/${model}
    output_filename=${BASE_OUTPUT_DIR}/real_world/${model}/${instance_name}__proba_0.75.hdf5

    if [ ! -f ${gt_filename} ] ; then
        echo "GT file ${gt_filename} not available, skipping" && return
    fi
    if [ ! -d ${pred_dir} ] ; then
        echo "PRED dir ${pred_dir} not available, skipping" && return
    fi

    mkdir -p ${output_dir}

    if [ -f ${output_filename} ] ; then
        echo "OUTPUT file ${output_filename} exists, skipping" && return
    fi
    echo "Will compute ${output_filename}"
    echo "python3 ${COMBINE_SCRIPT} -u -k ${model} -t ${gt_filename} -p ${pred_filename} -o ${output_dir} -v 1>${output_filename}.out 2>${output_filename}.err &"

#   python3 ${COMBINE_SCRIPT} \
#       -t ${gt_filename} \
#       -p ${pred_dir} \
#       -o ${output_dir} \
#       -v
}



for instance_name in $( cat ${INSTANCE_LIST_FILENAME} )
do
    for method in voronoi sharpness ecnet
    do
        run_combine ${method} ${instance_name}
    done
done

