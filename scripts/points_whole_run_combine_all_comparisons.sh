#!/usr/bin/env bash

set -x
set -e


LOGDIR=/logs
BASE_GT_DIR=${LOGDIR}/whole_models_inference_final/gt/points
BASE_PREDICTIONS_DIR=${LOGDIR}/whole_models_inference_final/predictions/points_comparison
COMBINE_SCRIPT=/code/scripts/points_whole_combine_predictions.py
INSTANCE_LIST_FILENAME=${LOGDIR}/whole_models_inference_final/printed_models.txt
BASE_OUTPUT_DIR=${LOGDIR}/whole_models_inference_final/combined/points


run_combine() {
    local res=$1
    local noise=$2
    local model=$3
    local instance_name=$4

    gt_filename=${BASE_GT_DIR}/${res}/${noise}/abc_0050_${instance_name}.hdf5
    pred_dir=${BASE_PREDICTIONS_DIR}/${res}/${noise}/abc_0050_${instance_name}.hdf5
    output_dir=${BASE_OUTPUT_DIR}/${res}/${noise}/${model}
    # output_filename=${BASE_OUTPUT_DIR}/${res}/${noise}/${model}/${task}/abc_0050_${instance_name}__ground_truth.hdf5
    output_filename=${BASE_OUTPUT_DIR}/${res}/${noise}/${model}/abc_0050_${instance_name}__proba_0.75.hdf5

    if [ ! -f ${gt_filename} ] ; then
        echo "GT file ${gt_filename} not available, skipping" && return
    fi
    if [ ! -f ${pred_filename} ] ; then
        echo "PRED dir ${pred_filename} not available, skipping" && return
    fi

    mkdir -p ${output_dir}

    if [ -f ${output_filename} ] ; then
        echo "OUTPUT file ${output_filename} exists, skipping" && return
    fi
    echo "Will compute ${output_filename}"
    echo "python3 ${COMBINE_SCRIPT} -k ${model} -t ${gt_filename} -p ${pred_dir} -o ${output_dir} -v 1>${output_filename}.out 2>${output_filename}.err &"

#   python3 ${COMBINE_SCRIPT} \
#       -t ${gt_filename} \
#       -p ${pred_dir} \
#       -o ${output_dir} \
#       -v
}



for instance_name in $( cat ${INSTANCE_LIST_FILENAME} )
do
    for res in high_res med_res low_res
    do
        for method in voronoi sharpness ecnet
        do
            run_combine ${res} 0.0 ${method} ${instance_name}
        done
    done
    for noise in 0.005 0.02 0.08
    do
        for method in voronoi sharpness ecnet
        do
            run_combine high_res ${noise} ${method} ${instance_name}
        done
    done
done

