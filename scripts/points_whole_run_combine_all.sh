#!/usr/bin/env bash

set -x
set -e


LOGDIR=/logs
BASE_GT_DIR=${LOGDIR}/whole_models_inference_final/gt/points
BASE_PREDICTIONS_DIR=${LOGDIR}/whole_models_inference_final/predictions/points
COMBINE_SCRIPT=/code/scripts/points_whole_combine_predictions.py
INSTANCE_LIST_FILENAME=${LOGDIR}/whole_models_inference_final/printed_models.txt
BASE_OUTPUT_DIR=${LOGDIR}/whole_models_inference_final/combined/points


run_combine() {
    local res=$1
    local noise=$2
    local model=$3
    local task=$4
    local instance_name=$5

    gt_filename=${BASE_GT_DIR}/${res}/${noise}/abc_0050_${instance_name}.hdf5
    pred_dir=${BASE_PREDICTIONS_DIR}/${res}/${noise}/${model}/${task}/abc_0050_${instance_name}/predictions
    output_dir=${BASE_OUTPUT_DIR}/${res}/${noise}/${model}/${task}
    output_filename=${BASE_OUTPUT_DIR}/${res}/${noise}/${model}/${task}/abc_0050_${instance_name}__ground_truth.hdf5

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
    echo "python3 ${COMBINE_SCRIPT} -t ${gt_filename} -p ${pred_dir} -o ${output_dir} -v 1>${output_filename}.out 2>${output_filename}.err &"

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
        run_combine ${res} 0.0 d6 regression ${instance_name} ${res}_whole.json
        run_combine ${res} 0.0 d6-v regression ${instance_name} ${res}_whole.json
    done
    for noise in 0.005 0.02 0.08
    do
        run_combine high_res ${noise} d6 regression ${instance_name} high_res_whole_${noise}.json
        run_combine high_res ${noise} d6-v regression ${instance_name} high_res_whole_${noise}.json
    done
done

