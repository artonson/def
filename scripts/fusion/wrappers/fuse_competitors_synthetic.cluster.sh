#!/bin/bash

set -x

DATASET_FILE_DIR="/gpfs/gpfs0/3ddl/sharp_features/whole_fused"

DATASETS="points_whole__high_res_whole.json.txt"
#DATASETS="points_whole__high_res_whole.json.txt
#points_whole__med_res_whole.json.txt
#points_whole__low_res_whole.json.txt
#points_whole__high_res_whole_0.005.json.txt
#points_whole__high_res_whole_0.02.json.txt
#points_whole__high_res_whole_0.08.json.txt"

METHODS="voronoi
ecnet
sharpness_seg"
#METHODS="sharpness"
#vcm
#ecn
#sfh
#pie"

PROJECT_ROOT=/trinity/home/a.artemov/repos/sharp_features2
FUSION_ANALYSIS_SBATCH="${PROJECT_ROOT}/scripts/fusion/slurm/fusion_analysis.sbatch.sh"
COMPUTE_METRICS_SBATCH="${PROJECT_ROOT}/scripts/fusion/slurm/compute_metrics.sbatch.sh"
PLOT_SNAPSHOTS_SBATCH="${PROJECT_ROOT}/scripts/fusion/slurm/plot_snapshots.sbatch.sh"

run_slurm_jobs() {
  local dataset_file
  local method_dirname
  local task_count

  dataset_file=$1
  method_dirname=$2
  task_count=$3

  # Run fusion analysis to compute __absdiff stuff.
# local WAIT_JOBS
# WAIT_JOBS=$( sbatch \
#   --parsable \
#   --array=1-"${task_count}" \
#   "${FUSION_ANALYSIS_SBATCH}" \
#     -m "${method_dirname}" \
#     -i "${dataset_file}" )
#
# # simply wait for the ${WAIT_JOBS} to complete, doing nothing
# [[ -n "${WAIT_JOBS}" ]] && \
# srun \
#   --dependency afterany:"${WAIT_JOBS}" \
#   sleep 1 >/dev/null 2>&1

  # Compute metrics
  sbatch \
    --parsable \
    --array=1-"${task_count}" \
    "${COMPUTE_METRICS_SBATCH}" \
      -m "${method_dirname}" \
      -i "${dataset_file}" \
      -a 

#  # Draw HTMLs
#  sbatch \
#    --parsable \
#    --array=1-"${task_count}" \
#    "${PLOT_SNAPSHOTS_SBATCH}" \
#      -m "${method_dirname}" \
#      -i "${dataset_file}"
#
}

for dataset in ${DATASETS}; do

  for method in ${METHODS}; do
    task_count=1
    run_slurm_jobs \
      "${DATASET_FILE_DIR}/${dataset}" \
      "${method}" \
      ${task_count}
  done

done
