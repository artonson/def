#!/bin/bash

#SBATCH --job-name=sharpf-fuse-image-synth
#SBATCH --output=/trinity/home/e.bogomolov/tmp/sharpf_images/%A_%a.out
#SBATCH --error=/trinity/home/e.bogomolov/tmp/sharpf_images/%A_%a.err
#SBATCH --array=1-550
#SBATCH --time=00:10:00
#SBATCH --partition=htc
#SBATCH --cpus-per-task=1
#SBATCH --ntasks=1
#SBATCH --mem-per-cpu=4g
#SBATCH --oversubscribe

__usage="
Usage: $0 [-v] <[input_filename]

  -v:   if set, verbose mode is activated (more output from the script generally)

Example:
  sbatch $( basename "$0" ) -v <inputs.txt
"

usage() { echo "$__usage" >&2; }

# Get all the required options and set the necessary variables
VERBOSE=false
while getopts "v" opt
do
    case ${opt} in
        v) VERBOSE=true;;
        *) usage; exit 1 ;;
    esac
done

if [[ "${VERBOSE}" = true ]]; then
    set -x
    VERBOSE_ARG="--verbose"
fi

# get image filenames from here
PROJECT_ROOT=/trinity/home/e.bogomolov/data_quality
source "${PROJECT_ROOT}"/env.sh

CODE_PATH_CONTAINER="/code"
CODE_PATH_HOST=${PROJECT_ROOT}

echo "******* LAUNCHING IMAGE ${SIMAGE_FILENAME} *******"
echo "  "
echo "  HOST OPTIONS:"
echo "  code path:            ${CODE_PATH_HOST}"
echo "  "
echo "  CONTAINER OPTIONS:"
echo "  code path:            ${CODE_PATH_CONTAINER}"
echo "  "

OMP_NUM_THREADS=${SLURM_CPUS_PER_TASK}

METRICS_SCRIPT="${CODE_PATH_CONTAINER}/scripts/compute_metrics.py"

# Read SLURM_ARRAY_TASK_ID num lines from standard input,
# stopping at line whose number equals SLURM_ARRAY_TASK_ID
count=0
while IFS=' ' read -r source_filename resolution_3d; do
    (( count++ ))
    if (( count == SLURM_ARRAY_TASK_ID )); then
        break
    fi
done <"${1:-/dev/stdin}"

INPUT_BASE_DIR=/gpfs/gpfs0/3ddl/sharp_features/data_v2_cvpr
FUSION_BASE_DIR=/gpfs/gpfs0/3ddl/sharp_features/whole_fused/data_v2_cvpr
output_path_global="${FUSION_BASE_DIR}/$( realpath --relative-to  ${INPUT_BASE_DIR} "${source_filename}" )"

fused_gt="${output_path_global}/$( basename "${source_filename}" .hdf5)__ground_truth.hdf5"

fused_pred_min="${output_path_global}/$( basename "${source_filename}" .hdf5)__min.hdf5"
fused_pred_min__metrics="${output_path_global}/$( basename "${source_filename}" .hdf5)__min__metrics.txt"

fused_pred_adv60="${output_path_global}/$( basename "${source_filename}" .hdf5)__adv60__min.hdf5"
fused_pred_adv60__metrics="${output_path_global}/$( basename "${source_filename}" .hdf5)__adv60__metrics.txt"

fused_pred_linreg="${output_path_global}/$( basename "${source_filename}" .hdf5)__adv60__min__linreg.hdf5"
fused_pred_linreg__metrics="${output_path_global}/$( basename "${source_filename}" .hdf5)__linreg__metrics.txt"

if [[ -f ${fused_pred_min} ]]
then
  singularity exec \
    --bind ${CODE_PATH_HOST}:${CODE_PATH_CONTAINER} \
    --bind "${PWD}":/run/user \
    --bind /gpfs:/gpfs \
    "${SIMAGE_FILENAME}" \
        bash -c 'export OMP_NUM_THREADS='"${OMP_NUM_THREADS}; \\
        python3 ${METRICS_SCRIPT} \\
          -t ${fused_gt} \\
          -p ${fused_pred_min} \\
          -o ${fused_pred_min__metrics} \\
          -r ${resolution_3d} \\
          ${VERBOSE_ARG} \\
             1> >(tee ${output_path_global}/${SLURM_ARRAY_TASK_ID}.out) \\
             2> >(tee ${output_path_global}/${SLURM_ARRAY_TASK_ID}.err)"
fi

if [[ -f ${fused_pred_adv60} ]]
then
  singularity exec \
    --bind ${CODE_PATH_HOST}:${CODE_PATH_CONTAINER} \
    --bind "${PWD}":/run/user \
    --bind /gpfs:/gpfs \
    "${SIMAGE_FILENAME}" \
        bash -c 'export OMP_NUM_THREADS='"${OMP_NUM_THREADS}; \\
        python3 ${METRICS_SCRIPT} \\
          -t ${fused_gt} \\
          -p ${fused_pred_adv60} \\
          -o ${fused_pred_adv60__metrics} \\
          -r ${resolution_3d} \\
          ${VERBOSE_ARG} \\
             1> >(tee ${output_path_global}/${SLURM_ARRAY_TASK_ID}.out) \\
             2> >(tee ${output_path_global}/${SLURM_ARRAY_TASK_ID}.err)"
fi

if [[ -f ${fused_pred_linreg} ]]
then
  singularity exec \
    --bind ${CODE_PATH_HOST}:${CODE_PATH_CONTAINER} \
    --bind "${PWD}":/run/user \
    --bind /gpfs:/gpfs \
    "${SIMAGE_FILENAME}" \
        bash -c 'export OMP_NUM_THREADS='"${OMP_NUM_THREADS}; \\
        python3 ${METRICS_SCRIPT} \\
          -t ${fused_gt} \\
          -p ${fused_pred_linreg} \\
          -o ${fused_pred_linreg__metrics} \\
          -r ${resolution_3d} \\
          ${VERBOSE_ARG} \\
             1> >(tee ${output_path_global}/${SLURM_ARRAY_TASK_ID}.out) \\
             2> >(tee ${output_path_global}/${SLURM_ARRAY_TASK_ID}.err)"
fi
