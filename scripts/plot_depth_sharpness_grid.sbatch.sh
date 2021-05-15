#!/bin/bash

#SBATCH --job-name=def-plot-dsg
#SBATCH --output=/trinity/home/a.artemov/tmp/def-plot-dsg/%A_%a.out
#SBATCH --error=/trinity/home/a.artemov/tmp/def-plot-dsg/%A_%a.err
#SBATCH --array=1-1
#SBATCH --time=00:10:00
#SBATCH --partition=htc
#SBATCH --cpus-per-task=1
#SBATCH --ntasks=1
#SBATCH --mem-per-cpu=4g
#SBATCH --oversubscribe

__usage="
Usage: $0 [-v] [input_filename]

  -v:   if set, verbose mode is activated (more output from the script generally)

Example:
  sbatch $( basename "$0" ) -v inputs.txt
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

set -x
if [[ "${VERBOSE}" = true ]]; then
    set -x
    VERBOSE_ARG="--verbose"
fi

# get image filenames from here
PROJECT_ROOT=/trinity/home/a.artemov/repos/sharp_features
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

PLOT_SCRIPT="${CODE_PATH_CONTAINER}/scripts/plot_depth_sharpness_grid.py"

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
output_path_global="${FUSION_BASE_DIR}/$( realpath --relative-to  ${INPUT_BASE_DIR} "${source_filename%.*}" )"

views_gt=${source_filename}
views_gt__grid="${output_path_global}/$( basename "${source_filename}" .hdf5)__ground_truth.png"

MAX_DISTANCE=1.1
singularity exec \
  --bind ${CODE_PATH_HOST}:${CODE_PATH_CONTAINER} \
  --bind "${PWD}":/run/user \
  --bind /gpfs:/gpfs \
  "${SIMAGE_FILENAME}" \
      bash -c 'export OMP_NUM_THREADS='"${OMP_NUM_THREADS}; \\
      python3 ${PLOT_SCRIPT} \\
        -i ${views_gt} \
        -o ${views_gt__grid} \
        -s ${MAX_DISTANCE} \\
        -di -si -dp -sp \\
        --ncols 1 \\
        -f 8 8 \\
        -c auto -cx \\
        -dv 0 -sv 0 -bgd \\
        ${VERBOSE_ARG} \\
           1> >(tee ${output_path_global}/${SLURM_ARRAY_TASK_ID}.out) \\
           2> >(tee ${output_path_global}/${SLURM_ARRAY_TASK_ID}.err)"


#if [[ -f ${fused_pred_min} ]]
#then
#  echo ${fused_pred_min__metrics}
#fi
#
#if [[ -f ${fused_pred_adv60} ]]
#then
#  echo ${fused_pred_adv60__metrics}
#  singularity exec \
#    --bind ${CODE_PATH_HOST}:${CODE_PATH_CONTAINER} \
#    --bind "${PWD}":/run/user \
#    --bind /gpfs:/gpfs \
#    "${SIMAGE_FILENAME}" \
#        bash -c 'export OMP_NUM_THREADS='"${OMP_NUM_THREADS}; \\
#        python3 ${PLOT_SCRIPT} \\
#          -t ${fused_gt} \\
#          -p ${fused_pred_adv60} \\
#          -o ${fused_pred_adv60__metrics} \\
#          -r ${resolution_3d} \\
#          ${VERBOSE_ARG} \\
#             1> >(tee ${output_path_global}/${SLURM_ARRAY_TASK_ID}.out) \\
#             2> >(tee ${output_path_global}/${SLURM_ARRAY_TASK_ID}.err)"
#fi
#
#if [[ -f ${fused_pred_linreg} ]]
#then
#  echo ${fused_pred_linreg__metrics}
#  singularity exec \
#    --bind ${CODE_PATH_HOST}:${CODE_PATH_CONTAINER} \
#    --bind "${PWD}":/run/user \
#    --bind /gpfs:/gpfs \
#    "${SIMAGE_FILENAME}" \
#        bash -c 'export OMP_NUM_THREADS='"${OMP_NUM_THREADS}; \\
#        python3 ${PLOT_SCRIPT} \\
#          -t ${fused_gt} \\
#          -p ${fused_pred_linreg} \\
#          -o ${fused_pred_linreg__metrics} \\
#          -r ${resolution_3d} \\
#          ${VERBOSE_ARG} \\
#             1> >(tee ${output_path_global}/${SLURM_ARRAY_TASK_ID}.out) \\
#             2> >(tee ${output_path_global}/${SLURM_ARRAY_TASK_ID}.err)"
#fi
