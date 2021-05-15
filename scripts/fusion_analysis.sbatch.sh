#!/bin/bash

#SBATCH --job-name=def-fuse-analysis
#SBATCH --output=/trinity/home/a.artemov/tmp/def-fuse-analysis/%A_%a.out
#SBATCH --error=/trinity/home/a.artemov/tmp/def-fuse-analysis/%A_%a.err
#SBATCH --array=1-1
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

FUSION_ANALYSIS_SCRIPT="${CODE_PATH_CONTAINER}/scripts/fusion/fusion_analysis.py"

# Read SLURM_ARRAY_TASK_ID num lines from standard input,
# stopping at line whole number equals SLURM_ARRAY_TASK_ID
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

fused_gt="${output_path_global}/$( basename "${source_filename}" .hdf5)__ground_truth.hdf5"

fused_pred_min="${output_path_global}/$( basename "${source_filename}" .hdf5)__min.hdf5"
fused_pred_min_absdiff="${output_path_global}/$( basename "${source_filename}" .hdf5)__min__absdiff.hdf5"

fused_pred_adv60="${output_path_global}/$( basename "${source_filename}" .hdf5)__adv60__min.hdf5"
fused_pred_adv60_absdiff="${output_path_global}/$( basename "${source_filename}" .hdf5)__adv60__absdiff.hdf5"

fused_pred_linreg="${output_path_global}/$( basename "${source_filename}" .hdf5)__adv60__min__linreg.hdf5"
fused_pred_linreg_absdiff="${output_path_global}/$( basename "${source_filename}" .hdf5)__linreg__absdiff.hdf5"

if [[ -f ${fused_pred_min} ]]
then
  echo ${fused_pred_min_absdiff}
  singularity exec \
    --bind ${CODE_PATH_HOST}:${CODE_PATH_CONTAINER} \
    --bind "${PWD}":/run/user \
    --bind /gpfs:/gpfs \
    "${SIMAGE_FILENAME}" \
        bash -c 'export OMP_NUM_THREADS='"${OMP_NUM_THREADS}; \\
        python3 ${FUSION_ANALYSIS_SCRIPT} \\
          -t ${fused_gt} \\
          -p ${fused_pred_min} \\
          -o ${fused_pred_min_absdiff} \\
             1> >(tee ${output_path_global}/${SLURM_ARRAY_TASK_ID}.out) \\
             2> >(tee ${output_path_global}/${SLURM_ARRAY_TASK_ID}.err)"
fi

if [[ -f ${fused_pred_adv60} ]]
then
  echo ${fused_pred_adv60_absdiff}
  singularity exec \
    --bind ${CODE_PATH_HOST}:${CODE_PATH_CONTAINER} \
    --bind "${PWD}":/run/user \
    --bind /gpfs:/gpfs \
    "${SIMAGE_FILENAME}" \
        bash -c 'export OMP_NUM_THREADS='"${OMP_NUM_THREADS}; \\
        python3 ${FUSION_ANALYSIS_SCRIPT} \\
          -t ${fused_gt} \\
          -p ${fused_pred_adv60} \\
          -o ${fused_pred_adv60_absdiff} \\
             1> >(tee ${output_path_global}/${SLURM_ARRAY_TASK_ID}.out) \\
             2> >(tee ${output_path_global}/${SLURM_ARRAY_TASK_ID}.err)"
fi

if [[ -f ${fused_pred_linreg} ]]
then
  echo ${fused_pred_linreg_absdiff}
  singularity exec \
    --bind ${CODE_PATH_HOST}:${CODE_PATH_CONTAINER} \
    --bind "${PWD}":/run/user \
    --bind /gpfs:/gpfs \
    "${SIMAGE_FILENAME}" \
        bash -c 'export OMP_NUM_THREADS='"${OMP_NUM_THREADS}; \\
        python3 ${FUSION_ANALYSIS_SCRIPT} \\
          -t ${fused_gt} \\
          -p ${fused_pred_linreg} \\
          -o ${fused_pred_linreg_absdiff} \\
             1> >(tee ${output_path_global}/${SLURM_ARRAY_TASK_ID}.out) \\
             2> >(tee ${output_path_global}/${SLURM_ARRAY_TASK_ID}.err)"
fi
