#!/bin/bash

#SBATCH --job-name=sharpf-fuse-points-synth
#SBATCH --output=/trinity/home/a.artemov/tmp/sharpf_points/%A_%a.out
#SBATCH --error=/trinity/home/a.artemov/tmp/sharpf_points/%A_%a.err
#SBATCH --array=1-550
#SBATCH --time=6:00:00
#SBATCH --partition=htc
#SBATCH --cpus-per-task=2
#SBATCH --ntasks=1
#SBATCH --mem-per-cpu=6g
#SBATCH --oversubscribe
#SBATCH --reservation=SIGGRAPH

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
PROJECT_ROOT=/trinity/home/a.artemov/repos/sharp_features2
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

N_TASKS=${SLURM_NTASKS}
OMP_NUM_THREADS=${SLURM_CPUS_PER_TASK}

COMBINE_SCRIPT="${CODE_PATH_CONTAINER}/scripts/fusion/fuse_points.py"

# Read SLURM_ARRAY_TASK_ID num lines from standard input,
# stopping at line whole number equals SLURM_ARRAY_TASK_ID
count=0
while IFS=' ' read -r TRUE_FILENAME_GLOBAL PRED_PATH_GLOBAL OUTPUT_PATH_GLOBAL PARAM_RESOLUTION_3D; do
    (( count++ ))
    if (( count == SLURM_ARRAY_TASK_ID )); then
        break
    fi
done <"${1:-/dev/stdin}"


singularity exec \
  --bind ${CODE_PATH_HOST}:${CODE_PATH_CONTAINER} \
  --bind "${PWD}":/run/user \
  --bind /gpfs:/gpfs \
  "${SIMAGE_FILENAME}" \
      bash -c 'export OMP_NUM_THREADS='"${OMP_NUM_THREADS}; \\
      python3 ${COMBINE_SCRIPT} \\
        --true-filename ${TRUE_FILENAME_GLOBAL} \\
        --pred-data ${PRED_PATH_GLOBAL} \\
        --output-dir ${OUTPUT_PATH_GLOBAL} \\
        --jobs ${SLURM_CPUS_PER_TASK} \\
         ${VERBOSE_ARG} \\
           1> >(tee ${LOGS_PATH_CONTAINER}/${CHUNK}_${SLURM_ARRAY_TASK_ID}.out) \\
           2> >(tee ${LOGS_PATH_CONTAINER}/${CHUNK}_${SLURM_ARRAY_TASK_ID}.err)"
