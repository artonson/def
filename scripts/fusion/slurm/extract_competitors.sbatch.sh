#!/bin/bash

#SBATCH --job-name=def-extract-competitors
#SBATCH --output=/trinity/home/a.artemov/tmp/def-extract-competitors/%A_%a.out
#SBATCH --error=/trinity/home/a.artemov/tmp/def-extract-competitors/%A_%a.err
#SBATCH --array=1-1
#SBATCH --time=00:10:00
#SBATCH --partition=htc
#SBATCH --cpus-per-task=1
#SBATCH --ntasks=1
#SBATCH --mem-per-cpu=4g

__usage="
Usage: $0 [-v] -m method -i input_filename

  -v:   if set, verbose mode is activated (more output from the script generally)
  -m:   method which is used as sub-dir where to read predictions and store output
  -i:   input filename with format FILENAME<space>RESOLUTION_3D

Example:
  sbatch $( basename "$0" ) -v -m def -i inputs.txt -o output_base_dir
"

usage() { echo "$__usage" >&2; }

set -x
# Get all the required options and set the necessary variables
VERBOSE=false
while getopts "vi:m:o:" opt
do
    case ${opt} in
        v) VERBOSE=true;;
        i) INPUT_FILENAME=$OPTARG;;
        m) METHOD=$OPTARG;;
        o) OUTPUT_BASE_DIR=$OPTARG;;
        *) usage; exit 1 ;;
    esac
done

if [[ "${VERBOSE}" = true ]]; then
    set -x
    VERBOSE_ARG="--verbose"
fi

if [[ ! ${OUTPUT_BASE_DIR} ]]; then
    echo "output_base_dir is not set" && usage && exit 1
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

SCRIPT="${CODE_PATH_CONTAINER}/scripts/fusion/extract_competitor_from_predictions.py"

# Read SLURM_ARRAY_TASK_ID num lines from standard input,
# stopping at line whole number equals SLURM_ARRAY_TASK_ID
count=0
while IFS=' ' read -r input_filename; do
    (( count++ ))
    if (( count == SLURM_ARRAY_TASK_ID )); then
        break
    fi
done <"${INPUT_FILENAME:-/dev/stdin}"

# /path/to/output/<shape-id>/<method>/<shape-id>__fused.hdf5
output_path_global="${OUTPUT_BASE_DIR}/$(basename "${input_filename%.*}")/${METHOD}"
mkdir -p ${output_path_global}
output_filename=${output_path_global}/$( basename "${input_filename%.*}__fused.hdf5" )

echo "${input_filename} ${output_filename}"

singularity exec \
  --bind ${CODE_PATH_HOST}:${CODE_PATH_CONTAINER} \
  --bind "${PWD}":/run/user \
  --bind /gpfs:/gpfs \
  "${SIMAGE_FILENAME}" \
      bash -c "python3 ${SCRIPT} \\
        -i ${input_filename} \\
        -o ${output_filename} \\
        -k ${METHOD} \\
        ${VERBOSE_ARG} \\
           1> >(tee ${output_path_global}/${SLURM_ARRAY_TASK_ID}.out) \\
           2> >(tee ${output_path_global}/${SLURM_ARRAY_TASK_ID}.err)"
