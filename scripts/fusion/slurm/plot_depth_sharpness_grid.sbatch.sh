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
#SBATCH --reservation=SIGGRAPH

__usage="
Usage: $0 [-v] -m method -i input_filename

  -v:   if set, verbose mode is activated (more output from the script generally)
  -m:   method which is used as sub-dir where to read predictions and store output
  -i:   input filename with format FILENAME<space>RESOLUTION_3D

Example:
  sbatch $( basename "$0" ) -v -m def -i inputs.txt
"

usage() { echo "$__usage" >&2; }

# Get all the required options and set the necessary variables
VERBOSE=false
while getopts "vi:m:" opt
do
    case ${opt} in
        v) VERBOSE=true;;
        i) INPUT_FILENAME=$OPTARG;;
        m) METHOD=$OPTARG;;
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
done <"${INPUT_FILENAME:-/dev/stdin}"

INPUT_BASE_DIR=/gpfs/gpfs0/3ddl/sharp_features/data_v2_cvpr
FUSION_BASE_DIR=/gpfs/gpfs0/3ddl/sharp_features/whole_fused/data_v2_cvpr
output_path_global="${FUSION_BASE_DIR}/$( realpath --relative-to  ${INPUT_BASE_DIR} "${source_filename%.*}" )/${METHOD}"

views_gt=${source_filename}
views_gt__grid="${output_path_global}/$( basename "${source_filename}" .hdf5)__ground_truth.png"
views_pred="${output_path_global}/$( basename "${source_filename}" .hdf5)__predictions.hdf5"
views_pred__grid="${output_path_global}/$( basename "${source_filename}" .hdf5)__predictions.png"
views_absdiff="${output_path_global}/$( basename "${source_filename}" .hdf5)__absdiff.hdf5"
views_absdiff__grid="${output_path_global}/$( basename "${source_filename}" .hdf5)__absdiff.png"
views_result__grid="${output_path_global}/$( basename "${source_filename}" .hdf5)__result.png"


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

singularity exec \
  --bind ${CODE_PATH_HOST}:${CODE_PATH_CONTAINER} \
  --bind "${PWD}":/run/user \
  --bind /gpfs:/gpfs \
  "${SIMAGE_FILENAME}" \
      bash -c 'export OMP_NUM_THREADS='"${OMP_NUM_THREADS}; \\
      python3 ${PLOT_SCRIPT} \\
        -i ${views_pred} \
        -o ${views_pred__grid} \
        -s ${MAX_DISTANCE} \\
        -di -si -sp \\
        --ncols 1 \\
        -f 8 8 \\
        -c auto -cx \\
        -dv 0 -sv 0 -bgd \\
        ${VERBOSE_ARG} \\
           1> >(tee ${output_path_global}/${SLURM_ARRAY_TASK_ID}.out) \\
           2> >(tee ${output_path_global}/${SLURM_ARRAY_TASK_ID}.err)"

singularity exec \
  --bind ${CODE_PATH_HOST}:${CODE_PATH_CONTAINER} \
  --bind "${PWD}":/run/user \
  --bind /gpfs:/gpfs \
  "${SIMAGE_FILENAME}" \
      bash -c 'export OMP_NUM_THREADS='"${OMP_NUM_THREADS}; \\
      python3 ${PLOT_SCRIPT} \\
        -i ${views_absdiff} \
        -o ${views_absdiff__grid} \
        -s ${MAX_DISTANCE} \\
        -di -si -sp \\
        --ncols 1 \\
        -f 8 8 \\
        -c auto -cx \\
        -dv 0 -sv 0 -bgd \\
        -scm plasma \\
        ${VERBOSE_ARG} \\
           1> >(tee ${output_path_global}/${SLURM_ARRAY_TASK_ID}.out) \\
           2> >(tee ${output_path_global}/${SLURM_ARRAY_TASK_ID}.err)"

wait
MAGICK_IMAGE=/trinity/shared/singularity-images/ImageMagick7.simg
MAGICK_CONVERT=/usr/local/bin/convert
singularity exec \
  --bind /gpfs:/gpfs \
  ${MAGICK_IMAGE} \
  ${MAGICK_CONVERT} \
    "${views_gt__grid}" \
    "${views_pred__grid}" \
    "${views_absdiff__grid}" \
    +append "${views_result__grid}"
