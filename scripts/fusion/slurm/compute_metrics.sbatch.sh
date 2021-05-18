#!/bin/bash

#SBATCH --job-name=def-fuse-metrics
#SBATCH --output=/trinity/home/a.artemov/tmp/def-fuse-metrics/%A_%a.out
#SBATCH --error=/trinity/home/a.artemov/tmp/def-fuse-metrics/%A_%a.err
#SBATCH --array=1-1
#SBATCH --time=00:10:00
#SBATCH --partition=htc
#SBATCH --cpus-per-task=1
#SBATCH --ntasks=1
#SBATCH --mem-per-cpu=4g
#SBATCH --oversubscribe
#SBATCH --reservation=SIGGRAPH

__usage="
Usage: $0 [-v] -m method -i input_filename -a

  -v:   if set, verbose mode is activated (more output from the script generally)
  -m:   method which is used as sub-dir where to read predictions and store output
  -i:   input filename with format FILENAME<space>RESOLUTION_3D

Example:
  sbatch $( basename "$0" ) -v -m def -i inputs.txt
"

usage() { echo "$__usage" >&2; }

# Get all the required options and set the necessary variables
VERBOSE=false
AGGREGATE=false
while getopts "vi:m:a" opt
do
    case ${opt} in
        v) VERBOSE=true;;
        a) AGGREGATE=true;;
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

METRICS_SCRIPT="${CODE_PATH_CONTAINER}/scripts/compute_metrics_ruslan.py"
INPUT_BASE_DIR=/gpfs/gpfs0/3ddl/sharp_features/data_v2_cvpr
FUSION_BASE_DIR=/gpfs/gpfs0/3ddl/sharp_features/whole_fused/data_v2_cvpr
FUSION_WRAPPERS_PATH="${PROJECT_ROOT}/scripts/fusion/slurm"
source ${FUSION_WRAPPERS_PATH}/suffix_proba.sh


if [[ ${AGGREGATE} = true ]]
then

  true_arg_v1=""
  pred_arg_v1=""
  true_arg_v2=""
  pred_arg_v2=""
  true_arg_v3=""
  pred_arg_v3=""
  res_3d=""
  while IFS=' ' read -r source_filename resolution_3d; do
    output_path_global="${FUSION_BASE_DIR}/$( realpath --relative-to  ${INPUT_BASE_DIR} "${source_filename%.*}" )/${METHOD}"
    res_3d=${resolution_3d}

    fused_gt="${output_path_global}/$( basename "${source_filename}" .hdf5)${fused_gt_suffix}.hdf5"
    fused_pred_v1="${output_path_global}/$( basename "${source_filename}" .hdf5)${fused_pred_v1_suffix}.hdf5"
    fused_pred_v2="${output_path_global}/$( basename "${source_filename}" .hdf5)${fused_pred_v2_suffix}.hdf5"
    fused_pred_v3="${output_path_global}/$( basename "${source_filename}" .hdf5)${fused_pred_v3_suffix}.hdf5"

    if [[ -f ${fused_pred_v1} ]]; then true_arg_v1="${true_arg_v1} -t ${fused_gt}"; pred_arg_v1="${pred_arg_v1} -p ${fused_pred_v1}"; fi
    if [[ -f ${fused_pred_v2} ]]; then true_arg_v2="${true_arg_v2} -t ${fused_gt}"; pred_arg_v2="${pred_arg_v2} -p ${fused_pred_v2}"; fi
    if [[ -f ${fused_pred_v3} ]]; then true_arg_v3="${true_arg_v3} -t ${fused_gt}"; pred_arg_v3="${pred_arg_v3} -p ${fused_pred_v3}"; fi

  done <"${INPUT_FILENAME:-/dev/stdin}"

  fused_pred_v1__metrics="$( dirname "${INPUT_FILENAME}" )/$( basename "${INPUT_FILENAME}" .hdf5)${fused_pred_v1_suffix}__${METHOD}__metrics.txt"
  fused_pred_v2__metrics="$( dirname "${INPUT_FILENAME}" )/$( basename "${INPUT_FILENAME}" .hdf5)${fused_pred_v2_suffix}__${METHOD}__metrics.txt"
  fused_pred_v3__metrics="$( dirname "${INPUT_FILENAME}" )/$( basename "${INPUT_FILENAME}" .hdf5)${fused_pred_v3_suffix}__${METHOD}__metrics.txt"

else

  # Read SLURM_ARRAY_TASK_ID num lines from standard input,
  # stopping at line whose number equals SLURM_ARRAY_TASK_ID
  res_3d=""
  count=0
  while IFS=' ' read -r source_filename resolution_3d; do
    res_3d=${resolution_3d}
      (( count++ ))
      if (( count == SLURM_ARRAY_TASK_ID )); then
          break
      fi
  done <"${INPUT_FILENAME:-/dev/stdin}"

  output_path_global="${FUSION_BASE_DIR}/$( realpath --relative-to  ${INPUT_BASE_DIR} "${source_filename%.*}" )/${METHOD}"

  fused_gt="${output_path_global}/$( basename "${source_filename}" .hdf5)${fused_gt_suffix}.hdf5"

  fused_pred_v1="${output_path_global}/$( basename "${source_filename}" .hdf5)${fused_pred_v1_suffix}.hdf5"
  fused_pred_v1__metrics="${output_path_global}/$( basename "${source_filename}" .hdf5)${fused_pred_v1_suffix}__metrics.txt"

  fused_pred_v2="${output_path_global}/$( basename "${source_filename}" .hdf5)${fused_pred_v2_suffix}.hdf5"
  fused_pred_v2__metrics="${output_path_global}/$( basename "${source_filename}" .hdf5)${fused_pred_v2_suffix}__metrics.txt"

  fused_pred_v3="${output_path_global}/$( basename "${source_filename}" .hdf5)${fused_pred_v3_suffix}.hdf5"
  fused_pred_v3__metrics="${output_path_global}/$( basename "${source_filename}" .hdf5)${fused_pred_v3_suffix}__metrics.txt"

  true_arg_v1="-t ${fused_gt}"
  pred_arg_v1="-p ${fused_pred_v1}"
  true_arg_v2="-t ${fused_gt}"
  pred_arg_v2="-p ${fused_pred_v2}"
  true_arg_v3="-t ${fused_gt}"
  pred_arg_v3="-p ${fused_pred_v3}"

fi

if [[ -n ${pred_arg_v1} ]]
then
  echo ${fused_pred_v1__metrics}
  singularity exec \
    --bind ${CODE_PATH_HOST}:${CODE_PATH_CONTAINER} \
    --bind "${PWD}":/run/user \
    --bind /gpfs:/gpfs \
    "${SIMAGE_FILENAME}" \
        bash -c 'export OMP_NUM_THREADS='"${OMP_NUM_THREADS}; \\
        python3 ${METRICS_SCRIPT} \\
          ${true_arg_v1} \\
          ${pred_arg_v1} \\
          -o ${fused_pred_v1__metrics} \\
          -r ${res_3d} \\
          ${VERBOSE_ARG} \\
             1> >(tee ${output_path_global}/${SLURM_ARRAY_TASK_ID}.out) \\
             2> >(tee ${output_path_global}/${SLURM_ARRAY_TASK_ID}.err)"
fi

if [[ -n ${pred_arg_v2} ]]
then
  echo ${fused_pred_v2__metrics}
  singularity exec \
    --bind ${CODE_PATH_HOST}:${CODE_PATH_CONTAINER} \
    --bind "${PWD}":/run/user \
    --bind /gpfs:/gpfs \
    "${SIMAGE_FILENAME}" \
        bash -c 'export OMP_NUM_THREADS='"${OMP_NUM_THREADS}; \\
        python3 ${METRICS_SCRIPT} \\
          ${true_arg_v2} \\
          ${pred_arg_v2} \\
          -o ${fused_pred_v2__metrics} \\
          -r ${res_3d} \\
          ${VERBOSE_ARG} \\
             1> >(tee ${output_path_global}/${SLURM_ARRAY_TASK_ID}.out) \\
             2> >(tee ${output_path_global}/${SLURM_ARRAY_TASK_ID}.err)"
fi

if [[ -n ${pred_arg_v3} ]]
then
  echo ${fused_pred_v3__metrics}
  singularity exec \
    --bind ${CODE_PATH_HOST}:${CODE_PATH_CONTAINER} \
    --bind "${PWD}":/run/user \
    --bind /gpfs:/gpfs \
    "${SIMAGE_FILENAME}" \
        bash -c 'export OMP_NUM_THREADS='"${OMP_NUM_THREADS}; \\
        python3 ${METRICS_SCRIPT} \\
          ${true_arg_v3} \\
          ${pred_arg_v3} \\
          -o ${fused_pred_v3__metrics} \\
          -r ${res_3d} \\
          ${VERBOSE_ARG} \\
             1> >(tee ${output_path_global}/${SLURM_ARRAY_TASK_ID}.out) \\
             2> >(tee ${output_path_global}/${SLURM_ARRAY_TASK_ID}.err)"
fi
