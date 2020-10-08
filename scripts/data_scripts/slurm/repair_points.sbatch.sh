#!/bin/bash

#SBATCH --job-name=sharpf-repair
#SBATCH --output=logs/repair_%A.out
#SBATCH --error=logs/repair_%A.err
#SBATCH --time=12:00:00
#SBATCH --partition=htc
#SBATCH --cpus-per-task=32
#SBATCH --ntasks=1
#SBATCH --mem-per-cpu=10g
#SBATCH --oversubscribe

# module load apps/singularity-3.2.0

__usage="
Usage: $0 -i input_filename -o output_dir -a data_dir -l logs_dir -f config_file [-v]"

usage() { echo "$__usage" >&2; }

# Get all the required options and set the necessary variables
VERBOSE=false
while getopts "i:o:a:l:f:v" opt
do
    case ${opt} in
        i) INPUT_FILE_HOST=$OPTARG;;
        o) OUTPUT_PATH_HOST=$OPTARG;;
        d) ABC_PATH_HOST=$OPTARG;;
        l) LOGS_PATH_HOST=$OPTARG;;
        f) DATASET_CONFIG=$OPTARG;;
        v) VERBOSE=true;;
        *) usage; exit 1 ;;
    esac
done

if [[ "${VERBOSE}" = true ]]; then
    set -x
    VERBOSE_ARG="--verbose"
fi

# get image filenames from here
PROJECT_ROOT=/trinity/home/a.artemov/repos/sharp_features
source "${PROJECT_ROOT}"/env.sh

if [[ ! ${DATASET_CONFIG} ]]; then
    echo "config_file is not set" && usage && exit 1
fi

INPUT_PATH_CONTAINER="/in"
if [[ ! ${INPUT_FILE_HOST} ]]; then
    echo "input_filename is not set" && usage && exit 1
fi
INPUT_PATH_HOST=$( dirname ${INPUT_FILE_HOST})

OUTPUT_PATH_CONTAINER="/out"
if [[ ! ${OUTPUT_PATH_HOST} ]]; then
    echo "output_dir is not set" && usage && exit 1
fi

ABC_PATH_CONTAINER="/data"
if [[ ! ${ABC_PATH_HOST} ]]; then
    echo "data_dir is not set" && usage && exit 1
fi

LOGS_PATH_CONTAINER="/logs"
if [[ ! ${LOGS_PATH_HOST} ]]; then
    echo "logs_dir is not set" && usage && exit 1
fi

CODE_PATH_CONTAINER="/code"
CODE_PATH_HOST=${PROJECT_ROOT}

echo "******* LAUNCHING IMAGE ${SIMAGE_FILENAME} *******"
echo "  "
echo "  HOST OPTIONS:"
echo "  data path:            ${ABC_PATH_HOST}"
echo "  code path:            ${CODE_PATH_HOST}"
echo "  logs path:            ${LOGS_PATH_HOST}"
echo "  input path:           ${INPUT_PATH_HOST}"
echo "  output path:          ${OUTPUT_PATH_HOST}"
echo "  "
echo "  CONTAINER OPTIONS:"
echo "  data path:            ${ABC_PATH_CONTAINER}"
echo "  code path:            ${CODE_PATH_CONTAINER}"
echo "  logs path:            ${LOGS_PATH_CONTAINER}"
echo "  input path:           ${INPUT_PATH_CONTAINER}"
echo "  output path:          ${OUTPUT_PATH_CONTAINER}"
echo "  "

SCRIPT="${CODE_PATH_CONTAINER}/scripts/data_scripts/repair_curves_annotations.py"
CONFIGS_PATH_CONTAINER="${CODE_PATH_CONTAINER}/scripts/data_scripts/configs/pointcloud_datasets"
CONFIG_PATH="${CONFIGS_PATH_CONTAINER}/${DATASET_CONFIG}"

INPUT_FILENAME=$( basename ${INPUT_FILE_GPFS} )

singularity exec \
  --bind ${CODE_PATH_HOST}:${CODE_PATH_CONTAINER} \
  --bind ${ABC_PATH_HOST}:${ABC_PATH_CONTAINER} \
  --bind ${LOGS_PATH_HOST}:${LOGS_PATH_CONTAINER} \
  --bind ${INPUT_PATH_HOST}:${INPUT_PATH_CONTAINER} \
  --bind ${OUTPUT_PATH_HOST}:${OUTPUT_PATH_CONTAINER} \
  --bind "${PWD}":/run/user \
  "${SIMAGE_FILENAME}" \
      bash -c "python3 ${SCRIPT} \\
        --abc-dir ${ABC_PATH_CONTAINER} \\
        --chunk 0 --chunk 1 \\
        --input-dir ${INPUT_PATH_CONTAINER}/${INPUT_FILENAME} \\
        --output-dir ${OUTPUT_PATH_CONTAINER} \\
        --dataset-config ${CONFIG_PATH} \\
        --jobs ${SLURM_CPUS_PER_TASK} \\
        --verbose
           1> >(tee ${LOGS_PATH_CONTAINER}/${INPUT_FILENAME}.out) \\
           2> >(tee ${LOGS_PATH_CONTAINER}/${INPUT_FILENAME}.err)"
