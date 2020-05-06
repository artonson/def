#!/bin/bash

#SBATCH --job-name=sharpf-data
#SBATCH --output=array_%A_%a.out
#SBATCH --error=array_%A_%a.err
#SBATCH --array=1-80
#SBATCH --time=24:00:00
#SBATCH --partition=htc
#SBATCH --cpus-per-task=4
#SBATCH --ntasks=1
#SBATCH --mem-per-cpu=8g
#SBATCH --oversubscribe

module load apps/singularity-3.2.0

__usage="
Usage: $0 -c chunk -o output_dir -d data_dir -l logs_dir -f config_file [-v]

  -c:   zero-based chunk identifier
  -o: 	output directory where patches will be written
  -d: 	input data directory
  -l:   server logs dir
  -f:   dataset config file (from scripts/data_scripts/configs/pointcloud_datasets dir
  -v:   if set, verbose mode is activated (more output from the script generally)

Example:
sbatch make_patches.sbatch.sh
  -d /gpfs/gpfs0/3ddl/datasets/abc \\
  -o /gpfs/gpfs0/3ddl/datasets/abc/eccv  \\
  -l /home/artonson/tmp/logs  \\
  -v
"

usage() { echo "$__usage" >&2; }

# Get all the required options and set the necessary variables
VERBOSE=false
while getopts "c:o:d:l:f:v" opt
do
    case ${opt} in
        c) CHUNK=$OPTARG;;
        o) OUTPUT_PATH_HOST=$OPTARG;;
        d) DATA_PATH_HOST=$OPTARG;;
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

if [[ ! ${CHUNK} ]]; then
    echo "chunk is not set" && usage && exit 1
fi

OUTPUT_PATH_CONTAINER="/out"
if [[ ! ${OUTPUT_PATH_HOST} ]]; then
    echo "output_dir is not set" && usage && exit 1
fi

DATA_PATH_CONTAINER="/data"
if [[ ! ${DATA_PATH_HOST} ]]; then
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
echo "  data path:            ${DATA_PATH_HOST}"
echo "  code path:            ${CODE_PATH_HOST}"
echo "  logs path:            ${LOGS_PATH_HOST}"
echo "  output path:          ${OUTPUT_PATH_HOST}"
echo "  "
echo "  CONTAINER OPTIONS:"
echo "  data path:            ${DATA_PATH_CONTAINER}"
echo "  code path:            ${CODE_PATH_CONTAINER}"
echo "  logs path:            ${LOGS_PATH_CONTAINER}"
echo "  output path:          ${OUTPUT_PATH_CONTAINER}"
echo "  "

N_TASKS=${SLURM_NTASKS}
OMP_NUM_THREADS=${SLURM_CPUS_PER_TASK}
MAKE_DATA_SCRIPT="${CODE_PATH_CONTAINER}/scripts/data_scripts/generate_pointcloud_data.py"
PC_CONFIGS_PATH_CONTAINER="${CODE_PATH_CONTAINER}/scripts/data_scripts/configs/pointcloud_datasets"
DATASET_PATH="${PC_CONFIGS_PATH_CONTAINER}/${DATASET_CONFIG}"

CHUNK_SIZE=100
SLICE_START=$(( ${CHUNK_SIZE} * ${SLURM_ARRAY_TASK_ID} ))
SLICE_END=$(( ${CHUNK_SIZE} * (${SLURM_ARRAY_TASK_ID} + 1) ))
echo "SLURM_ARRAY_TASK_ID=${SLURM_ARRAY_TASK_ID} SLICE_START=${SLICE_START} SLICE_END=${SLICE_END}"

singularity exec \
  --bind ${CODE_PATH_HOST}:${CODE_PATH_CONTAINER} \
  --bind ${DATA_PATH_HOST}:${DATA_PATH_CONTAINER} \
  --bind ${LOGS_PATH_HOST}:${LOGS_PATH_CONTAINER} \
  --bind ${OUTPUT_PATH_HOST}:${OUTPUT_PATH_CONTAINER} \
  --bind "${PWD}":/run/user \
  "${SIMAGE_FILENAME}" \
      bash -c 'export OMP_NUM_THREADS='"${OMP_NUM_THREADS}; \\
      python3 ${MAKE_DATA_SCRIPT} \\
        --input-dir ${DATA_PATH_CONTAINER} \\
        --chunk ${CHUNK} \\
        --output-dir ${OUTPUT_PATH_CONTAINER} \\
        --jobs ${N_TASKS} \\
        -n1 ${SLICE_START} -n2 ${SLICE_END} \\
        --dataset-config ${DATASET_PATH} \\
         ${VERBOSE_ARG} \\
           1> >(tee ${LOGS_PATH_CONTAINER}/${CHUNK}_${SLURM_ARRAY_TASK_ID}.out) \\
           2> >(tee ${LOGS_PATH_CONTAINER}/${CHUNK}_${SLURM_ARRAY_TASK_ID}.err)"

